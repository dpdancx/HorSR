from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import torch.fft
from basicsr.archs import block as B
from basicsr.utils.registry import ARCH_REGISTRY
from torchsummary import summary
from torchstat import stat
from thop import profile
from ptflops import get_model_complexity_info


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x





class GlobalFilter(nn.Module):
    def __init__(self, dim=64, w_dim=4, h=48, w=25, scale=1/3):
        super().__init__()
        self.scale = scale
        self.w_dim = w_dim
        self.complex_weight = nn.Parameter(torch.randn(w_dim, h, w, 2, dtype=torch.float32) * 0.02, requires_grad=True)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1 = torch.chunk(x, self.w_dim, dim=1)
        w = torch.chunk(self.complex_weight, self.w_dim, dim=0)
        x_sum = []

        for i in range(self.w_dim):
            x2 = x1[i].to(torch.float32)
            B, C, a, b = x2.shape
            w_i = w[i]
            x3 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')
            if not w_i.shape[1:3] == x2.shape[2:4]:
                w_i = F.interpolate(w_i.permute(3, 0, 1, 2), size=x3.shape[2:4], mode='bilinear', align_corners=True).permute(1, 2, 3, 0)
            w_i = torch.view_as_complex(w_i.contiguous())
            x3 = x3 * w_i
            x4 = torch.fft.irfft2(x3, s=(a, b), dim=(2, 3), norm='ortho')
            x_sum.append(x4)
        out = torch.cat(x_sum[:], dim=1)

        out = self.post_norm(out)
        return out




class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s
        # print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]

        x = self.proj_out(x)
        x = x*self.scale
        return x


class Block(nn.Module):
    r""" HorNet block
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)  # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x




@ARCH_REGISTRY.register()
class HorSR(nn.Module):
    def __init__(self, in_chans=3, out_chans=3,
                 depths=[1, 2, 3, 2], base_dim=64, drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 gnconv=[partial(gnconv, order=2, s=1/3),
                         partial(gnconv, order=3, s=1/3),
                         partial(gnconv, order=4, s=1/3),
                         partial(gnconv, order=5, s=1/3)], block=Block, uniform_init=False,  upscale=4, **kwargs):
        super(HorSR, self).__init__()
        self.stem = conv_layer(in_channels=in_chans, out_channels=base_dim, kernel_size=3)
        self.GFF = GlobalFilter()

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        if not isinstance(gnconv, list):
            gnconv = [gnconv, gnconv, gnconv, gnconv]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=base_dim, drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value, gnconv=gnconv[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]



        self.uniform_init = uniform_init

        self.apply(self._init_weights)
        upsample_block = B.pixelshuffle_block
        self.upsampler = upsample_block(base_dim, out_chans, upscale_factor=upscale)
        self.scale_idx = 0

    def _init_weights(self, m):
        if not self.uniform_init:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            for j, blk in enumerate(self.stages[i]):
                f = self.GFF(x)
                x = blk(x+f)
        return x

    def forward(self, x):
        out_fea = self.stem(x)
        out_lr = self.forward_features(out_fea) + out_fea
        output = self.upsampler(out_lr)
        return output


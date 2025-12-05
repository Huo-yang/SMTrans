import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


def create_wavelet_filter_1d(wave, in_size, out_size, dtype=torch.float):
    w = pywt.Wavelet(wave)
    # 小波分解低通滤波器系数
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=dtype)
    # 小波分解高通滤波器系数
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=dtype)
    # 构建二维小波分类滤波器 分别对应L，H
    dec_filters = torch.stack([dec_lo, dec_hi], dim=0)
    # 对每个输入通道重复小波滤波器
    dec_filters = dec_filters[:, None, :].repeat(in_size, 1, 1)

    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=dtype).flip(dims=[0])
    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=dtype).flip(dims=[0])
    rec_filters = torch.stack([rec_lo, rec_hi], dim=0)
    rec_filters = rec_filters[:, None, :].repeat(out_size, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform_1d(x, filters):
    b, c, l = x.shape
    pad = (filters.shape[-1] // 2 - 1,)
    x = F.conv1d(x, filters, stride=2, padding=pad, groups=c)
    x = x.view(b, c, 2, l // 2)
    return x


def inverse_wavelet_transform_1d(x, filters):
    b, c, _, l_half = x.shape
    pad = (filters.shape[-1] // 2 - 1,)
    x = x.view(b, c * 2, l_half)
    x = F.conv_transpose1d(x, filters, stride=2, padding=pad, groups=c)
    return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv1d, self).__init__()
        assert in_channels == out_channels

        self.in_channels = in_channels
        self.stride = stride
        self.wt_levels = wt_levels

        self.wt_filter, self.iwt_filter = create_wavelet_filter_1d(wt_type, in_channels, in_channels)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding='same',
                                   stride=1, groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1])

        self.wavelet_convs = nn.ModuleList([
            nn.Conv1d(in_channels * 2, in_channels * 2, kernel_size, padding='same', stride=1,
                      groups=in_channels * 2, bias=False)
            for _ in range(self.wt_levels)
        ])

        self.wavelet_scale = nn.ModuleList([
            _ScaleModule([1, in_channels * 2, 1], init_scale=0.1)
            for _ in range(self.wt_levels)
        ])

        if stride > 1:
            self.do_stride = nn.AvgPool1d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x:torch.tensor)->torch.tensor:
        x_ll_levels, x_hh_levels, shapes = [], [], []
        curr_x_ll = x

        for i in range(self.wt_levels):
            shape = curr_x_ll.shape
            shapes.append(shape)
            if shape[2] % 2 != 0:
                curr_x_ll = F.pad(curr_x_ll, (0, 1))

            x_w = wavelet_transform_1d(curr_x_ll, self.wt_filter)
            curr_x_ll = x_w[:, :, 0, :]
            curr_x_tag = x_w.view(shape[0], shape[1] * 2, -1)
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag)).view(x_w.shape)

            x_ll_levels.append(curr_x_tag[:, :, 0, :])
            x_hh_levels.append(curr_x_tag[:, :, 1, :])

        next_x_ll = 0
        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_levels.pop() + next_x_ll
            curr_x_hh = x_hh_levels.pop()
            shape = shapes.pop()
            curr_x = torch.stack([curr_x_ll, curr_x_hh], dim=2)
            next_x_ll = inverse_wavelet_transform_1d(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, :shape[2]]

        x_tag = next_x_ll
        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


if __name__ == "__main__":
    wtconv1d = WTConv1d(in_channels=16, out_channels=16, wt_levels=1)
    inp = torch.randn(1, 16, 128)
    out = wtconv1d(inp)
    print("Input shape:", inp.shape)
    print("Output shape:", out.shape)

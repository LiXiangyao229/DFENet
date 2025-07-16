import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import einops
import numpy

import torch
import torch.nn as nn



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class PSConv(nn.Module):

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()

        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]

        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)

        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):

        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))

        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))

        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class DFE(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super(DFE, self).__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(self.c // 2, self.c // 2, 3, 1, 1)
        #self.cv3 = Conv(self.c, self.c, 3, 1)
        self.PSConv=PSConv(self.c, self.c)

        self.ratio_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # GAP
            nn.Flatten(),
            nn.Linear(self.c, 3),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

        self.cv5 = nn.Identity()
        self.cv7 = nn.Identity()
        self.cv6 = Conv(self.c, c2, 1, 1)
        self._prev_sizes = None

    def forward(self, x):

        x = self.cv1(x)
        x1, x2 = torch.split(x, self.c // 2, dim=1)
        x1 = self.cv2(x1)
        x3 = torch.cat([x1, x2], 1)
        x4 = self.PSConv(x3)

        ratios = self.ratio_generator(x4)
        ratios = ratios.mean(dim=0)

        total_channels = x4.size(1)
        sizes = (ratios * total_channels).round().int().tolist()

        size_diff = total_channels - sum(sizes)
        sizes[0] += size_diff

        if self._prev_sizes != sizes:
            self.cv5 = DWConv(sizes[0], sizes[0], 3, 1, 2).to(x4.device)
            self.cv7 = DWConv(sizes[2], sizes[2], 5, 1, 3).to(x4.device)
            self._prev_sizes = sizes

        # 分割特征图
        x5, x6, x7 = torch.split(x4, sizes, dim=1)

        # 处理各分支
        x5 = self.cv5(x5)
        x7 = self.cv7(x7)

        # 合并结果
        x8 = torch.cat([x5, x6, x7], dim=1)
        x9 = x8 + x3
        x10 = self.cv6(x9)

        return x10


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class EfficientAdditiveAttentions(nn.Module):
    def __init__(self, in_dims=512, token_dim=256, num_heads=2, use_non_linear=True, use_pos_encoding=True):
        super().__init__()

        self.num_heads = num_heads
        self.token_dim = token_dim

        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)
        self.to_v = nn.Linear(in_dims, token_dim)


        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))

        self.scale_factor = token_dim ** -0.5

        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)

        self.final = nn.Linear(token_dim * num_heads, token_dim)

        self.use_non_linear = use_non_linear
        if use_non_linear:
            self.activation = nn.GELU()

        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(1, 1, token_dim * num_heads))

        self.dynamic_attention_network = nn.Sequential(
            nn.Linear(token_dim * num_heads, token_dim * num_heads),
            nn.GELU(),
            nn.Linear(token_dim * num_heads, 1)
        )

    def forward(self, x):

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        Q = self.to_query(x)
        K = self.to_key(x)

        Q = torch.nn.functional.normalize(Q, dim=-1)
        K = torch.nn.functional.normalize(K, dim=-1)

        Q_weight = Q @ self.w_g
        A = Q_weight * self.scale_factor
        A = torch.nn.functional.normalize(A, dim=1)

        G = torch.sum(A * Q, dim=1)

        G = einops.repeat(G, "b d -> b repeat d", repeat=K.shape[1])

        dynamic_attention_weights = self.dynamic_attention_network(Q)
        dynamic_attention_weights = dynamic_attention_weights.squeeze(-1)

        dynamic_attention_weights = F.softmax(dynamic_attention_weights, dim=-1)
        out = (dynamic_attention_weights.unsqueeze(-1) * G * K) + Q

        if self.use_non_linear:
            out = self.activation(out)

        out = self.final(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return out


class CA(nn.Module):
    def __init__(self, in_planes, ratio=4, flag=True):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.flag = flag
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out) * x if self.flag else self.sigmoid(out)
        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale



class PFC512(nn.Module):
    def __init__(self):
        super().__init__()

        self.CA = CA(128)

    def forward(self, x):

        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)

        s = F.interpolate(s, m.shape[2:], mode='nearest')
        s1 = self.TripletAttention(s)
        s2 = s1 + m
        s3 = self.TripletAttention(s2)
        s4 = s3 + l

        lms = torch.cat([s2, s4, s], dim=1)
        return lms


class PFC256(nn.Module):
    def __init__(self):
        super().__init__()
        self.CA = CA(64)

    def forward(self, x):

        l, m, s = x[0], x[1], x[2]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)

        s = F.interpolate(s, m.shape[2:], mode='nearest')
        s1 = self.TripletAttention(s)
        s2 = s1 + m
        s3 = self.TripletAttention(s2)
        s4 = s3 + l

        lms = torch.cat([s2, s4, s], dim=1)
        return lms


class SFA(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=True, BatchNorm=nn.BatchNorm2d):
        super(SFA, self).__init__()

        self.with_channel = with_channel
        self.after_relu = after_relu

        self.f_x = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,kernel_size=1, bias=False),
            BatchNorm(mid_channels)
        )

        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels,kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )

        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()

        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],mode='bilinear', align_corners=False)

        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(y, size=[input_size[2], input_size[3]],mode='bilinear', align_corners=False)
        x = (1 - sim_map) * x + sim_map * y

        return x



class Zoom_cat2(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv_l_post_down = Conv(in_dim, 2*in_dim, 3, 1, 1)

    def forward(self, x):
        """l,m,s表示大中小三个尺度，最终会被整合到m这个尺度上"""
        l, m = x[0], x[1]
        tgt_size = m.shape[2:]
        l = F.adaptive_max_pool2d(l, tgt_size) + F.adaptive_avg_pool2d(l, tgt_size)
        # l = self.conv_l_post_down(l)
        # m = self.conv_m(m)
        # s = self.conv_s_pre_up(s)
        # s = self.conv_s_post_up(s)
        lms = torch.cat([l, m], dim=1)
        return lms
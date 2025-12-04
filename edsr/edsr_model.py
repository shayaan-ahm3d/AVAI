import math

from torch import nn
from torch.nn import Module, ReLU, BatchNorm2d
from torch.optim import Optimizer, Adam

class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size=3, bias=True, act=nn.ReLU(True), res_scale=0.1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2, bias=bias))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        return res + x

class Upsampler(nn.Sequential):
    def __init__(self, scale: int, n_feats: int, use_batch_norm: bool = True, use_activation: bool = True, bias: bool = True):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(n_feats, 4 * n_feats, 3, padding=1, bias=bias))
                m.append(nn.PixelShuffle(2))
                if use_batch_norm: 
                    m.append(BatchNorm2d(n_feats))
                if use_activation:
                    m.append(ReLU())
        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, 3, padding=1, bias=bias))
            m.append(nn.PixelShuffle(3))
            if use_batch_norm:
                m.append(BatchNorm2d(n_feats))
            if use_activation:
                m.append(ReLU())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class Edsr(nn.Module):
    def __init__(self, scale, n_resblocks, n_feats, n_colors=3, res_scale=0.1):
        super(Edsr, self).__init__()

        kernel_size = 3
        act = nn.ReLU(True)

        self.head = nn.Conv2d(n_colors, n_feats, kernel_size, padding=kernel_size//2)

        m_body: list[Module] = [
            ResBlock(n_feats, kernel_size, act=act, res_scale=res_scale)
            for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2))
        self.body = nn.Sequential(*m_body)

        self.tail = Upsampler(scale, n_feats, use_activation=True)
        self.last_conv = nn.Conv2d(n_feats, n_colors, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        x = self.last_conv(x)
        return x
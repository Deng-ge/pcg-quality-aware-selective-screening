from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch to use PCGResNetAttn.") from exc


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = ConvBNAct(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(x)
        h = self.bn2(self.c2(h))
        return self.act(h + x)


class AttnPool1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.SiLU(),
            nn.Conv1d(channels, 1, 1),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        x = x.mean(dim=2)
        w = torch.softmax(self.attn(x), dim=-1)
        z = (x * w).sum(dim=-1)
        if return_attn:
            return z, w
        return z


class PCGResNetAttn(nn.Module):
    """Small ResNet with temporal attention pooling for Log-Mel PCG features."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(1, 32, 3, 2, 1),
            ConvBNAct(32, 64, 3, 2, 1),
        )
        self.stage1 = nn.Sequential(ResidBlock(64), ResidBlock(64))
        self.down1 = ConvBNAct(64, 128, 3, 2, 1)
        self.stage2 = nn.Sequential(ResidBlock(128), ResidBlock(128))
        self.down2 = ConvBNAct(128, 192, 3, 2, 1)
        self.stage3 = nn.Sequential(ResidBlock(192), ResidBlock(192))
        self.pool = AttnPool1D(192)
        self.head = nn.Sequential(
            nn.Linear(192, 128),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        if return_attn:
            z, w = self.pool(x, return_attn=True)
            return self.head(z), w
        return self.head(self.pool(x))


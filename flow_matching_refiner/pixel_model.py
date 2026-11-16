"""AE-free, spatially conditioned flow in normalized mask-logit space."""
import math

import torch
from torch import nn
from torch.nn import functional as F

LOGIT_SCALE = 6.0


def source_state(coarse):
    return torch.logit(coarse.clamp(1e-4, 1-1e-4)) / LOGIT_SCALE


def probabilities(state):
    return torch.sigmoid(LOGIT_SCALE * state)


def blend_probabilities(coarse, refined, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    # Exact original probabilities, without a logit/sigmoid round trip.
    if alpha == 0:
        return coarse.clone()
    return (1-alpha)*coarse + alpha*refined


def smooth_time_embedding(t):
    frequencies = torch.tensor([1., 2., 4., 8.], device=t.device)
    phase = 2 * math.pi * t[:,None] * frequencies[None,:]
    return torch.cat((t[:,None], phase.sin(), phase.cos()), dim=1)


class Block(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(4, outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(4, outgoing)
        self.film = nn.Linear(64, 2*outgoing)

    def forward(self, x, time):
        x = F.silu(self.norm1(self.conv1(x)))
        gain, bias = self.film(time).chunk(2, 1)
        x = x * (1 + gain[:,:,None,None]) + bias[:,:,None,None]
        return F.silu(self.norm2(self.conv2(x)))


class PixelVelocity(nn.Module):
    """One state, fixed coarse and three spatial RGB channels; zero initial flow."""
    def __init__(self, base=12):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(9, 64), nn.SiLU(), nn.Linear(64, 64))
        self.encoders = nn.ModuleList([
            Block(5, base), Block(base, 2*base),
            Block(2*base, 4*base), Block(4*base, 8*base),
        ])
        self.decoders = nn.ModuleList([
            Block(12*base, 4*base), Block(6*base, 2*base), Block(3*base, base),
        ])
        self.head = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, state, t, coarse, rgb):
        time = self.time(smooth_time_embedding(t))
        x = torch.cat((state, coarse, 2*rgb-1), dim=1)
        skips = []
        for index, block in enumerate(self.encoders):
            x = block(x, time)
            skips.append(x)
            if index < 3:
                x = F.avg_pool2d(x, 2)
        for block, skip in zip(self.decoders, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat((x, skip), dim=1), time)
        return self.head(x)


def objective(velocity, target_velocity, endpoint_state, truth, valid):
    """Vessel-weighted velocity regression plus a proper decoded segmentation loss."""
    valid = valid.to(truth.dtype)
    weights = valid * (1 + 4*truth)
    mse = ((velocity-target_velocity).square()*weights).sum() / weights.sum().clamp_min(1)
    logits = LOGIT_SCALE * endpoint_state
    bce = (F.binary_cross_entropy_with_logits(logits, truth, reduction="none")*valid).sum()
    bce = bce / valid.sum().clamp_min(1)
    p, g = (logits.sigmoid()*valid).flatten(1), (truth*valid).flatten(1)
    dice = 1 - ((2*(p*g).sum(1)+1e-6)/(p.sum(1)+g.sum(1)+1e-6)).mean()
    return mse + .5*bce + .5*dice, (mse, bce, dice)

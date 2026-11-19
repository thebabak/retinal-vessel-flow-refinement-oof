"""Conditional linear-path flow matching on the frozen mask encoding.

This module learns velocity, not DDPM noise. No DDIM scheduler is used.
The coarse map is supplied as a fixed spatial condition throughout the ODE.
"""
import math

import torch
from torch import nn

from diffusion_refiner.models import DiffusionUNet, ImageEncoder


def linear_path(source, target, t):
    """Return x_t and its exact conditional velocity for a linear bridge."""
    if source.shape != target.shape or t.shape != (source.shape[0],):
        raise ValueError("Expected matching endpoint tensors and a batch of times")
    if not bool(((t >= 0) & (t <= 1)).all()):
        raise ValueError("Flow time must lie in [0, 1]")
    weight = t.reshape((-1,) + (1,) * (source.ndim - 1))
    return (1 - weight) * source + weight * target, target - source


def time_embedding(t, dim=64):
    """Continuous time embedding; the [0,1] interval spans 1000 time units."""
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2)
    )
    phase = 1000 * t[:, None] * frequencies[None, :]
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class ConditionalVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.time_mlp = nn.Sequential(nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
        self.velocity_unet = DiffusionUNet()
        # Concatenation retains a fixed, spatially resolved coarse condition.
        self.velocity_unet.in_conv = nn.Conv2d(128, 64, 3, padding=1)

    def image_condition(self, rgb):
        return self.image_encoder(rgb)

    def forward(self, state, t, coarse_encoding, image_condition):
        embedding = time_embedding(t)
        condition = image_condition + self.time_mlp(embedding)
        # DiffusionUNet ignores its t_emb argument, so time is explicitly injected.
        return self.velocity_unet(
            torch.cat((state, coarse_encoding), dim=1), embedding, condition
        )


@torch.no_grad()
def integrate(velocity, source, coarse_encoding, image_condition, steps, solver="euler"):
    """Integrate t=0 to t=1 without access to ground truth. Return state and NFE."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if solver not in ("euler", "midpoint"):
        raise ValueError("solver must be euler or midpoint")
    state = source.clone()
    dt = 1.0 / steps
    for index in range(steps):
        t = torch.full((len(source),), index * dt, device=source.device, dtype=source.dtype)
        first = velocity(state, t, coarse_encoding, image_condition)
        if solver == "midpoint":
            first = velocity(
                state + 0.5 * dt * first, t + 0.5 * dt,
                coarse_encoding, image_condition,
            )
        state = state + dt * first
    return state, steps * (2 if solver == "midpoint" else 1)


def masked_dice_loss(prediction, target, valid):
    p = (prediction * valid).flatten(1)
    g = (target * valid).flatten(1)
    dice = (2 * (p * g).sum(1) + 1e-6) / (p.sum(1) + g.sum(1) + 1e-6)
    return 1 - dice.mean()

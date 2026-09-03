"""Compatibility fallback for the pinned CUDA fast-Hadamard extension.

The released QuaRot code imports ``hadamard_transform`` from the extension.
This node has no nvcc and its RTX 5090 requires a newer PyTorch runtime than
the release pin, so this module preserves the same normalized Walsh-Hadamard
operation with ordinary PyTorch tensor operations.  It changes performance,
not the numerical method or any quantization setting.
"""

from __future__ import annotations

import torch


def hadamard_transform(value: torch.Tensor, scale: float | torch.Tensor = 1.0) -> torch.Tensor:
    n = value.shape[-1]
    if n < 1 or n & (n - 1):
        raise ValueError(f"Hadamard dimension must be a power of two, got {n}")
    output = value.contiguous()
    width = 1
    while width < n:
        blocks = output.reshape(*output.shape[:-1], n // (2 * width), 2, width)
        left = blocks[..., 0, :]
        right = blocks[..., 1, :]
        output = torch.cat((left + right, left - right), dim=-1).reshape_as(value)
        width *= 2
    return output * torch.as_tensor(scale, device=value.device, dtype=value.dtype)

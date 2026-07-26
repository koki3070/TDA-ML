"""Small PyTorch helpers shared by scripts and tests."""

from __future__ import annotations

import torch


def maybe_backward(loss: torch.Tensor) -> bool:
    """Run ``backward()`` only when ``loss`` participates in the autograd graph."""
    if loss.requires_grad:
        loss.backward()
        return True
    return False

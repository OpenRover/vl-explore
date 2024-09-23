# ==============================================================================
# Author: Yuxuan Zhang (robotics@z-yx.cc)
# License: MIT
# ==============================================================================
import torch


def norm(x: torch.Tensor, dim: int = -1, keepdim: bool = True) -> torch.Tensor:
    if dim < 0:
        dim = x.ndim + dim
    assert 0 <= dim < x.ndim, f"invalid dim: {dim}"
    return x / x.norm(dim=dim, keepdim=keepdim)

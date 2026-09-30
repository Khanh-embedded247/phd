"""Workaround: torch 1.9 + Ada (RTX 40xx) fails cusolver/cusparse on GPU inverse.

Import this module once before running MapTR so CUDA inverses fall back to CPU.
"""

from __future__ import annotations

import torch

_orig_inverse = torch.inverse


def _inverse_cpu_fallback(input, *args, **kwargs):
    if input.is_cuda:
        out = _orig_inverse(input.detach().cpu(), *args, **kwargs)
        return out.to(device=input.device, dtype=input.dtype)
    return _orig_inverse(input, *args, **kwargs)


torch.inverse = _inverse_cpu_fallback  # type: ignore[assignment]

try:
    _orig_linalg_inv = torch.linalg.inv

    def _linalg_inv_cpu_fallback(input, *args, **kwargs):
        if input.is_cuda:
            out = _orig_linalg_inv(input.detach().cpu(), *args, **kwargs)
            return out.to(device=input.device, dtype=input.dtype)
        return _orig_linalg_inv(input, *args, **kwargs)

    torch.linalg.inv = _linalg_inv_cpu_fallback  # type: ignore[assignment]
except Exception:
    pass

"""S64 subgoal 3 smoke test — PatchEncoder forward/backward on CPU."""
from __future__ import annotations

import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, "/root/capsule/code/full_automatic_execution_01")

from lib.patch_encoder import PatchEncoder  # noqa: E402


def main() -> int:
    torch.manual_seed(0)

    enc = PatchEncoder(out_dim=64)
    n_params = enc.param_count
    print(f"PatchEncoder param count: {n_params:,}", flush=True)
    assert n_params <= 250_000, f"param count {n_params} exceeds 250k budget"

    # Forward timing: (32, 1, 16, 16, 16) → (32, 64) in under 1 s
    x = torch.randn(32, 1, 16, 16, 16)
    # Warm-up: allocate + JIT any lazy kernels.
    with torch.no_grad():
        _ = enc(x)

    t0 = time.time()
    with torch.no_grad():
        y = enc(x)
    t_forward = time.time() - t0
    print(f"forward (32, 1, 16, 16, 16) → {tuple(y.shape)} in {t_forward*1000:.1f} ms",
          flush=True)
    assert y.shape == (32, 64), f"bad output shape {y.shape}"
    assert t_forward < 1.0, f"forward took {t_forward:.3f}s, exceeds 1 s budget"
    assert torch.isfinite(y).all(), "non-finite forward output"

    # Backward: finite, non-zero gradients against a random MSE target.
    enc.zero_grad()
    y2 = enc(x)
    target = torch.randn_like(y2)
    loss = nn.functional.mse_loss(y2, target)
    print(f"loss (random target) = {loss.item():.4f}", flush=True)
    t0 = time.time()
    loss.backward()
    t_back = time.time() - t0
    print(f"backward in {t_back*1000:.1f} ms", flush=True)

    grads_finite = all(
        torch.isfinite(p.grad).all() for p in enc.parameters() if p.grad is not None
    )
    grad_max = max(
        float(p.grad.abs().max()) for p in enc.parameters() if p.grad is not None
    )
    grad_nonzero = grad_max > 0
    print(f"grads finite: {grads_finite}; max|grad| = {grad_max:.6f}", flush=True)
    assert grads_finite, "non-finite gradient somewhere"
    assert grad_nonzero, "all gradients zero"

    # Output diversity check: independent inputs should produce different embeddings.
    cos = nn.functional.cosine_similarity(y[0:1], y[1:], dim=1)
    print(f"cosine similarity y[0] vs y[1:]: "
          f"min={cos.min().item():.3f} med={cos.median().item():.3f} "
          f"max={cos.max().item():.3f}", flush=True)
    assert cos.min().item() < 0.99, "embeddings are too similar across random inputs"

    print("\n=== PASSED ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

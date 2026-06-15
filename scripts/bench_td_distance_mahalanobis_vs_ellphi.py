#!/usr/bin/env python3
"""Micro-benchmark: Mahalanobis vs ellphi for ``compute_topo_distance_matrix``.

This script times wall-clock latency for building a batched (B, N, N) distance
matrix used in the topological loss path via ``tda_ml.losses.compute_topo_distance_matrix``.

What is compared
    * ``distance_mode="mahalanobis"`` — native differentiable anisotropic path.
    * ``distance_mode="ellphi"`` — tangency / ellphi backend (``ellphi_backend``).

What is **not** included
    Full training loops, classification metrics, dataset loading, or the
    trainer’s mixed-precision / dataloader overhead. Inputs are **synthetic**
    random tensors so absolute milliseconds are indicative only; use the same
    machine to interpret **ratios** (ellphi / mahalanobis).

GPU timing
    When ``device`` is CUDA, ``torch.cuda.synchronize()`` wraps timed regions to
    reduce async-launch skew.

Backward line
    One optional line reports ``sum(distance_matrix)`` backward time for ellphi
    at B=1 (same N), using ``tda_ml.torch_util.maybe_backward``. If the ellphi
    backend is non-differentiable (e.g. ``--ellphi-backend cpp``), the graph may
    be absent and the line shows ``N/A`` — use ``auto`` or ``torch`` when you
    need backward timing on CPU/GPU.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tda_ml.losses import compute_topo_distance_matrix  # noqa: E402
from tda_ml.torch_util import maybe_backward  # noqa: E402

_DEFAULT_REPEATS = 40
_DEFAULT_WARMUP = 8


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_batches(
    device: torch.device,
    fn: Callable[[], None],
    *,
    repeats: int = _DEFAULT_REPEATS,
    warmup: int = _DEFAULT_WARMUP,
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
        _sync_device(device)
    times: list[float] = []
    for _ in range(repeats):
        _sync_device(device)
        t0 = time.perf_counter()
        fn()
        _sync_device(device)
        times.append(time.perf_counter() - t0)
    mean_t = statistics.mean(times)
    if len(times) < 2:
        return mean_t, 0.0
    return mean_t, statistics.stdev(times)


def bench_distance_matrix(
    device: torch.device,
    batch_size: int,
    n_points: int,
    *,
    ellphi_backend: str = "cpp",
    repeats: int = _DEFAULT_REPEATS,
    warmup: int = _DEFAULT_WARMUP,
) -> None:
    """Time Mahalanobis vs ellphi forward passes (wall-clock mean ± stdev); then optionally one ellphi backward for ``sum(d)`` at B=1."""
    torch.manual_seed(0)
    points = torch.randn(batch_size, n_points, 2, device=device)
    params = torch.randn(batch_size, n_points, 5, device=device) * 0.1
    params[..., 2:4] = params[..., 2:4].abs() + 0.05

    def maha_forward() -> None:
        with torch.no_grad():
            compute_topo_distance_matrix(
                points,
                params,
                distance_mode="mahalanobis",
                ellphi_backend=ellphi_backend,
            )

    def ellphi_forward() -> None:
        with torch.no_grad():
            compute_topo_distance_matrix(
                points,
                params,
                distance_mode="ellphi",
                ellphi_backend=ellphi_backend,
            )

    m_mean, m_std = _time_batches(device, maha_forward, repeats=repeats, warmup=warmup)
    e_mean, e_std = _time_batches(device, ellphi_forward, repeats=repeats, warmup=warmup)

    print(f"  distance matrix (batch={batch_size}, N={n_points}, ellphi_backend={ellphi_backend})")
    print(f"    mahalanobis forward: {m_mean * 1000:.2f} ± {m_std * 1000:.2f} ms")
    print(f"    ellphi forward:      {e_mean * 1000:.2f} ± {e_std * 1000:.2f} ms")
    if m_mean > 0:
        print(f"    ratio (ellphi / maha): {e_mean / m_mean:.2f}x")

    p0 = points[0].clone().detach().requires_grad_(True)
    par0 = params[0].clone().detach().requires_grad_(True)
    d0 = compute_topo_distance_matrix(
        p0.unsqueeze(0),
        par0.unsqueeze(0),
        distance_mode="ellphi",
        ellphi_backend=ellphi_backend,
    )
    loss = d0.sum()
    _sync_device(device)
    t0 = time.perf_counter()
    if maybe_backward(loss):
        _sync_device(device)
        b_ms = (time.perf_counter() - t0) * 1000
        print(f"    ellphi backward sum(d) (B=1, same N): {b_ms:.2f} ms")
    else:
        hint = ""
        if str(ellphi_backend).lower() == "cpp":
            hint = " (hint: use --ellphi-backend auto or torch for a grad-capable ellphi path)"
        print(f"    ellphi backward sum(d) (B=1, same N): N/A (no grad path){hint}")


def _parse_device(name: str) -> torch.device:
    n = name.strip().lower()
    if n in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if n == "cpu":
        return torch.device("cpu")
    if n == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return torch.device("cuda")
    raise SystemExit(f"Unknown --device {name!r} (use auto, cpu, cuda)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Benchmark Mahalanobis vs ellphi distance-matrix compute (microbench).",
    )
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | cuda")
    p.add_argument(
        "--repeats",
        type=int,
        default=_DEFAULT_REPEATS,
        help="Timed iterations after warmup (>=2 recommended for meaningful ±).",
    )
    p.add_argument("--warmup", type=int, default=_DEFAULT_WARMUP, help="Warmup iterations (discarded)")
    p.add_argument(
        "--ellphi-backend",
        type=str,
        default="cpp",
        help="Backend string forwarded to compute_topo_distance_matrix (e.g. cpp).",
    )
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be >= 1")
    if args.warmup < 0:
        p.error("--warmup must be >= 0")

    device = _parse_device(args.device)
    print(f"Device: {device}")
    print()

    # Default (B, N) pairs match historical C-3-style smoke profile; override via code if needed.
    for bs, n in ((16, 100), (4, 100)):
        bench_distance_matrix(
            device,
            bs,
            n,
            ellphi_backend=args.ellphi_backend,
            repeats=args.repeats,
            warmup=args.warmup,
        )
        print()


if __name__ == "__main__":
    main()

"""Runtime environment profile utilities."""

from __future__ import annotations

from typing import Any

import torch


def build_runtime_profile(
    config: dict[str, Any],
    device: torch.device,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    use_amp_effective: bool,
    amp_dtype_effective: str,
) -> dict[str, Any]:
    perf_cfg = config.get("performance", {})
    training_cfg = config.get("training", {})

    return {
        "device_requested": config.get("device", "auto"),
        "device_effective": device.type,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "enable_tf32_config": bool(perf_cfg.get("enable_tf32", True)),
        "cudnn_benchmark_config": bool(perf_cfg.get("cudnn_benchmark", True)),
        "matmul_precision_config": str(perf_cfg.get("matmul_precision", "high")),
        "use_amp_config": bool(training_cfg.get("use_amp", perf_cfg.get("use_amp", True))),
        "amp_dtype_config": str(training_cfg.get("amp_dtype", perf_cfg.get("amp_dtype", "float16"))),
        "use_amp_effective": use_amp_effective,
        "amp_dtype_effective": amp_dtype_effective,
        "deterministic_algorithms": bool(
            config.get("reproducibility", {}).get("deterministic_algorithms", False)
        ),
    }

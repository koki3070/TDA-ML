"""Reproducibility helpers: DBSCAN grid resolution, fallback policy, manifest fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tda_ml.ellphi_torch import _has_ellphi_grad_api
from tda_ml.numerical_eps import (
    EIGENVALUE_FLOOR,
    INLIER_PROB_MIN,
    NUMERICAL_EPS,
    PCA_RIDGE_EPS,
)

DEFAULT_DBSCAN_EPS_LINSPACE = (0.15, 1.5, 15)
DEFAULT_DBSCAN_MIN_SAMPLES = [3, 5, 7, 10, 15]

# Skill-aligned run status (see computational-reproducibility failure semantics).
RUN_STATUS_NOT_RUN = "not-run"
RUN_STATUS_SKIPPED = "skipped"
RUN_STATUS_FAILED = "failed"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_EMPTY_RESULT = "empty-result"
RUN_STATUS_ZERO_RESULT = "zero-result"


def default_dbscan_eps_values() -> list[float]:
    lo, hi, n = DEFAULT_DBSCAN_EPS_LINSPACE
    return [float(x) for x in np.linspace(lo, hi, int(n))]


def _evaluation_dbscan_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return (config.get("evaluation") or {}).get("dbscan") or {}


def dbscan_grid_from_config(config: dict[str, Any]) -> tuple[list[float], list[int]]:
    """Resolve DBSCAN grid from ``evaluation.dbscan``; hard-fail if unset."""
    ev = _evaluation_dbscan_cfg(config)
    eps_raw = ev.get("eps_values")
    ms_raw = ev.get("min_samples_values")
    if eps_raw is None:
        raise ValueError(
            "evaluation.dbscan.eps_values must be set in config (no implicit default). "
            f"Example: linspace {DEFAULT_DBSCAN_EPS_LINSPACE}"
        )
    if ms_raw is None:
        raise ValueError(
            "evaluation.dbscan.min_samples_values must be set in config (no implicit default)."
        )
    return [float(x) for x in eps_raw], [int(x) for x in ms_raw]


def resolve_dbscan_grid(
    config: dict[str, Any],
    *,
    eps_values: list[float] | None = None,
    min_samples_values: list[int] | None = None,
) -> tuple[list[float], list[int]]:
    """Explicit args win; otherwise read ``evaluation.dbscan`` from *config*."""
    if eps_values is not None and min_samples_values is not None:
        return [float(x) for x in eps_values], [int(x) for x in min_samples_values]
    cfg_eps, cfg_ms = dbscan_grid_from_config(config)
    eps_out = [float(x) for x in eps_values] if eps_values is not None else cfg_eps
    ms_out = [int(x) for x in min_samples_values] if min_samples_values is not None else cfg_ms
    return eps_out, ms_out


def dbscan_backend_from_config(config: dict[str, Any]) -> str:
    ev = _evaluation_dbscan_cfg(config)
    backend = ev.get("backend")
    if backend is None:
        raise ValueError("evaluation.dbscan.backend must be set in config.")
    return str(backend).strip().lower()


def dbscan_metric_from_config(config: dict[str, Any]) -> str:
    ev = _evaluation_dbscan_cfg(config)
    return str(ev.get("metric", "max")).strip().lower()


REQUIRED_LOSS_WEIGHT_KEYS = ("w_class", "w_topo", "w_aniso", "w_size")


def reproducibility_settings(config: dict[str, Any]) -> dict[str, bool]:
    rep = config.get("reproducibility") or {}
    return {
        "strict_topo_samples": bool(rep.get("strict_topo_samples", True)),
        "allow_nan_batch_skip": bool(rep.get("allow_nan_batch_skip", False)),
        "allow_empty_cloud_fallback": bool(rep.get("allow_empty_cloud_fallback", False)),
        "allow_skip_degenerate_grid_cells": bool(
            rep.get("allow_skip_degenerate_grid_cells", False)
        ),
        "allow_otsu_threshold_fallback": bool(
            rep.get("allow_otsu_threshold_fallback", False)
        ),
        "allow_legacy_loss_keys": bool(rep.get("allow_legacy_loss_keys", False)),
    }


def assert_loss_config_explicit(config: dict[str, Any]) -> None:
    """Hard-fail when primary loss weights would come from legacy ``training.lambda_*``."""
    if reproducibility_settings(config)["allow_legacy_loss_keys"]:
        return
    loss_cfg = config.get("loss") or {}
    missing = [k for k in REQUIRED_LOSS_WEIGHT_KEYS if k not in loss_cfg]
    if missing:
        raise ValueError(
            f"loss section must define {missing}; legacy training.lambda_* fallbacks "
            "are disabled. Set reproducibility.allow_legacy_loss_keys=true to opt in."
        )
    training_cfg = config.get("training") or {}
    if "lr" not in training_cfg:
        raise ValueError(
            "training.lr must be set explicitly; no implicit default."
        )


def record_fallback(manifest: dict[str, Any], name: str, detail: str) -> None:
    """Append a fallback event to *manifest* (creates ``fallbacks`` list)."""
    manifest.setdefault("fallbacks", []).append({"name": name, "detail": detail})
    manifest["fallback_status"] = "recorded"


def assert_ellphi_differentiable_available(*, ellphi_differentiable: bool) -> str:
    """Return impl label or raise if differentiable ellphi was requested but unavailable."""
    if not ellphi_differentiable:
        return "ellphi_numpy"
    if not _has_ellphi_grad_api():
        raise RuntimeError(
            "distance_backend='ellphi' with ellphi_differentiable=True requires "
            "ellphi.grad (coef_from_cov_grad / pdist_tangency_grad). "
            "Install/update ellphi or set ellphi_differentiable=false explicitly."
        )
    return "ellphi_torch_grad"


def build_dbscan_eval_manifest_fields(config: dict[str, Any]) -> dict[str, Any]:
    eps, ms = dbscan_grid_from_config(config)
    return {
        "backend": dbscan_backend_from_config(config),
        "metric": dbscan_metric_from_config(config),
        "eps_values": eps,
        "min_samples_values": ms,
    }


def baseline_grids_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Baseline-only hyperparameter grids (IF / LOF / shared DBSCAN)."""
    bl = (config.get("evaluation") or {}).get("baselines") or {}
    cont = bl.get("contamination_values")
    lof_n = bl.get("lof_n_neighbors")
    if cont is None:
        raise ValueError(
            "evaluation.baselines.contamination_values must be set in config."
        )
    if lof_n is None:
        raise ValueError("evaluation.baselines.lof_n_neighbors must be set in config.")
    eps, ms = dbscan_grid_from_config(config)
    return {
        "eps_values": eps,
        "min_samples_values": ms,
        "contamination_values": [float(x) for x in cont],
        "lof_n_neighbors": [int(x) for x in lof_n],
    }


def build_reproducibility_manifest_fields(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "settings": reproducibility_settings(config),
        "numerical_eps_module": "tda_ml.numerical_eps",
        "numerical_constants": {
            "NUMERICAL_EPS": NUMERICAL_EPS,
            "PCA_RIDGE_EPS": PCA_RIDGE_EPS,
            "EIGENVALUE_FLOOR": EIGENVALUE_FLOOR,
            "INLIER_PROB_MIN": INLIER_PROB_MIN,
        },
    }


def map_final_status_to_run_status(final_status: str) -> str:
    if final_status == "completed":
        return RUN_STATUS_COMPLETED
    if final_status == "early-aborted":
        return RUN_STATUS_FAILED
    return final_status


def write_json(path: Path | str, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_grid_log(path: Path | str, grid_log: list[dict[str, Any]]) -> None:
    write_json(path, {"grid_log": grid_log})

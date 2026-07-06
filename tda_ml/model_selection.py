"""Checkpoint-selection policy for the training loop.

Prefer ``val_topo`` so the saved checkpoint matches the topology loss used
during training (same ellphi/local_pca teacher PD as ``train_epoch``).
``wdist`` / ``dbscan_mcc`` use DBSCAN + Euclidean inlier-point W-Dist
(paper reporting metric; not the training objective).

metric: 'val_topo' | 'wdist' | 'dbscan_mcc' | 'threshold_mcc' | 'val_loss'

'threshold_mcc' reproduces the legacy behavior. The threshold MCC is always
tracked separately for early-abort diagnostics regardless of this setting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tda_ml.dbscan_eval import evaluate_model_grid
from tda_ml.topo_wdist import topo_wdist_options_from_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectionSettings:
    metric: str
    eval_every: int
    eps_values: list | None
    min_samples_values: list | None
    backend: str
    minimize: bool


def selection_settings_from_config(config) -> SelectionSettings:
    sel_cfg = (config.get("training", {}).get("selection") or {})
    metric = sel_cfg.get("metric", "threshold_mcc")
    eval_every = max(1, int(sel_cfg.get("eval_every", 1)))
    dbscan_cfg = (sel_cfg.get("dbscan") or {})
    backend = (
        config.get("model", {}).get("topology_loss", {}).get("distance_backend", "mahalanobis")
    )
    settings = SelectionSettings(
        metric=metric,
        eval_every=eval_every,
        eps_values=dbscan_cfg.get("eps_values"),
        min_samples_values=dbscan_cfg.get("min_samples_values"),
        backend=backend,
        minimize=metric in ("wdist", "val_loss", "val_topo"),
    )
    if metric == "val_topo":
        logger.info("Model selection: val_topo (same TopologicalLoss as training)")
    elif metric in ("wdist", "dbscan_mcc"):
        logger.info(
            "Model selection: %s via DBSCAN grid (backend=%s, eval_every=%d)",
            metric, backend, eval_every,
        )
    else:
        logger.info("Model selection: %s", metric)
    return settings


@dataclass(frozen=True)
class EpochSelection:
    value: float | None = None
    eps: float | None = None
    min_samples: int | None = None
    wdist: float | None = None
    mcc_dbscan: float | None = None


def compute_epoch_selection(
    settings: SelectionSettings,
    *,
    epoch: int,
    epochs: int,
    val_mcc: float,
    val_loss: float,
    val_topo_loss: float,
    model,
    val_loader,
    device,
    config,
) -> EpochSelection:
    """Compute the per-epoch selection value; DBSCAN-grid metrics respect ``eval_every``."""
    if settings.metric == "threshold_mcc":
        return EpochSelection(value=float(val_mcc))
    if settings.metric == "val_loss":
        return EpochSelection(value=float(val_loss))
    if settings.metric == "val_topo":
        value = float(val_topo_loss)
        print(f"Epoch {epoch}: selection[val_topo] val_topo_loss={value:.5f}")
        return EpochSelection(value=value)

    run_sel_eval = (epoch % settings.eval_every == 0) or (epoch == epochs)
    if settings.metric in ("wdist", "dbscan_mcc") and run_sel_eval:
        objective = "wdist" if settings.metric == "wdist" else "mcc"
        grid = evaluate_model_grid(
            model,
            val_loader,
            device,
            backend=settings.backend,
            eps_values=settings.eps_values,
            min_samples_values=settings.min_samples_values,
            objective=objective,
            topo_options=topo_wdist_options_from_config(config),
        )
        print(
            f"Epoch {epoch}: selection[{settings.metric}] val_wdist={grid.wdist:.5f} "
            f"val_mcc_dbscan={grid.mcc:.4f} (eps={grid.eps:.3f}, min_samples={grid.min_samples})"
        )
        return EpochSelection(
            value=grid.wdist if objective == "wdist" else grid.mcc,
            eps=grid.eps,
            min_samples=grid.min_samples,
            wdist=grid.wdist,
            mcc_dbscan=grid.mcc,
        )
    return EpochSelection()

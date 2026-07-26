"""Checkpoint-selection policy for the training loop.

Default production protocol (``val_topo``):
- During training: save ``best_model.pth`` at minimum ``val_topo_loss`` (no DBSCAN grid).
- After training: ``evaluate_paper_protocol.py`` grid-searches DBSCAN on val once, then
  reports test MCC at the chosen ``(eps, min_samples)``.

Optional ``wdist`` / ``dbscan_mcc`` run a val DBSCAN grid every ``eval_every`` epochs to
pick checkpoints; use only for tuning or ablations (much slower on CPU).

metric: 'val_topo' | 'wdist' | 'dbscan_mcc' | 'threshold_mcc' | 'val_loss'

'threshold_mcc' reproduces the legacy behavior. The threshold MCC is always
tracked separately for early-abort diagnostics regardless of this setting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from tda_ml.dbscan_eval import evaluate_model_grid
from tda_ml.reproducibility import reproducibility_settings, resolve_dbscan_grid
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
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError(
            "training must be an explicit mapping; refusing silent selection defaults"
        )
    sel_cfg = training.get("selection")
    if not isinstance(sel_cfg, dict):
        raise ValueError(
            "training.selection must be set explicitly; refusing silent val_topo default"
        )
    if "metric" not in sel_cfg:
        raise ValueError(
            "training.selection.metric must be set explicitly; "
            "refusing silent val_topo default"
        )
    metric = str(sel_cfg["metric"]).strip()
    eval_every = max(1, int(sel_cfg.get("eval_every", 1)))
    dbscan_cfg = sel_cfg.get("dbscan") or {}
    topo = (config.get("model") or {}).get("topology_loss") or {}
    if "distance_backend" not in topo:
        raise ValueError(
            "model.topology_loss.distance_backend must be set explicitly; "
            "refusing silent mahalanobis default in selection"
        )
    backend = str(topo["distance_backend"]).lower().strip()
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
        rep = reproducibility_settings(config)
        eps_values, min_samples_values = resolve_dbscan_grid(
            config,
            eps_values=settings.eps_values,
            min_samples_values=settings.min_samples_values,
        )
        grid = evaluate_model_grid(
            model,
            val_loader,
            device,
            config=config,
            backend=settings.backend,
            eps_values=eps_values,
            min_samples_values=min_samples_values,
            objective=objective,
            topo_options=topo_wdist_options_from_config(config),
            allow_skip_degenerate_grid_cells=rep["allow_skip_degenerate_grid_cells"],
            grid_log_path=Path(config["outputs"]["log_dir"]) / f"dbscan_grid_epoch_{epoch}.json",
            manifest_ref=config.get("_manifest"),
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

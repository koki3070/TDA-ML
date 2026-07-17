"""Preflight checks before expensive experiment execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from tda_ml.config import deep_update, load_config
from tda_ml.reproducibility import (
    RUN_STATUS_NOT_RUN,
    assert_ellphi_differentiable_available,
    assert_loss_config_explicit,
    assert_ellphi_repo_matches_pin,
    baseline_grids_from_config,
    build_dbscan_eval_manifest_fields,
    build_ellphi_repo_manifest_fields,
    dbscan_grid_from_config,
    write_json,
)

TuneObjectiveKind = Literal["mcc", "wdist"]

_KNOWN_TUNE_OBJECTIVES: dict[str, TuneObjectiveKind] = {
    "val_topo_wdist_min": "wdist",
    "val_topo_wdist_min_at_val_topo_best_ckpt": "wdist",
    "val_dbscan_mcc_max": "mcc",
    "val_dbscan_mcc": "mcc",
}


def _require_import(name: str, import_fn) -> None:
    try:
        import_fn()
    except ImportError as exc:
        raise RuntimeError(f"Required dependency {name!r} is not importable: {exc}") from exc


def classify_tune_objective(
    objective: str,
    *,
    objective_kind: str | None = None,
) -> TuneObjectiveKind:
    """Resolve tune objective kind from explicit field or whitelist (no substring guess)."""
    if objective_kind is not None:
        kind = str(objective_kind).lower().strip()
        if kind in ("mcc", "wdist"):
            return kind  # type: ignore[return-value]
        raise ValueError(
            f"Unrecognized tune objective_kind {objective_kind!r}; expected 'mcc' or 'wdist'."
        )

    obj = str(objective).strip()
    if obj in _KNOWN_TUNE_OBJECTIVES:
        return _KNOWN_TUNE_OBJECTIVES[obj]
    lower = obj.lower()
    if lower in _KNOWN_TUNE_OBJECTIVES:
        return _KNOWN_TUNE_OBJECTIVES[lower]

    raise ValueError(
        f"Unrecognized tune objective {objective!r}; set objective_kind in tune JSON "
        f"or use a whitelist name: {sorted(_KNOWN_TUNE_OBJECTIVES)}."
    )


def preflight_training_config(
    config: dict[str, Any],
    *,
    project_root: Path | str | None = None,
    data_root: Path | str | None = None,
) -> dict[str, Any]:
    """Validate training setup; return manifest preview fields."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    data_path = Path(data_root) if data_root is not None else root / "data"

    assert_loss_config_explicit(config)

    _require_import("torch_topological", lambda: __import__("torch_topological"))
    _require_import("scipy", lambda: __import__("scipy"))

    topo = (config.get("model") or {}).get("topology_loss") or {}
    backend = str(topo.get("distance_backend", "mahalanobis")).lower()
    ellphi_diff = bool(topo.get("ellphi_differentiable", True))
    if backend == "ellphi":
        _require_import("ellphi", lambda: __import__("ellphi"))
        assert_ellphi_repo_matches_pin(project_root=root)
        impl = assert_ellphi_differentiable_available(ellphi_differentiable=ellphi_diff)
    else:
        impl = backend

    dtype = str((config.get("data") or {}).get("dataset_type", "mnist")).lower().strip()
    if dtype == "mnist":
        if not data_path.is_dir():
            raise FileNotFoundError(
                f"MNIST data root missing: {data_path}. "
                "Download MNIST first (e.g. run a short training job or place cached data under ./data)."
            )

    out_base = (config.get("outputs") or {}).get("base_dir")
    if out_base:
        out_path = Path(out_base)
        if not out_path.is_absolute():
            out_path = root / out_path
        out_path.mkdir(parents=True, exist_ok=True)
        if not os_access_writable(out_path):
            raise PermissionError(f"Output base not writable: {out_path}")

    preview: dict[str, Any] = {
        "config_id": config.get("meta", {}).get("config_id"),
        "distance_backend": backend,
        "distance_backend_impl": impl,
        "homology_dimensions": list(topo.get("homology_dimensions", [0, 1])),
        "seed": config.get("data", {}).get("seed"),
        "ellphi_repo": build_ellphi_repo_manifest_fields(root),
    }
    if config.get("evaluation"):
        preview["dbscan_eval"] = build_dbscan_eval_manifest_fields(config)
    return preview


def preflight_tune_json(
    tune_json: Path,
    *,
    expected: TuneObjectiveKind | None = None,
) -> dict[str, Any]:
    if not tune_json.is_file():
        raise FileNotFoundError(f"Tune JSON not found: {tune_json}")
    payload = json.loads(tune_json.read_text(encoding="utf-8"))
    for key in ("best_params", "objective"):
        if key not in payload:
            raise ValueError(f"Tune JSON missing required key {key!r}: {tune_json}")
    params = payload["best_params"]
    for key in ("w_topo", "w_aniso", "w_size", "lr"):
        if key not in params:
            raise ValueError(f"Tune JSON best_params missing {key!r}: {tune_json}")
    kind = classify_tune_objective(
        str(payload["objective"]),
        objective_kind=payload.get("objective_kind"),
    )
    if expected is not None and kind != expected:
        raise ValueError(
            f"Tune JSON objective {payload['objective']!r} is {kind!r}, expected {expected!r}: "
            f"{tune_json}"
        )
    payload["_objective_kind"] = kind
    return payload


def preflight_tune_study(
    *,
    base_config: str,
    project_root: Path | str,
    out_base: Path | str,
    study_objective: TuneObjectiveKind,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    cfg = load_config(base_config, project_root=root)
    if config_overrides:
        cfg = deep_update(cfg, config_overrides)
    dbscan_grid_from_config(cfg)
    preview = preflight_training_config(cfg, project_root=root)
    out = Path(out_base)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    preview["out_base"] = str(out)
    preview["base_config"] = base_config
    preview["study_objective"] = study_objective
    preview["config_overrides"] = config_overrides or {}
    write_json(out / "STUDY_PREFLIGHT.json", preview)
    return preview


def preflight_mcc_tune_study(
    *,
    base_config: str,
    project_root: Path | str,
    out_base: Path | str,
) -> dict[str, Any]:
    return preflight_tune_study(
        base_config=base_config,
        project_root=project_root,
        out_base=out_base,
        study_objective="mcc",
    )


def preflight_wdist_tune_study(
    *,
    base_config: str,
    project_root: Path | str,
    out_base: Path | str,
    config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return preflight_tune_study(
        base_config=base_config,
        project_root=project_root,
        out_base=out_base,
        study_objective="wdist",
        config_overrides=config_overrides,
    )


def preflight_tune_production_run(
    *,
    base_config: str,
    tune_json: Path,
    project_root: Path | str,
    out_base: Path | str,
) -> dict[str, Any]:
    root = Path(project_root)
    tune_payload = preflight_tune_json(tune_json)
    cfg = load_config(base_config, project_root=root)
    preview = preflight_training_config(cfg, project_root=root)
    preview["tune_json"] = str(tune_json.resolve())
    preview["tune_objective"] = tune_payload.get("objective")
    preview["tune_objective_kind"] = tune_payload["_objective_kind"]
    preview["checkpoint_selection"] = (
        (cfg.get("training") or {}).get("selection") or {}
    ).get("metric", "val_topo")
    out = Path(out_base)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "RUN_PREFLIGHT.json", preview)
    return preview


def write_not_run_manifest(
    log_dir: Path | str,
    *,
    reason: str,
    config: dict[str, Any] | None = None,
    entry: str = "tda_ml.main",
) -> Path:
    """Record preflight failure with skill-aligned ``not-run`` status."""
    from tda_ml.supervised_diagnostics import git_revision

    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run_status": RUN_STATUS_NOT_RUN,
        "final_status": RUN_STATUS_NOT_RUN,
        "preflight_error": reason,
        "command_entry": entry,
        "source_revision": git_revision(),
    }
    if config is not None:
        payload["config_id"] = config.get("meta", {}).get("config_id")
        payload["seed"] = (config.get("data") or {}).get("seed")
    out = p / "run_manifest.json"
    write_json(out, payload)
    return out


def preflight_baseline_eval(
    config: dict[str, Any],
    *,
    project_root: Path | str,
    out_dir: Path | str,
) -> dict[str, Any]:
    root = Path(project_root)
    preview = preflight_training_config(config, project_root=root)
    preview["baseline_grids"] = baseline_grids_from_config(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "BASELINE_PREFLIGHT.json", preview)
    return preview


def preflight_paper_eval_run_dir(run_dir: Path, *, checkpoint_name: str = "best_model.pth") -> None:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")
    ckpt = run_dir / checkpoint_name
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")


def os_access_writable(path: Path) -> bool:
    try:
        test = path / ".preflight_write_test"
        test.write_text("", encoding="utf-8")
        test.unlink(missing_ok=True)
        return True
    except OSError:
        return False

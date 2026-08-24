"""Preflight checks before expensive experiment execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from tda_ml.config import deep_update, load_config
from tda_ml.persistence_dimensions import normalize_homology_dimensions
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
    "val_dbscan_mcc_max_at_val_topo_best_ckpt": "mcc",
    "val_dbscan_mcc": "mcc",
}

PAPER_NO_CLS_BARRIER_ANISO_THRESHOLD = 6.0

# Near-tangent outlier geometry (legacy opt-in path; not the paper main table).
PAPER_NO_CLS_OUTLIER_CONTRACT: dict[str, Any] = {
    "outlier_mode": "local_pca_tangent",
    "tangent_pca_k": 10,
    "tangent_offset_min": 0.15,
    "tangent_offset_max": 0.40,
    "tangent_angle_jitter_deg": 30.0,
    "tangent_stroke_clearance": 0.08,
    "tangent_direction": "tangent",
}

PAPER_NO_CLS_CONTRACT: dict[str, Any] = {
    # MNIST Methods opt-in (near-tangent + H1). Matches
    # methods_mnist_neartangent.
    # Paper main table is RINGS_NO_CLS_CONTRACT (thin_rings), not this stack.
    "homology_dimensions": [1],
    "teacher_mode": "local_pca",
    "prob_weighting": False,
    "aniso_mode": "elongate_barrier",
    "aniso_barrier_threshold": PAPER_NO_CLS_BARRIER_ANISO_THRESHOLD,
    "distance_backend": "ellphi",
    "size_mode": "power",
    "w_class": 0.0,
    "teacher_local_pca_k": 10,
    "teacher_local_pca_normalize_axes": True,
    # After unit normalization, scale major axes to the student/DBSCAN neighborhood
    # band (~0.4). Unit teachers (major=1) inflate filtration and bias orientation.
    "teacher_local_pca_major_scale": 0.4,
    **PAPER_NO_CLS_OUTLIER_CONTRACT,
}

# Alias: barrier is the paper contract (kept for callers that still name it).
PAPER_NO_CLS_BARRIER_CONTRACT: dict[str, Any] = PAPER_NO_CLS_CONTRACT

# Ablation / legacy: same stack without the aspect ceiling (aniso_mode=elongate).
PAPER_NO_CLS_ELONGATE_CONTRACT: dict[str, Any] = {
    key: value
    for key, value in PAPER_NO_CLS_CONTRACT.items()
    if key not in ("aniso_mode", "aniso_barrier_threshold")
} | {"aniso_mode": "elongate"}

# Thin-rings paper contract: radial outliers, same loss stack as production YAML.
RINGS_NO_CLS_OUTLIER_CONTRACT: dict[str, Any] = {
    "dataset_type": "thin_rings",
    "outlier_mode": "ring_radial",
    "ring_count_min": 1,
    "ring_count_max": 2,
    "ring_radius_min": 0.25,
    "ring_radius_max": 0.60,
    "ring_center_box": 0.30,
    "ring_outlier_offset_min": 0.03,
    "ring_outlier_offset_max": 0.06,
    "ring_outlier_clearance": 0.03,
}

RINGS_NO_CLS_CONTRACT: dict[str, Any] = {
    "homology_dimensions": [0, 1],
    "teacher_mode": "local_pca",
    "prob_weighting": False,
    "aniso_mode": "elongate_barrier",
    "aniso_barrier_threshold": PAPER_NO_CLS_BARRIER_ANISO_THRESHOLD,
    "distance_backend": "ellphi",
    "size_mode": "power",
    "w_class": 0.0,
    "teacher_local_pca_k": 10,
    "teacher_local_pca_normalize_axes": True,
    # Data-intrinsic scale: median raw local-PCA major on rings val clean
    # (seed=42, 500 clouds).
    "teacher_local_pca_major_scale": 0.083,
    **RINGS_NO_CLS_OUTLIER_CONTRACT,
}

_KNOWN_TEACHER_MODES = frozenset({"euclidean", "local_pca"})


def _assert_shared_no_cls_loss_fields(config: dict[str, Any], *, label: str) -> dict[str, Any]:
    """Shared local_pca / power / barrier fields; homology value is contract-specific."""
    topo = (config.get("model") or {}).get("topology_loss") or {}
    loss = config.get("loss") or {}
    actual: dict[str, Any] = {}

    if "homology_dimensions" not in topo:
        raise ValueError(
            f"{label} config must explicitly define "
            "model.topology_loss.homology_dimensions"
        )
    actual["homology_dimensions"] = list(topo["homology_dimensions"])

    if "teacher_mode" not in loss:
        raise ValueError(
            f"{label} config must explicitly define loss.teacher_mode='local_pca'"
        )
    actual["teacher_mode"] = str(loss["teacher_mode"]).strip().lower()

    if "prob_weighting" not in topo:
        raise ValueError(
            f"{label} config must explicitly define "
            "model.topology_loss.prob_weighting=false"
        )
    actual["prob_weighting"] = bool(topo["prob_weighting"])

    if "aniso_mode" not in loss:
        raise ValueError(
            f"{label} config must explicitly define loss.aniso_mode "
            "('elongate_barrier' contract, or 'elongate' ablation)"
        )
    actual["aniso_mode"] = str(loss["aniso_mode"]).strip().lower()
    if actual["aniso_mode"] == "elongate_barrier":
        if "aniso_barrier_threshold" not in loss:
            raise ValueError(
                f"{label} barrier variant must explicitly define "
                "loss.aniso_barrier_threshold "
                f"(declared value: {PAPER_NO_CLS_BARRIER_ANISO_THRESHOLD})"
            )
        actual["aniso_barrier_threshold"] = float(loss["aniso_barrier_threshold"])

    if "distance_backend" not in topo:
        raise ValueError(
            f"{label} config must explicitly define "
            "model.topology_loss.distance_backend='ellphi'"
        )
    actual["distance_backend"] = str(topo["distance_backend"]).strip().lower()

    if "size_mode" not in loss:
        raise ValueError(
            f"{label} config must explicitly define loss.size_mode='power'"
        )
    actual["size_mode"] = str(loss["size_mode"]).strip().lower()

    if "w_class" not in loss:
        raise ValueError(
            f"{label} config must explicitly define loss.w_class=0.0"
        )
    actual["w_class"] = float(loss["w_class"])

    if "teacher_local_pca_k" not in loss:
        raise ValueError(
            f"{label} config must explicitly define loss.teacher_local_pca_k=10"
        )
    actual["teacher_local_pca_k"] = int(loss["teacher_local_pca_k"])

    if "teacher_local_pca_normalize_axes" not in loss:
        raise ValueError(
            f"{label} config must explicitly define "
            "loss.teacher_local_pca_normalize_axes=true"
        )
    actual["teacher_local_pca_normalize_axes"] = bool(
        loss["teacher_local_pca_normalize_axes"]
    )

    if "teacher_local_pca_major_scale" not in loss:
        raise ValueError(
            f"{label} config must explicitly define "
            "loss.teacher_local_pca_major_scale "
            "(rings: 0.083; MNIST Methods: 0.4)"
        )
    actual["teacher_local_pca_major_scale"] = float(
        loss["teacher_local_pca_major_scale"]
    )
    return actual


def assert_paper_no_cls_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Require the MNIST Methods near-tangent stack; never infer missing fields.

    Two anisotropy variants exist, selected explicitly by ``loss.aniso_mode``:
    ``elongate_barrier`` (Methods contract) and ``elongate`` (ablation without
    aspect ceiling). Both require H1-only and near-tangent outlier geometry.
    The barrier variant additionally requires ``loss.aniso_barrier_threshold``.
    Paper main table uses ``assert_rings_no_cls_contract`` instead.
    """
    data = config.get("data") or {}
    actual = _assert_shared_no_cls_loss_fields(config, label="Paper no_cls")

    for key in PAPER_NO_CLS_OUTLIER_CONTRACT:
        if key not in data:
            raise ValueError(
                f"Paper no_cls config must explicitly define data.{key} "
                f"(near-tangent contract value: {PAPER_NO_CLS_OUTLIER_CONTRACT[key]!r})"
            )
    actual["outlier_mode"] = str(data["outlier_mode"]).strip().lower()
    actual["tangent_pca_k"] = int(data["tangent_pca_k"])
    actual["tangent_offset_min"] = float(data["tangent_offset_min"])
    actual["tangent_offset_max"] = float(data["tangent_offset_max"])
    actual["tangent_angle_jitter_deg"] = float(data["tangent_angle_jitter_deg"])
    actual["tangent_stroke_clearance"] = float(data["tangent_stroke_clearance"])
    actual["tangent_direction"] = str(data["tangent_direction"]).strip().lower()

    expected = (
        PAPER_NO_CLS_CONTRACT
        if actual.get("aniso_mode") == "elongate_barrier"
        else PAPER_NO_CLS_ELONGATE_CONTRACT
    )
    if actual != expected:
        raise ValueError(
            "Paper no_cls contract mismatch: "
            f"expected={expected}, actual={actual}"
        )
    return actual


def assert_rings_no_cls_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Require the thin-rings Methods contract (radial outliers; not MNIST main table)."""
    data = config.get("data") or {}
    actual = _assert_shared_no_cls_loss_fields(config, label="Rings no_cls")

    if actual.get("aniso_mode") != "elongate_barrier":
        raise ValueError(
            "Rings no_cls contract requires loss.aniso_mode='elongate_barrier' "
            f"(got {actual.get('aniso_mode')!r})"
        )

    for key in RINGS_NO_CLS_OUTLIER_CONTRACT:
        if key not in data:
            raise ValueError(
                f"Rings no_cls config must explicitly define data.{key} "
                f"(rings contract value: {RINGS_NO_CLS_OUTLIER_CONTRACT[key]!r})"
            )
    actual["dataset_type"] = str(data["dataset_type"]).strip().lower()
    actual["outlier_mode"] = str(data["outlier_mode"]).strip().lower()
    actual["ring_count_min"] = int(data["ring_count_min"])
    actual["ring_count_max"] = int(data["ring_count_max"])
    actual["ring_radius_min"] = float(data["ring_radius_min"])
    actual["ring_radius_max"] = float(data["ring_radius_max"])
    actual["ring_center_box"] = float(data["ring_center_box"])
    actual["ring_outlier_offset_min"] = float(data["ring_outlier_offset_min"])
    actual["ring_outlier_offset_max"] = float(data["ring_outlier_offset_max"])
    actual["ring_outlier_clearance"] = float(data["ring_outlier_clearance"])

    if actual != RINGS_NO_CLS_CONTRACT:
        raise ValueError(
            "Rings no_cls contract mismatch: "
            f"expected={RINGS_NO_CLS_CONTRACT}, actual={actual}"
        )
    return actual


def resolve_experiment_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Select MNIST paper vs rings Methods contract from ``data.dataset_type``."""
    data = config.get("data") or {}
    dataset_type = str(data.get("dataset_type", "")).strip().lower()
    if not dataset_type:
        raise ValueError(
            "data.dataset_type must be set explicitly (mnist|thin_rings); "
            "refusing silent default when resolving experiment contract"
        )
    if dataset_type == "thin_rings":
        return assert_rings_no_cls_contract(config)
    if dataset_type == "mnist":
        return assert_paper_no_cls_contract(config)
    raise ValueError(
        f"data.dataset_type must be 'mnist' or 'thin_rings', got {dataset_type!r}"
    )


def paper_aniso_fields(config: dict[str, Any]) -> dict[str, Any]:
    """Extract the declared anisotropy variant fields from a config.

    Returns ``{"aniso_mode": ...}`` plus ``aniso_barrier_threshold`` for the
    barrier variant. Hard-fails on missing declarations so scripts mirror the
    base config instead of hardcoding a variant.
    """
    loss = config.get("loss") or {}
    if "aniso_mode" not in loss:
        raise ValueError(
            "loss.aniso_mode must be set explicitly "
            "('elongate' or 'elongate_barrier')"
        )
    mode = str(loss["aniso_mode"]).strip().lower()
    if mode not in ("elongate", "elongate_barrier"):
        raise ValueError(
            f"loss.aniso_mode={mode!r} is not a declared paper variant "
            "('elongate' or 'elongate_barrier')"
        )
    fields: dict[str, Any] = {"aniso_mode": mode}
    if mode == "elongate_barrier":
        if "aniso_barrier_threshold" not in loss:
            raise ValueError(
                "loss.aniso_barrier_threshold must be set explicitly for "
                "aniso_mode='elongate_barrier'"
            )
        fields["aniso_barrier_threshold"] = float(loss["aniso_barrier_threshold"])
    return fields


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
    """Resolve tune objective kind from whitelist; optional kind must agree."""
    obj = str(objective).strip()
    from_name = _KNOWN_TUNE_OBJECTIVES.get(obj)
    if from_name is None:
        from_name = _KNOWN_TUNE_OBJECTIVES.get(obj.lower())
    if from_name is None:
        raise ValueError(
            f"Unrecognized tune objective {objective!r}; use a whitelist name: "
            f"{sorted(_KNOWN_TUNE_OBJECTIVES)}."
        )

    if objective_kind is not None:
        kind = str(objective_kind).lower().strip()
        if kind not in ("mcc", "wdist"):
            raise ValueError(
                f"Unrecognized tune objective_kind {objective_kind!r}; "
                "expected 'mcc' or 'wdist'."
            )
        if kind != from_name:
            raise ValueError(
                f"objective_kind {kind!r} conflicts with objective {objective!r} "
                f"(whitelist kind={from_name!r})"
            )
    return from_name


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
    loss = config.get("loss") or {}
    training = config.get("training") or {}
    missing_topo = [
        key
        for key in ("distance_backend", "homology_dimensions", "prob_weighting")
        if key not in topo
    ]
    if missing_topo:
        raise ValueError(
            "model.topology_loss must explicitly define "
            f"{missing_topo}; refusing silent defaults"
        )
    if "teacher_mode" not in loss and "teacher_mode" not in training:
        raise ValueError(
            "loss.teacher_mode must be set explicitly; refusing silent euclidean default"
        )

    backend = str(topo["distance_backend"]).lower().strip()
    if backend != "ellphi":
        raise ValueError(
            f"Training PD distance_backend must be 'ellphi' (got {backend!r}). "
            "Mahalanobis is reserved for DBSCAN clustering "
            "(evaluation.dbscan.backend / --dbscan-backend), not topo filtration."
        )
    homology_dimensions = list(normalize_homology_dimensions(topo["homology_dimensions"]))
    prob_weighting = bool(topo["prob_weighting"])
    teacher_mode = str(
        loss.get("teacher_mode", training.get("teacher_mode"))
    ).strip().lower()
    if teacher_mode not in _KNOWN_TEACHER_MODES:
        raise ValueError(
            f"Unknown teacher_mode {teacher_mode!r}; "
            f"supported={sorted(_KNOWN_TEACHER_MODES)}"
        )
    for key in ("aniso_mode", "size_mode"):
        if key not in loss and key not in training:
            raise ValueError(
                f"loss.{key} must be set explicitly; refusing silent Trainer defaults"
            )
    aniso_mode = str(
        loss.get("aniso_mode", training.get("aniso_mode"))
    ).strip().lower()
    if aniso_mode in ("barrier", "elongate_barrier"):
        if "aniso_barrier_threshold" not in loss and "barrier_threshold" not in training:
            raise ValueError(
                "loss.aniso_barrier_threshold must be set explicitly for "
                f"aniso_mode={aniso_mode!r}; refusing silent 6.0 default"
            )
    if "ellphi_differentiable" not in topo:
        raise ValueError(
            "model.topology_loss.ellphi_differentiable must be set explicitly; "
            "refusing silent true default"
        )
    ellphi_diff = bool(topo["ellphi_differentiable"])
    if backend == "ellphi":
        _require_import("ellphi", lambda: __import__("ellphi"))
        assert_ellphi_repo_matches_pin(project_root=root)
        impl = assert_ellphi_differentiable_available(ellphi_differentiable=ellphi_diff)
        if prob_weighting:
            raise ValueError(
                "distance_backend='ellphi' requires model.topology_loss.prob_weighting=false"
            )
    else:
        impl = backend

    data_cfg = config.get("data") or {}
    if "dataset_type" not in data_cfg:
        raise ValueError(
            "data.dataset_type must be set explicitly (mnist|thin_rings); "
            "refusing silent 'mnist' default in preflight"
        )
    dtype = str(data_cfg["dataset_type"]).lower().strip()
    if dtype == "mnist":
        if not data_path.is_dir():
            raise FileNotFoundError(
                f"MNIST data root missing: {data_path}. "
                "Download MNIST first (e.g. run a short training job or place cached data under "
                f"{root / 'data'})."
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
        "homology_dimensions": homology_dimensions,
        "prob_weighting": prob_weighting,
        "teacher_mode": teacher_mode,
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
    expected_contract: dict[str, Any] | None = None,
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
    size_keys = ("size_ref" in params, "size_power" in params)
    if any(size_keys) and not all(size_keys):
        raise ValueError(
            "Tune JSON best_params must define both size_ref and size_power "
            f"together (or neither): {tune_json}"
        )
    kind = classify_tune_objective(
        str(payload["objective"]),
        objective_kind=payload.get("objective_kind"),
    )
    if expected is not None and kind != expected:
        raise ValueError(
            f"Tune JSON objective {payload['objective']!r} is {kind!r}, expected {expected!r}: "
            f"{tune_json}"
        )
    if expected_contract is not None:
        missing = [key for key in expected_contract if key not in payload]
        if missing:
            raise ValueError(
                f"Tune JSON predates the declared experiment contract; missing {missing}: "
                f"{tune_json}. Re-tune with the matching Methods stack "
                "(MNIST near-tangent H1 or thin-rings H0+H1)."
            )
        actual_contract = {key: payload[key] for key in expected_contract}
        if actual_contract != expected_contract:
            raise ValueError(
                "Tune JSON experiment contract does not match the production config: "
                f"expected={expected_contract}, actual={actual_contract}: {tune_json}"
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
    out = Path(out_base)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    try:
        cfg = load_config(base_config, project_root=root)
        if config_overrides:
            cfg = deep_update(cfg, config_overrides)
        contract = resolve_experiment_contract(cfg)
        dbscan_grid_from_config(cfg)
        preview = preflight_training_config(cfg, project_root=root)
    except Exception as exc:
        write_not_run_manifest(
            out / "logs_not_run",
            reason=str(exc),
            entry=f"preflight_tune_study:{study_objective}",
            config={"meta": {"config_id": base_config}, "data": {}},
        )
        write_json(
            out / "STUDY_PREFLIGHT.json",
            {
                "run_status": RUN_STATUS_NOT_RUN,
                "preflight_status": "failed",
                "preflight_error": str(exc),
                "base_config": base_config,
                "study_objective": study_objective,
            },
        )
        raise
    preview["out_base"] = str(out)
    preview["base_config"] = base_config
    preview["study_objective"] = study_objective
    preview["paper_no_cls_contract"] = contract
    preview["config_overrides"] = config_overrides or {}
    preview["run_status"] = "pending"
    write_json(out / "STUDY_PREFLIGHT.json", preview)
    return preview


def preflight_mcc_tune_study(
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
        study_objective="mcc",
        config_overrides=config_overrides,
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
    config_overrides: dict[str, Any] | None = None,
    preflight_filename: str | None = None,
) -> dict[str, Any]:
    root = Path(project_root)
    out = Path(out_base)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    artifact_name = preflight_filename or "RUN_PREFLIGHT.json"
    try:
        cfg = load_config(base_config, project_root=root)
        if config_overrides:
            cfg = deep_update(cfg, config_overrides)
        contract = resolve_experiment_contract(cfg)
        tune_payload = preflight_tune_json(
            tune_json,
            expected_contract=contract,
        )
        tune_params = tune_payload["best_params"]
        loss_apply: dict[str, Any] = {
            "w_topo": float(tune_params["w_topo"]),
            "w_aniso": float(tune_params["w_aniso"]),
            "w_size": float(tune_params["w_size"]),
        }
        if "size_ref" in tune_params:
            loss_apply["size_ref"] = float(tune_params["size_ref"])
            loss_apply["size_power"] = float(tune_params["size_power"])
        cfg = deep_update(
            cfg,
            {
                "loss": loss_apply,
                "training": {"lr": float(tune_params["lr"])},
            },
        )
        preview = preflight_training_config(cfg, project_root=root)
    except Exception as exc:
        write_json(
            out / artifact_name,
            {
                "run_status": RUN_STATUS_NOT_RUN,
                "preflight_status": "failed",
                "preflight_error": str(exc),
                "tune_json": str(Path(tune_json).resolve())
                if Path(tune_json).is_file()
                else str(tune_json),
                "base_config": base_config,
            },
        )
        raise
    preview["tune_json"] = str(Path(tune_json).resolve())
    preview["tune_objective"] = tune_payload.get("objective")
    preview["tune_objective_kind"] = tune_payload["_objective_kind"]
    preview["tune_best_params"] = tune_params
    preview["tune_source_revision"] = tune_payload.get("source_revision")
    preview["tune_study_name"] = tune_payload.get("study_name")
    preview["paper_no_cls_contract"] = contract
    preview["config_overrides"] = config_overrides or {}
    selection = (cfg.get("training") or {}).get("selection") or {}
    if "metric" not in selection:
        raise ValueError(
            "training.selection.metric must be set explicitly; "
            "refusing silent val_topo default in production preflight"
        )
    preview["checkpoint_selection"] = selection["metric"]
    preview["applied_size_from_tune"] = "size_ref" in tune_params
    write_json(out / artifact_name, preview)
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


def preflight_paper_eval_run_dir(run_dir: Path) -> None:
    """Require ``best_model.pth`` (val_topo); no alternate checkpoint names."""
    from tda_ml.checkpoint_io import resolve_val_topo_checkpoint

    if not run_dir.is_dir():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")
    resolve_val_topo_checkpoint(run_dir)


def os_access_writable(path: Path) -> bool:
    try:
        test = path / ".preflight_write_test"
        test.write_text("", encoding="utf-8")
        test.unlink(missing_ok=True)
        return True
    except OSError:
        return False

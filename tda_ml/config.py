"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml



def _require_mapping(data: Any, *, path: Path) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(
            f"Config YAML must be a mapping at top level: {path} (got {type(data).__name__})"
        )
    return data

def default_project_root() -> Path:
    """Directory containing ``configs/`` (repository root when developing from a git checkout)."""
    return Path(__file__).resolve().parent.parent


def default_data_root() -> Path:
    """MNIST / dataset cache under the package-inferred repository root (cwd-independent)."""
    return default_project_root() / "data"


def load_config(config_name: str, *, project_root: Path | str | None = None) -> dict[str, Any]:
    """
    Merge ``configs/base.yaml`` with an environment-specific YAML.

    Parameters
    ----------
    config_name:
        Short name (``dev`` -> ``configs/dev.yaml``) or path ending in ``.yaml``.
    project_root:
        Root that contains ``configs/base.yaml``. Defaults to the repository root
        inferred from this package location (not the process current working directory).
    """
    root = Path(project_root).resolve() if project_root is not None else default_project_root()

    base_path = root / "configs" / "base.yaml"
    with base_path.open("r", encoding="utf-8") as f:
        base_config = _require_mapping(yaml.safe_load(f), path=base_path)

    if config_name.endswith(".yaml"):
        env_config_path = Path(config_name)
        if not env_config_path.is_absolute():
            env_config_path = root / env_config_path
    else:
        env_config_path = root / "configs" / f"{config_name}.yaml"

    with env_config_path.open("r", encoding="utf-8") as f:
        env_config = _require_mapping(yaml.safe_load(f), path=env_config_path)

    return deep_update(base_config, env_config)


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``src`` into ``dst`` (``src`` wins on scalar conflicts)."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def model_kwargs_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments for :class:`tda_ml.models.AnisotropicOutlierClassifier`."""
    model_cfg = config.get("model") or {}
    return {
        "point_dim": int(model_cfg.get("point_dim", 2)),
        "feature_dim": int(model_cfg.get("feature_dim", 128)),
    }

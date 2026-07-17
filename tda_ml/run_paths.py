"""Short, consistent output directory and visualization file naming."""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any

_TIMESTAMP_FMT = "%m%d_%H%M%S"
OUTPUT_CATEGORIES = frozenset({"supervised", "supervised_no_cls", "tune"})

_BACKEND_SHORT = {
    "ellphi": "eph",
    "mahalanobis": "mah",
}


def short_backend(name: str) -> str:
    return _BACKEND_SHORT.get(name, name[:3])


def shorten_config_id(config_id: str, *, max_len: int = 24) -> str:
    """Derive a compact slug from a legacy config_id."""
    m = re.fullmatch(r"teacher_local_pca_power_seed(\d+)", config_id)
    if m:
        return f"pwr_s{m.group(1)}"

    m = re.fullmatch(r"teacher_local_pca_(ellphi|mahalanobis)_seed(\d+)", config_id)
    if m:
        return f"lpc_{short_backend(m.group(1))}_s{m.group(2)}"

    m = re.fullmatch(r"backend_(ellphi|mahalanobis)_seed(\d+)", config_id)
    if m:
        return f"{short_backend(m.group(1))}_s{m.group(2)}"

    m = re.fullmatch(r"tune_mcc_t(\d+)", config_id)
    if m:
        return f"t{m.group(1)}"

    m = re.fullmatch(r"tune_elongate_t(\d+)", config_id)
    if m:
        return f"t{m.group(1)}"

    m = re.fullmatch(r"barrier_smoke_(.+)", config_id)
    if m:
        return f"bar_{m.group(1)}"

    m = re.fullmatch(r"barrier_quick_probe_(.+)", config_id)
    if m:
        return f"barq_{m.group(1)}"

    m = re.fullmatch(r"smoke_(.+)", config_id)
    if m:
        return f"smk_{m.group(1)}"

    slug = config_id
    for prefix in ("elongate_n100_no_cls_", "teacher_local_pca_"):
        if slug.startswith(prefix):
            slug = slug[len(prefix) :]
    if len(slug) > max_len:
        slug = slug[:max_len]
    return slug


def resolve_run_slug(config: dict[str, Any]) -> str:
    outputs = config.get("outputs") or {}
    meta = config.get("meta") or {}
    if outputs.get("run_slug"):
        return str(outputs["run_slug"])
    if meta.get("run_slug"):
        return str(meta["run_slug"])
    return shorten_config_id(str(meta.get("config_id", "run")))


def make_run_stamp(when: datetime.datetime | None = None) -> str:
    return (when or datetime.datetime.now()).strftime(_TIMESTAMP_FMT)


def build_run_dir(
    config: dict[str, Any],
    when: datetime.datetime | None = None,
) -> tuple[str, str, str]:
    """Return ``(run_dir, run_slug, run_stamp)``.

    Hard-fails if the resolved path already exists so two runs never merge
    checkpoints/metrics into one directory.
    """
    base_dir = config.get("outputs", {}).get("base_dir", "outputs")
    slug = resolve_run_slug(config)
    stamp = make_run_stamp(when)
    run_path = Path(base_dir) / f"{slug}_{stamp}"
    if run_path.exists():
        raise FileExistsError(
            f"run_dir already exists: {run_path}; refusing to merge separate runs"
        )
    return str(run_path), slug, stamp


def experiment_base(category: str, slug: str, when: datetime.datetime | None = None) -> str:
    """Batch output root: ``outputs/<category>/<MMDD>_<slug>``."""
    if category not in OUTPUT_CATEGORIES:
        allowed = ", ".join(sorted(OUTPUT_CATEGORIES))
        raise ValueError(f"category must be one of {allowed}, got {category!r}")
    stamp = (when or datetime.datetime.now()).strftime("%m%d")
    return f"outputs/{category}/{stamp}_{slug}"


def tune_base(slug: str, when: datetime.datetime | None = None) -> str:
    """Shortcut for ``experiment_base('tune', slug)``."""
    return experiment_base("tune", slug, when=when)


def visualization_filename(epoch: int) -> str:
    return f"e{epoch}.png"

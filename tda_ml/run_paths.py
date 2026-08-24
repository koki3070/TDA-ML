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
    """Derive a compact slug from a config_id (including legacy aliases)."""
    m = re.fullmatch(r"paper_seed(\d+)", config_id)
    if m:
        return f"paper_s{m.group(1)}"

    # Pre-rename production id. Kept distinct from paper_s* so re-running an old
    # config_id cannot land in the current paper run namespace.
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

    m = re.fullmatch(r"tune_t(\d+)", config_id)
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

    # Named paper/tune/methods YAMLs: drop dataset/contract tokens but keep the
    # role, otherwise paper_* / tune_* / methods_* collapse to the same slug.
    m = re.fullmatch(r"(paper|tune|methods)_n\d+_o\d+_nocls_(.+)", config_id)
    if m:
        slug = f"{m.group(1)}_{m.group(2)}"
        return slug[:max_len] if len(slug) > max_len else slug

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


def assert_no_legacy_paper_run_namespace(out_base: Path) -> None:
    """Refuse leftover ``pwr_s*`` trees (pre-PR#7) and mixed namespaces.

    Production runs were renamed ``pwr_s*`` → ``paper_s*``. The check is
    **out_base-wide**, not per-seed: leftover ``pwr_s123_*`` must block a
    freshness probe for seed 42, otherwise the 30ep driver would treat that
    seed as missing and start ``paper_s42_*`` beside the legacy tree.

    Aggregating or skipping via silent reuse of legacy trees would be an
    implicit fallback; hard-fail instead and require a fresh ``paper_s*``
    tree (or a clean out_base).
    """
    out_base = Path(out_base)
    if not out_base.exists():
        return
    legacy_dirs = sorted(p for p in out_base.glob("pwr_s*") if p.is_dir())
    modern_dirs = sorted(p for p in out_base.glob("paper_s*") if p.is_dir())

    if legacy_dirs and modern_dirs:
        raise RuntimeError(
            f"Mixed legacy pwr_s* and paper_s* under {out_base}: "
            f"legacy={[p.name for p in legacy_dirs]}, "
            f"modern={[p.name for p in modern_dirs]}. "
            "Refuse to aggregate or skip; use one namespace only "
            "(move/delete legacy trees or choose a fresh --out-base)."
        )
    if legacy_dirs:
        raise RuntimeError(
            f"Legacy pwr_s* run tree(s) under {out_base}: "
            f"{[p.name for p in legacy_dirs]}. "
            "PR #7 renamed production runs to paper_s*; legacy trees are not "
            "aggregated or treated as fresh. Move/delete them or use a fresh "
            "--out-base, then re-run production."
        )

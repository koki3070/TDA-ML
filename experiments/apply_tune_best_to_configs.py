#!/usr/bin/env python3
"""Apply Optuna best params from JSON into elongate ellphi-tuned YAML configs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIGS = (
    "configs/elongate_n100_ellphi_tuned.yaml",
    "configs/elongate_n100_no_cls_full120_ellphi_tuned.yaml",
    "configs/elongate_n100_no_cls_full120_teacher_local_pca.yaml",
)


def _fmt_lr(x: float) -> str:
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _fmt_weight(x: float) -> str:
    return f"{x:.4f}".rstrip("0").rstrip(".")


def _set_scalar_line(text: str, key: str, value: str, comment: str | None = None) -> str:
    pat = re.compile(rf"^(\s*{re.escape(key)}:\s*)([^\s#]+)(.*)$", re.MULTILINE)
    suffix = f"   # {comment}" if comment else ""

    def repl(m: re.Match[str]) -> str:
        return f"{m.group(1)}{value}{suffix}"

    new, n = pat.subn(repl, text, count=1)
    if n != 1:
        raise ValueError(f"could not update {key} in config")
    return new


def _set_source_comment(text: str, source_line: str) -> str:
    pat = re.compile(r"^# Source:.*$", re.MULTILINE)
    if pat.search(text):
        return pat.sub(f"# {source_line}", text, count=1)
    # prepend after first comment block line
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.startswith("#") and i < 5:
            continue
        lines.insert(1, f"# {source_line}\n")
        return "".join(lines)
    return f"# {source_line}\n{text}"


def apply_best(json_path: Path, configs: tuple[str, ...] = CONFIGS) -> dict:
    payload = json.loads(json_path.read_text())
    params = payload["best_params"]
    w_topo = float(params["w_topo"])
    w_aniso = float(params["w_aniso"])
    w_size = float(params["w_size"])
    lr = float(params["lr"])
    best_wd = float(payload["best_value_wdist"])
    source = f"Source: {json_path} (v3 val_topo ckpt, val topo W-Dist={best_wd:.5f})"

    w_topo_s = _fmt_weight(w_topo)
    w_aniso_s = _fmt_weight(w_aniso)
    w_size_s = _fmt_weight(w_size)
    lr_s = _fmt_lr(lr)

    updated = []
    for rel in configs:
        path = REPO_ROOT / rel
        text = path.read_text()
        text = _set_source_comment(text, source)
        text = _set_scalar_line(text, "w_topo", w_topo_s, "tuned ellphi v2")
        text = _set_scalar_line(text, "w_aniso", w_aniso_s)
        text = _set_scalar_line(text, "w_size", w_size_s)
        text = _set_scalar_line(text, "lr", lr_s, "tuned ellphi v2")
        path.write_text(text)
        updated.append(str(path))

    return {
        "json": str(json_path),
        "w_topo": w_topo_s,
        "w_aniso": w_aniso_s,
        "w_size": w_size_s,
        "lr": lr_s,
        "best_value_wdist": best_wd,
        "configs": updated,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--json",
        type=Path,
        default=REPO_ROOT / "outputs/tune_elongate_ellphi_local_pca_v3/best_elongate_wdist_ellphi.json",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.json.is_file():
        raise SystemExit(f"missing {args.json}")
    summary = apply_best(args.json.resolve())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Safe checkpoint loading for PyTorch ``torch.save`` artifacts.

Training checkpoints are ``pickle``-based. Prefer ``weights_only=True`` when the
installed PyTorch supports it so arbitrary bytecode from untrusted ``.pth``
files is not executed. A restrictive-load failure hard-fails; there is no
silent retry with ``weights_only=False``. Callers that must load a trusted
legacy file pass ``weights_only=False`` explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch


def load_torch_checkpoint(
    path: str | Path,
    map_location: Any | None = None,
    *,
    weights_only: bool = True,
) -> Any:
    """
    Load an object saved with ``torch.save``.

    Parameters
    ----------
    path:
        Checkpoint path (``.pth`` / ``.pt``).
    map_location:
        Forwarded to ``torch.load``.
    weights_only:
        When ``True`` (default), use the restrictive unpickler. Failure does
        not fall back to ``weights_only=False``.
    """
    path = Path(path)
    common_kw: dict[str, Any] = {"map_location": map_location}
    try:
        return torch.load(path, **common_kw, weights_only=weights_only)
    except TypeError:
        # PyTorch without the ``weights_only`` keyword (API absence, not a
        # scientific fallback for a failed restrictive load).
        return torch.load(path, **common_kw)


def _looks_like_pytorch_state_dict(obj: Any) -> bool:
    """Heuristic: string keys and all tensor values (typical ``nn.Module`` state)."""
    if not isinstance(obj, dict) or not obj:
        return False
    if not all(isinstance(k, str) for k in obj):
        return False
    return all(isinstance(v, torch.Tensor) for v in obj.values())


def resolve_val_topo_checkpoint(run_dir: Path) -> tuple[str, int, float]:
    """Return ``(checkpoint_name, epoch, val_topo_loss)`` from ``best_model.pth`` only.

    Hard-fails if the val_topo checkpoint is missing or lacks selection metadata.
    No fallback to ``checkpoint_epoch_*.pth`` (skill: no silent checkpoint substitute).
    """
    best_path = run_dir / "best_model.pth"
    if not best_path.is_file():
        raise FileNotFoundError(
            f"Missing best_model.pth (val_topo selection checkpoint): {best_path}"
        )
    ckpt = load_torch_checkpoint(best_path, map_location="cpu")
    epoch = int(ckpt.get("epoch", -1))
    sel = ckpt.get("selection_value", ckpt.get("val_topo_loss"))
    if sel is None:
        raise RuntimeError(
            f"Checkpoint missing selection_value/val_topo_loss: {best_path}"
        )
    return "best_model.pth", epoch, float(sel)


def extract_model_state_dict(checkpoint: Any) -> dict[str, Any]:
    """Return ``model_state_dict`` payload if present; else ``checkpoint`` if it is a state dict."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        sd = checkpoint["model_state_dict"]
        if not _looks_like_pytorch_state_dict(sd):
            raise ValueError(
                "Checkpoint has key 'model_state_dict' but its value does not look like a PyTorch state dict "
                f"(type={type(sd).__name__})."
            )
        return sd
    if _looks_like_pytorch_state_dict(checkpoint):
        return cast(dict[str, Any], checkpoint)
    raise ValueError(
        "Expected a dict with string keys mapping to tensors (``nn.Module.state_dict()``), "
        "or a training checkpoint dict containing 'model_state_dict'."
    )

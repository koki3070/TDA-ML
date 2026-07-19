"""Run-time setup for the training entrypoint: device, torch runtime, DataLoaders."""

from __future__ import annotations

import logging
import os
from typing import NamedTuple

import torch

from tda_ml.data_loader import NoisyMNISTDataset, create_data_loader

logger = logging.getLogger(__name__)


class DataLoaderSettings(NamedTuple):
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None


def resolve_device(config) -> torch.device:
    if config.get("device") and config["device"] != "auto":
        return torch.device(config["device"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dataloader_settings(config, device) -> DataLoaderSettings:
    data_cfg = config["data"]
    cpu_threads = os.cpu_count() or 8
    default_workers = min(16, max(4, cpu_threads // 2))

    num_workers = int(data_cfg.get("num_workers", default_workers))
    num_workers = max(0, min(num_workers, cpu_threads))

    pin_memory = bool(data_cfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(data_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = data_cfg.get("prefetch_factor", 4 if num_workers > 0 else None)

    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None

    return DataLoaderSettings(num_workers, pin_memory, persistent_workers, prefetch_factor)


def configure_torch_runtime(config, device) -> None:
    perf_cfg = config.get("performance", {})
    if device.type != "cuda":
        return

    if perf_cfg.get("enable_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision(perf_cfg.get("matmul_precision", "high"))
    torch.backends.cudnn.benchmark = bool(perf_cfg.get("cudnn_benchmark", True))


def build_dataloaders(config, seed: int, settings: DataLoaderSettings):
    """Build (train, val, test) DataLoaders with the split/RNG scheme of the official runs.

    Train/val indices come from one ``randperm(60000)`` draw and test indices from a
    subsequent ``randperm(10000)`` on the same generator, so the RNG consumption order
    must not change.
    """
    data_cfg = config["data"]
    train_size = data_cfg.get("train_size", 4500)
    val_size = data_cfg.get("val_size", 500)
    test_size = data_cfg.get("test_size", 1000)

    generator = torch.Generator().manual_seed(seed)
    full_train_indices = torch.randperm(60000, generator=generator)[: train_size + val_size]

    train_indices = full_train_indices[:train_size]
    val_indices = full_train_indices[train_size:]

    loader_kwargs = dict(
        batch_size=data_cfg["batch_size"],
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        persistent_workers=settings.persistent_workers,
        prefetch_factor=settings.prefetch_factor,
    )
    if "outlier_mode" not in data_cfg:
        raise ValueError(
            "data.outlier_mode must be set explicitly (uniform|local_pca_tangent); "
            "refusing silent uniform default"
        )
    outlier_mode = str(data_cfg["outlier_mode"]).strip().lower()
    if outlier_mode not in ("uniform", "local_pca_tangent"):
        raise ValueError(
            f"data.outlier_mode must be 'uniform' or 'local_pca_tangent', got {outlier_mode!r}"
        )
    if "noise_std" not in data_cfg:
        raise ValueError(
            "data.noise_std must be set explicitly; refusing silent default"
        )

    dataset_kwargs = dict(
        root="./data",
        max_points=data_cfg["max_points"],
        num_outliers=data_cfg["num_outliers"],
        noise_std=float(data_cfg["noise_std"]),
        deterministic=True,
        noise_seed=seed,
        outlier_mode=outlier_mode,
        allow_empty_cloud_fallback=bool(
            (config.get("reproducibility") or {}).get("allow_empty_cloud_fallback", False)
            or data_cfg.get("allow_empty_cloud_fallback", False)
        ),
        allow_otsu_threshold_fallback=bool(
            (config.get("reproducibility") or {}).get("allow_otsu_threshold_fallback", False)
            or data_cfg.get("allow_otsu_threshold_fallback", False)
        ),
    )

    if outlier_mode == "local_pca_tangent":
        for key in (
            "tangent_pca_k",
            "tangent_offset_min",
            "tangent_offset_max",
            "tangent_angle_jitter_deg",
            "tangent_stroke_clearance",
            "tangent_direction",
        ):
            if key not in data_cfg:
                raise ValueError(
                    f"data.{key} must be set explicitly for outlier_mode=local_pca_tangent"
                )
        dataset_kwargs.update(
            tangent_pca_k=int(data_cfg["tangent_pca_k"]),
            tangent_offset_min=float(data_cfg["tangent_offset_min"]),
            tangent_offset_max=float(data_cfg["tangent_offset_max"]),
            tangent_angle_jitter_deg=float(data_cfg["tangent_angle_jitter_deg"]),
            tangent_stroke_clearance=float(data_cfg["tangent_stroke_clearance"]),
            tangent_direction=str(data_cfg["tangent_direction"]),
        )

    train_dataset = NoisyMNISTDataset(train=True, indices=train_indices, **dataset_kwargs)
    train_loader = create_data_loader(train_dataset, shuffle=True, **loader_kwargs)

    val_dataset = NoisyMNISTDataset(train=True, indices=val_indices, **dataset_kwargs)
    val_loader = create_data_loader(val_dataset, shuffle=False, **loader_kwargs)

    test_indices = torch.randperm(10000, generator=generator)[:test_size]
    test_dataset = NoisyMNISTDataset(train=False, indices=test_indices, **dataset_kwargs)
    test_loader = create_data_loader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader

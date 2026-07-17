import logging

import numpy as np
import torch
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

logger = logging.getLogger(__name__)


class NoisyMNISTDataset(Dataset):
    """
    MNIST dataset converted to noisy 2D point clouds.

    Each image is binarized, converted to a set of (x, y) coordinates,
    subsampled or padded to a fixed size, optionally perturbed with Gaussian
    noise, and augmented with outlier points.

    Args:
        root (str): Path to store/load MNIST data.
        train (bool): Use training split if True, else test split.
        num_samples (int): Number of samples to use (randomly subsampled).
        max_points (int): Fixed number of inlier points per sample.
        num_outliers (int): Number of outlier points to add.
        noise_std (float): Std of Gaussian jitter applied to inlier points.
        deterministic (bool): If True, fixes RNG per sample for reproducibility.
        indices (torch.Tensor, optional): Explicit index subset to use.
        noise_seed (int): Base seed for deterministic noise generation.
        outlier_mode (str): ``uniform`` (``[-1,1]^2``) or ``local_pca_tangent``
            (displace along local PCA PC1 of the clean inliers).
        tangent_pca_k (int): Neighbor count for local PCA when
            ``outlier_mode=local_pca_tangent``.
        tangent_offset_min / tangent_offset_max (float): Displacement magnitude
            range along PC1 (normalized coordinates).

    Returns (per item):
        data (Tensor): Shape (max_points + num_outliers, 2). Shuffled point cloud.
        labels (Tensor): Shape (max_points + num_outliers,). 0=inlier, 1=outlier.
        clean_pc (Tensor): Shape (max_points, 2). Noise-free inlier points (zero-padded).
    """

    def __init__(self, root='./data', train=True, num_samples=5000,
                 max_points=150, num_outliers=20, noise_std=0.01,
                 deterministic=False, indices=None, noise_seed=0, preload=True,
                 allow_empty_cloud_fallback=False,
                 allow_otsu_threshold_fallback=False,
                 outlier_mode="uniform",
                 tangent_pca_k=10,
                 tangent_offset_min=0.15,
                 tangent_offset_max=0.40):
        self.max_points = max_points
        self.num_outliers = num_outliers
        self.noise_std = noise_std
        self.deterministic = deterministic
        self.noise_seed = noise_seed
        self.preload = preload
        self.allow_empty_cloud_fallback = bool(allow_empty_cloud_fallback)
        self.allow_otsu_threshold_fallback = bool(allow_otsu_threshold_fallback)
        mode = str(outlier_mode).strip().lower()
        if mode not in ("uniform", "local_pca_tangent"):
            raise ValueError(
                f"outlier_mode must be 'uniform' or 'local_pca_tangent', got {outlier_mode!r}"
            )
        self.outlier_mode = mode
        self.tangent_pca_k = int(tangent_pca_k)
        self.tangent_offset_min = float(tangent_offset_min)
        self.tangent_offset_max = float(tangent_offset_max)

        full_dataset = datasets.MNIST(root, train=train, download=True)

        if indices is not None:
            self.images = full_dataset.data[indices]
            self.labels = full_dataset.targets[indices]
        elif num_samples < len(full_dataset):
            indices = torch.randperm(len(full_dataset))[:num_samples]
            self.images = full_dataset.data[indices]
            self.labels = full_dataset.targets[indices]
        else:
            self.images = full_dataset.data
            self.labels = full_dataset.targets

        self.preloaded_points = None
        if self.preload:
            self._preload_all_points()

    def _preload_all_points(self):
        """Pre-calculates the base point clouds for all images to speed up training."""
        logger.info("Preloading %s MNIST samples into memory...", len(self.images))
        self.preloaded_points = []
        for i in range(len(self.images)):
            img = self.images[i].numpy()
            points = self._image_to_base_points(img)
            self.preloaded_points.append(points)
        logger.info("MNIST preload complete.")

    def _image_to_base_points(self, img):
        """Converts a raw MNIST image to a normalized [-1, 1] point cloud."""
        try:
            thresh = threshold_otsu(img)
            binary_img = img > thresh
        except ValueError as exc:
            if not self.allow_otsu_threshold_fallback:
                raise RuntimeError(
                    "Otsu threshold failed for MNIST image; "
                    "set reproducibility.allow_otsu_threshold_fallback=true to opt in "
                    "to img>0 binarization."
                ) from exc
            binary_img = img > 0

        # np.argwhere returns (row, col) = (y, x)
        points = torch.tensor(np.argwhere(binary_img), dtype=torch.float32)
        if points.shape[0] > 0:
            # Map to Cartesian (x, y) and normalize to [-1, 1]
            # row -> 27-row (y), col -> x
            points = torch.stack([points[:, 1], 27 - points[:, 0]], dim=1)
            points = (points / 27.0) * 2.0 - 1.0
        return points

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves a single sample and converts it to a mathematical point cloud.

        The process follows these mathematical steps:
        1. Binarization & Normalization (Preloaded if enabled)
        2. Subsampling/Padding to fixed size (max_points)
        3. Noise Injection & Outlier Addition
        """
        if self.preloaded_points is not None:
            points = self.preloaded_points[idx]
        else:
            img = self.images[idx].numpy()
            points = self._image_to_base_points(img)

        total_capacity = self.max_points + self.num_outliers

        if self.deterministic:
            rng = torch.Generator()
            rng.manual_seed(self.noise_seed + idx)
        else:
            rng = None

        num_points = points.shape[0]

        if num_points == 0:
            if not self.allow_empty_cloud_fallback:
                raise RuntimeError(
                    f"Empty foreground point cloud at dataset index {idx}; "
                    "set data.allow_empty_cloud_fallback=true to opt in to random fallback."
                )
            fallback_n = min(8, self.max_points)
            if self.deterministic:
                points = torch.rand(fallback_n, 2, generator=rng) * 2.0 - 1.0
            else:
                points = torch.rand(fallback_n, 2) * 2.0 - 1.0
            num_points = points.shape[0]

        if num_points >= self.max_points:
            if self.deterministic:
                choice = torch.randperm(num_points, generator=rng)[:self.max_points]
            else:
                choice = torch.randperm(num_points)[:self.max_points]
            inliers = points[choice]
            clean_pc_points = points[choice]
        else:
            num_padding = self.max_points - num_points
            if self.deterministic:
                pad_indices = torch.randint(0, num_points, (num_padding,), generator=rng)
            else:
                pad_indices = torch.randint(0, num_points, (num_padding,))
            inliers = torch.cat([points, points[pad_indices]], dim=0)
            clean_pc_points = points

        if self.noise_std > 0:
            if self.deterministic:
                noise = torch.randn(inliers.size(), generator=rng) * self.noise_std
            else:
                noise = torch.randn_like(inliers) * self.noise_std
            inliers = inliers + noise

        if self.num_outliers > 0:
            if self.outlier_mode == "uniform":
                if self.deterministic:
                    outliers = torch.rand(self.num_outliers, 2, generator=rng) * 2.0 - 1.0
                else:
                    outliers = torch.rand(self.num_outliers, 2) * 2.0 - 1.0
            else:
                from tda_ml.tangent_outliers import sample_local_pca_tangent_outliers

                # Local PCA on clean digit geometry (pre-jitter) so tangent is defined
                # by the stroke, not by isotropic noise.
                pca_src = clean_pc_points if clean_pc_points.shape[0] >= 2 else inliers
                outliers = sample_local_pca_tangent_outliers(
                    pca_src,
                    self.num_outliers,
                    k=self.tangent_pca_k,
                    offset_min=self.tangent_offset_min,
                    offset_max=self.tangent_offset_max,
                    generator=rng,
                )
        else:
            outliers = torch.empty(0, 2)

        data = torch.zeros(total_capacity, 2)
        labels = torch.ones(total_capacity, dtype=torch.long)

        data[:self.max_points] = inliers
        labels[:self.max_points] = 0

        if self.num_outliers > 0:
            data[self.max_points:] = outliers

        if self.deterministic:
            perm = torch.randperm(total_capacity, generator=rng)
        else:
            perm = torch.randperm(total_capacity)
        data = data[perm]
        labels = labels[perm]

        clean_pc = torch.zeros(self.max_points, 2)
        clean_pc[:clean_pc_points.shape[0]] = clean_pc_points

        return data, labels, clean_pc


class MockOutlierDataset(Dataset):
    """
    Synthetic point-cloud dataset for unit tests (MNIST-independent).

    Each sample has ``max_points`` points, where ``num_outliers`` are labeled as outliers.
    ``clean_pc`` stores only inlier points in the leading rows, with zero padding afterward.
    """

    def __init__(self, num_samples=10, max_points=50, num_outliers=10):
        if max_points < num_outliers:
            raise ValueError("max_points must be >= num_outliers")
        self.num_samples = num_samples
        self.max_points = max_points
        self.num_outliers = num_outliers

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        g = torch.Generator().manual_seed((hash(idx) ^ hash(id(self))) % (2**31))
        n_in = self.max_points - self.num_outliers
        clean_pc = torch.zeros(self.max_points, 2)
        clean_pc[:n_in] = torch.randn(n_in, 2, generator=g)

        noisy_pc = torch.zeros(self.max_points, 2)
        noisy_pc[:n_in] = clean_pc[:n_in] + 0.01 * torch.randn(n_in, 2, generator=g)
        noisy_pc[n_in:] = torch.rand(self.num_outliers, 2, generator=g) * 2.0 - 1.0

        labels = torch.zeros(self.max_points, dtype=torch.long)
        labels[n_in:] = 1

        perm = torch.randperm(self.max_points, generator=g)
        noisy_pc = noisy_pc[perm]
        labels = labels[perm]

        return noisy_pc, labels, clean_pc


class PreloadedOutlierMNIST(NoisyMNISTDataset):
    """
    Wrapper around :class:`NoisyMNISTDataset` for legacy/test compatibility.

    Here ``max_points`` is interpreted as total cloud size (inliers + outliers).
    The parent class receives inlier capacity ``max_points - num_outliers``.
    ``__getitem__`` returns ``(data, labels, clean_pc)`` like :class:`NoisyMNISTDataset`;
    ``clean_pc`` is zero-padded to ``max_points`` rows.
    """

    def __init__(
        self,
        root="./data",
        train=True,
        num_samples=5000,
        max_points=150,
        num_outliers=20,
        **kwargs,
    ):
        if max_points < num_outliers:
            raise ValueError("max_points must be >= num_outliers")
        self._cloud_size = max_points
        n_inlier_slots = max_points - num_outliers
        super().__init__(
            root=root,
            train=train,
            num_samples=num_samples,
            max_points=n_inlier_slots,
            num_outliers=num_outliers,
            **kwargs,
        )

    def __getitem__(self, idx):
        data, labels, clean_pc = super().__getitem__(idx)
        m = self._cloud_size
        clean_pad = torch.zeros(m, 2, dtype=clean_pc.dtype, device=clean_pc.device)
        n = min(clean_pc.shape[0], m)
        clean_pad[:n] = clean_pc[:n]
        return data, labels, clean_pad


def get_dataset(config):
    """
    Construct a dataset based on ``config['data']['dataset_type']``.

    - ``mock`` -> :class:`MockOutlierDataset`
    - ``mnist`` -> :class:`PreloadedOutlierMNIST`
    """
    data_cfg = config.get("data") or {}
    dtype = str(data_cfg.get("dataset_type", "mnist")).lower().strip()
    if dtype == "mock":
        return MockOutlierDataset(
            num_samples=int(data_cfg.get("num_samples", 10)),
            max_points=int(data_cfg.get("max_points", 50)),
            num_outliers=int(data_cfg.get("num_outliers", 10)),
        )
    if dtype == "mnist":
        return PreloadedOutlierMNIST(
            root=str(data_cfg.get("root", "./data")),
            train=bool(data_cfg.get("train", False)),
            num_samples=int(data_cfg["num_samples"]),
            max_points=int(data_cfg["max_points"]),
            num_outliers=int(data_cfg["num_outliers"]),
            preload=bool(data_cfg.get("preload", True)),
        )
    raise ValueError(f"Unknown dataset_type: {dtype!r}")


def create_data_loader(
    dataset,
    batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=None,
    drop_last=False,
):
    """Wrap a dataset in a DataLoader with performance-friendly options."""
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**loader_kwargs)

"""Label-free one-class / novelty models for fair comparison with proposed.

Information contract
--------------------
Usable
  - ``X_clean`` (inlier coordinates only) at fit time
  - mixed-cloud coordinates at score / predict time
  - a *scalar* contamination ratio ``k`` when the threshold rule needs it
Not usable
  - per-point in/out labels at fit time
  - test labels for threshold or hyper-parameter selection
  - proposed internals (ellphi, PH / topological loss, learned ellipses)

sklearn ``OneClassSVM(kernel="rbf")`` is kernel SVDD (Tax & Duin / Schölkopf
with a Gaussian kernel). ``svdd_primal`` is a distinct input-space sphere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import torch
import torch.nn as nn
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from tda_ml.dbscan_eval import valid_clean_inliers
from tda_ml.metrics import (
    compute_recall_specificity_gmean_mcc,
    compute_recall_specificity_gmean_mcc_wdist,
)

ThresholdRule = Literal["val_mcc", "known_ratio", "inlier_recall"]
THRESHOLD_RULES: tuple[str, ...] = ("val_mcc", "known_ratio", "inlier_recall")

# Higher score => more outlier-like, for every method in this module.
HIGHER_IS_OUTLIER = True


class Scorer(Protocol):
    def score(self, points: np.ndarray) -> np.ndarray:
        """Return a 1-D array; larger means more outlier-like."""


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int
    tn: int
    fp: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.tn + self.fp + self.fn


@dataclass
class CloudMetricRow:
    recall: float
    specificity: float
    gmean: float
    mcc: float
    wdist: float | None
    counts: ConfusionCounts


def require_clean_xy(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"clean points must be (N, 2); got {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError(f"need at least 2 clean points; got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise ValueError("clean points contain non-finite values")
    return arr


def pool_points(
    clouds: list[np.ndarray], max_points: int, rng: np.random.Generator
) -> np.ndarray:
    if not clouds:
        raise ValueError("no clouds to pool")
    all_pts = np.concatenate([require_clean_xy(c) for c in clouds], axis=0)
    if len(all_pts) <= max_points:
        return all_pts
    idx = rng.choice(len(all_pts), size=int(max_points), replace=False)
    return all_pts[idx]


def contamination_ratio(n_inliers: int, n_outliers: int) -> float:
    total = int(n_inliers) + int(n_outliers)
    if total <= 0:
        raise ValueError(f"empty cloud: n_in={n_inliers} n_out={n_outliers}")
    if n_outliers < 0 or n_inliers < 0:
        raise ValueError("n_inliers and n_outliers must be >= 0")
    return float(n_outliers) / float(total)


def confusion_counts(y: np.ndarray, pred: np.ndarray) -> ConfusionCounts:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    if y.shape != pred.shape:
        raise ValueError(f"shape mismatch y={y.shape} pred={pred.shape}")
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return ConfusionCounts(tp=tp, tn=tn, fp=fp, fn=fn)


def add_counts(a: ConfusionCounts, b: ConfusionCounts) -> ConfusionCounts:
    return ConfusionCounts(
        tp=a.tp + b.tp, tn=a.tn + b.tn, fp=a.fp + b.fp, fn=a.fn + b.fn
    )


def _finite_scores(scores: np.ndarray) -> np.ndarray:
    sc = np.asarray(scores, dtype=np.float64).reshape(-1)
    if sc.size == 0:
        raise ValueError("empty score array")
    if not np.isfinite(sc).all():
        raise ValueError("scores contain non-finite values; refusing silent clip")
    return sc


def apply_score_threshold(
    scores: np.ndarray, threshold: float, *, higher_is_outlier: bool = True
) -> np.ndarray:
    sc = _finite_scores(scores)
    if higher_is_outlier:
        return (sc >= float(threshold)).astype(np.int64)
    return (sc <= float(threshold)).astype(np.int64)


def apply_known_ratio(
    scores: np.ndarray, k: float, *, higher_is_outlier: bool = True
) -> np.ndarray:
    """Flag the most outlier-like fraction ``k`` of points in *this* cloud.

    Uses only the scalar ratio ``k``, never point labels. Ties at the cutoff
    are broken by ``>=`` on the k-quantile (higher = outlier).
    """
    if not (0.0 < k < 1.0):
        raise ValueError(f"contamination k must be in (0, 1); got {k}")
    sc = _finite_scores(scores)
    q = float(np.quantile(sc, 1.0 - k if higher_is_outlier else k))
    return apply_score_threshold(sc, q, higher_is_outlier=higher_is_outlier)


def threshold_for_inlier_recall(
    clean_scores: np.ndarray,
    tau: float,
    *,
    higher_is_outlier: bool = True,
) -> float:
    """Lowest outlier cutoff such that inlier recall on clean val is >= tau.

    Clean val has no outliers, so inlier recall = fraction of clean points
    scored as inliers. ``tau`` example: 0.95.
    """
    if not (0.0 < tau <= 1.0):
        raise ValueError(f"tau must be in (0, 1]; got {tau}")
    sc = _finite_scores(clean_scores)
    if higher_is_outlier:
        # Keep the lowest ``tau`` fraction as inliers => cutoff at quantile tau.
        return float(np.quantile(sc, tau))
    return float(np.quantile(sc, 1.0 - tau))


def _mean_cloud_mcc(
    score_clouds: list[np.ndarray],
    label_clouds: list[np.ndarray],
    threshold: float,
    *,
    higher_is_outlier: bool,
) -> float:
    mccs: list[float] = []
    for sc, y in zip(score_clouds, label_clouds, strict=True):
        pred = apply_score_threshold(sc, threshold, higher_is_outlier=higher_is_outlier)
        _, _, _, mcc = compute_recall_specificity_gmean_mcc(y, pred)
        mccs.append(float(mcc))
    if not mccs:
        raise ValueError("no clouds for val MCC threshold search")
    return float(np.mean(mccs))


def select_threshold_val_mcc(
    score_clouds: list[np.ndarray],
    label_clouds: list[np.ndarray],
    *,
    higher_is_outlier: bool = True,
    n_quantiles: int = 99,
) -> tuple[float, float]:
    """Pick a global score cutoff maximizing mean per-cloud MCC on val mixed.

    Val labels are used only here (detection-head selection), matching proposed
    Maha-DBSCAN ``(eps, min_samples)`` selection. Test labels are not used.
    """
    if len(score_clouds) != len(label_clouds):
        raise ValueError("score/label cloud counts differ")
    all_scores = np.concatenate([_finite_scores(s) for s in score_clouds])
    qs = np.linspace(0.01, 0.99, int(n_quantiles))
    grid = np.unique(np.quantile(all_scores, qs))
    if grid.size == 0:
        raise ValueError("empty threshold grid")
    best_thr = float(grid[0])
    best_mcc = -2.0
    for thr in grid:
        mcc = _mean_cloud_mcc(
            score_clouds, label_clouds, float(thr), higher_is_outlier=higher_is_outlier
        )
        if mcc > best_mcc:
            best_mcc = mcc
            best_thr = float(thr)
    return best_thr, float(best_mcc)


def predict_clouds(
    score_clouds: list[np.ndarray],
    *,
    rule: ThresholdRule,
    threshold: float | None = None,
    k: float | None = None,
    higher_is_outlier: bool = True,
) -> list[np.ndarray]:
    preds: list[np.ndarray] = []
    for sc in score_clouds:
        if rule == "known_ratio":
            if k is None:
                raise ValueError("known_ratio requires k")
            preds.append(apply_known_ratio(sc, k, higher_is_outlier=higher_is_outlier))
        elif rule in ("val_mcc", "inlier_recall"):
            if threshold is None:
                raise ValueError(f"{rule} requires a frozen threshold")
            preds.append(
                apply_score_threshold(
                    sc, threshold, higher_is_outlier=higher_is_outlier
                )
            )
        else:
            raise ValueError(f"unknown threshold rule {rule!r}")
    return preds


def resolve_threshold(
    rule: ThresholdRule,
    *,
    val_mixed_scores: list[np.ndarray] | None = None,
    val_mixed_labels: list[np.ndarray] | None = None,
    val_clean_scores: list[np.ndarray] | None = None,
    k: float | None = None,
    tau: float = 0.95,
    higher_is_outlier: bool = True,
) -> dict[str, Any]:
    """Select the detection cutoff under a declared rule. Never reads test."""
    if rule not in THRESHOLD_RULES:
        raise ValueError(f"rule must be one of {THRESHOLD_RULES}; got {rule!r}")
    if rule == "known_ratio":
        if k is None:
            raise ValueError("known_ratio requires contamination k")
        return {
            "rule": rule,
            "threshold": None,
            "k": float(k),
            "val_mcc": None,
            "tau": None,
        }
    if rule == "inlier_recall":
        if not val_clean_scores:
            raise ValueError("inlier_recall requires clean-val scores")
        thr = threshold_for_inlier_recall(
            np.concatenate(val_clean_scores),
            tau,
            higher_is_outlier=higher_is_outlier,
        )
        return {
            "rule": rule,
            "threshold": thr,
            "k": None,
            "val_mcc": None,
            "tau": float(tau),
        }
    if not val_mixed_scores or val_mixed_labels is None:
        raise ValueError("val_mcc requires mixed-val scores and labels")
    thr, val_mcc = select_threshold_val_mcc(
        val_mixed_scores, val_mixed_labels, higher_is_outlier=higher_is_outlier
    )
    return {
        "rule": rule,
        "threshold": thr,
        "k": None,
        "val_mcc": val_mcc,
        "tau": None,
    }


def evaluate_predictions(
    pred_clouds: list[np.ndarray],
    label_clouds: list[np.ndarray],
    point_clouds: list[np.ndarray],
    clean_clouds: list[np.ndarray],
    *,
    compute_wdist: bool,
) -> dict[str, Any]:
    rows: list[CloudMetricRow] = []
    pooled = ConfusionCounts(0, 0, 0, 0)
    for pred, y, pts, clean in zip(
        pred_clouds, label_clouds, point_clouds, clean_clouds, strict=True
    ):
        counts = confusion_counts(y, pred)
        pooled = add_counts(pooled, counts)
        recall, spec, gmean, mcc = compute_recall_specificity_gmean_mcc(y, pred)
        wdist: float | None = None
        if compute_wdist:
            gt = valid_clean_inliers(clean)
            _, _, _, _, wdist = compute_recall_specificity_gmean_mcc_wdist(
                y, pred, points=pts, gt_inliers=gt
            )
        rows.append(
            CloudMetricRow(
                recall=recall,
                specificity=spec,
                gmean=gmean,
                mcc=mcc,
                wdist=wdist,
                counts=counts,
            )
        )
    if not rows:
        raise ValueError("no clouds to evaluate")
    out: dict[str, Any] = {
        "n_clouds": len(rows),
        "recall": float(np.mean([r.recall for r in rows])),
        "specificity": float(np.mean([r.specificity for r in rows])),
        "gmean": float(np.mean([r.gmean for r in rows])),
        "mcc": float(np.mean([r.mcc for r in rows])),
        "mcc_std_clouds": float(np.std([r.mcc for r in rows])),
        "tp": pooled.tp,
        "tn": pooled.tn,
        "fp": pooled.fp,
        "fn": pooled.fn,
        "wdist": None,
    }
    if compute_wdist:
        wd = [r.wdist for r in rows]
        if any(w is None or not np.isfinite(w) for w in wd):
            raise ValueError("non-finite W-Dist; refusing nanmean")
        out["wdist"] = float(np.mean(wd))
    return out


# ---------------------------------------------------------------------------
# Models (fit on clean points only)
# ---------------------------------------------------------------------------


class SklearnScoreWrapper:
    """Wrap sklearn decision_function so that higher = more outlier-like."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def score(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        return np.asarray(-self.estimator.decision_function(pts), dtype=np.float64)


class GMMScorer:
    def __init__(self, gmm: GaussianMixture) -> None:
        self.gmm = gmm

    def score(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        # score_samples = log-likelihood; invert so higher = outlier.
        return np.asarray(-self.gmm.score_samples(pts), dtype=np.float64)


class PrimalSVDDScorer:
    """Input-space sphere: centre = clean mean; score = squared radius.

    Distinct from RBF OneClassSVM (kernel SVDD). Soft radius is applied later
    by the shared threshold rule, not by a labelled QP.
    """

    def __init__(self, center: np.ndarray) -> None:
        self.center = np.asarray(center, dtype=np.float64).reshape(1, 2)

    def score(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        d = pts - self.center
        return np.sum(d * d, axis=1)


class _PointMLP(nn.Module):
    def __init__(self, hidden: int, latent: int, *, decode: bool) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder: nn.Module | None
        if decode:
            self.decoder = nn.Sequential(
                nn.Linear(latent, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 2),
            )
        else:
            self.decoder = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        if self.decoder is None:
            return z
        return self.decoder(z)


def _torch_loader(
    x: np.ndarray, batch_size: int, rng: np.random.Generator
) -> list[np.ndarray]:
    n = len(x)
    order = rng.permutation(n)
    batches = []
    for start in range(0, n, batch_size):
        batches.append(x[order[start : start + batch_size]])
    return batches


def _train_ae(
    x: np.ndarray,
    *,
    hidden: int,
    latent: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    rng: np.random.Generator,
) -> _PointMLP:
    torch.manual_seed(int(seed))
    model = _PointMLP(hidden, latent, decode=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    xt = x.astype(np.float32)
    for _ in range(int(epochs)):
        for batch in _torch_loader(xt, batch_size, rng):
            t = torch.from_numpy(batch.astype(np.float32)).to(device)
            recon = model(t)
            loss = torch.mean((recon - t) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


class AutoencoderScorer:
    def __init__(self, model: _PointMLP, device: torch.device) -> None:
        self.model = model
        self.device = device

    def score(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(pts).to(self.device)
            recon = self.model(t).cpu().numpy()
        return np.sum((pts - recon) ** 2, axis=1).astype(np.float64)


class DeepSVDDScorer:
    def __init__(
        self, model: _PointMLP, center: np.ndarray, device: torch.device
    ) -> None:
        self.model = model
        self.center = np.asarray(center, dtype=np.float64)
        self.device = device

    def score(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(pts).to(self.device)
            z = self.model.encode(t).cpu().numpy().astype(np.float64)
        d = z - self.center.reshape(1, -1)
        return np.sum(d * d, axis=1)


def fit_ocsvm_rbf(clean: np.ndarray, *, nu: float = 0.05) -> SklearnScoreWrapper:
    """RBF One-Class SVM ≡ kernel SVDD for a Gaussian kernel."""
    x = require_clean_xy(clean)
    pipe = Pipeline(
        [
            ("sc", StandardScaler()),
            ("clf", OneClassSVM(kernel="rbf", gamma="scale", nu=float(nu))),
        ]
    )
    pipe.fit(x)
    return SklearnScoreWrapper(pipe)


def fit_svdd_primal(clean: np.ndarray) -> PrimalSVDDScorer:
    x = require_clean_xy(clean)
    return PrimalSVDDScorer(x.mean(axis=0))


def fit_elliptic(clean: np.ndarray, *, seed: int) -> SklearnScoreWrapper:
    x = require_clean_xy(clean)
    # contamination is a sklearn API default for the *internal* cutoff; we
    # ignore predict() and use scores + the shared threshold rule instead.
    model = EllipticEnvelope(
        contamination=0.05, random_state=int(seed), support_fraction=0.9
    )
    model.fit(x)
    return SklearnScoreWrapper(model)


def fit_gmm(
    clean_train: np.ndarray,
    clean_val: np.ndarray,
    *,
    n_components_grid: tuple[int, ...] = (4, 8, 16),
    seed: int,
) -> tuple[GMMScorer, int]:
    """Select component count on clean-val log-likelihood (no mixed labels)."""
    x_tr = require_clean_xy(clean_train)
    x_va = require_clean_xy(clean_val)
    best_ll = -np.inf
    best_n: int | None = None
    best_gmm: GaussianMixture | None = None
    for n in n_components_grid:
        gmm = GaussianMixture(
            n_components=int(n),
            covariance_type="full",
            random_state=int(seed),
            max_iter=200,
            n_init=1,
        )
        gmm.fit(x_tr)
        ll = float(gmm.score(x_va))
        if not np.isfinite(ll):
            raise ValueError(f"GMM n={n} produced non-finite val log-likelihood")
        if ll > best_ll:
            best_ll = ll
            best_n = int(n)
            best_gmm = gmm
    if best_gmm is None or best_n is None:
        raise RuntimeError("GMM grid produced no model")
    return GMMScorer(best_gmm), best_n


def fit_lof_novelty(
    clean: np.ndarray, *, n_neighbors: int = 20
) -> SklearnScoreWrapper:
    x = require_clean_xy(clean)
    model = LocalOutlierFactor(
        n_neighbors=int(n_neighbors), novelty=True, contamination=0.05
    )
    model.fit(x)
    return SklearnScoreWrapper(model)


def fit_autoencoder(
    clean: np.ndarray,
    *,
    seed: int,
    hidden: int = 64,
    latent: int = 8,
    epochs: int = 30,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: torch.device | None = None,
    rng: np.random.Generator | None = None,
) -> AutoencoderScorer:
    x = require_clean_xy(clean)
    dev = device or torch.device("cpu")
    gen = rng if rng is not None else np.random.default_rng(seed)
    model = _train_ae(
        x,
        hidden=hidden,
        latent=latent,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=dev,
        rng=gen,
    )
    return AutoencoderScorer(model, dev)


def fit_deep_svdd(
    clean: np.ndarray,
    *,
    seed: int,
    hidden: int = 64,
    latent: int = 8,
    pretrain_epochs: int = 10,
    svdd_epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    device: torch.device | None = None,
    rng: np.random.Generator | None = None,
) -> DeepSVDDScorer:
    """Ruff-style Deep SVDD on 2-D points (clean only).

    Pretrain an AE, freeze the hypersphere centre to the mean encoder
    embedding of clean data, then minimise distance to that centre.
    """
    x = require_clean_xy(clean)
    dev = device or torch.device("cpu")
    gen = rng if rng is not None else np.random.default_rng(seed)
    ae = _train_ae(
        x,
        hidden=hidden,
        latent=latent,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device=dev,
        rng=gen,
    )
    encoder = ae.encoder
    # Freeze centre c (Ruff et al.): mean embedding, no labels.
    ae.eval()
    with torch.no_grad():
        z = encoder(torch.from_numpy(x.astype(np.float32)).to(dev))
        center = z.mean(dim=0).detach()
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    encoder.train()
    xt = x.astype(np.float32)
    c = center
    for _ in range(int(svdd_epochs)):
        for batch in _torch_loader(xt, batch_size, gen):
            t = torch.from_numpy(batch.astype(np.float32)).to(dev)
            z_b = encoder(t)
            loss = torch.mean(torch.sum((z_b - c) ** 2, dim=1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    encoder.eval()
    net = _PointMLP(hidden, latent, decode=False).to(dev)
    net.encoder.load_state_dict(encoder.state_dict())
    net.eval()
    return DeepSVDDScorer(net, center.cpu().numpy(), dev)


def score_clouds(scorer: Scorer, clouds: list[np.ndarray]) -> list[np.ndarray]:
    return [np.asarray(scorer.score(pts), dtype=np.float64) for pts in clouds]

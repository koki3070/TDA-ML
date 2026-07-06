"""TEMP benchmark: verify intra-batch topo-loss threading (correctness + speed).

Delete after use. Compares workers=1 vs workers=8 on identical inputs:
- max abs diff of topo loss value
- max abs diff of gradients w.r.t. model parameters
- per-batch wall time for a few full train steps (fwd + bwd)
"""

from __future__ import annotations

import time

import torch

from tda_ml.config import load_config
from tda_ml.data_loader import NoisyMNISTDataset, create_data_loader
from tda_ml.models import AnisotropicOutlierClassifier
from tda_ml.config import model_kwargs_from_config
from tda_ml.losses import TopologicalLoss
from tda_ml.trainer import Trainer


def get_batches(cfg, n_clouds, batch_size):
    data_cfg = cfg["data"]
    gen = torch.Generator().manual_seed(0)
    idx = torch.randperm(60000, generator=gen)[:n_clouds]
    ds = NoisyMNISTDataset(
        root="./data", train=True, max_points=data_cfg["max_points"],
        num_outliers=data_cfg["num_outliers"], indices=idx,
        deterministic=True, noise_seed=42,
    )
    return create_data_loader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def grad_vector(model):
    return torch.cat([
        p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1)
        for p in model.parameters()
    ])


def run_once(model, trainer, topo_fn, data, clean_pc):
    model.zero_grad(set_to_none=True)
    logits, params = model(data)
    clean_pd_info = trainer._compute_clean_pd_info(clean_pc)
    loss = topo_fn(data, params, logits, clean_pd_info)
    loss.backward()
    return float(loss.detach()), grad_vector(model)


def main():
    torch.manual_seed(0)
    cfg = load_config("_conv_n100")
    device = torch.device("cpu")
    batch_size = cfg["data"]["batch_size"]

    loader = get_batches(cfg, n_clouds=batch_size * 3, batch_size=batch_size)
    batches = [(d, c) for d, _, c in loader]

    model = AnisotropicOutlierClassifier(**model_kwargs_from_config(cfg)).to(device)
    trainer = Trainer(model, cfg, device=device)

    # The multiseed driver overrides the backend to ellphi at runtime; match that
    # here so the benchmark exercises the per-sample ellphi path (not vectorized mahalanobis).
    backend = "ellphi"
    print(f"backend={backend} batch_size={batch_size} max_points={cfg['data']['max_points']}")

    topo1 = TopologicalLoss(weight=cfg["loss"]["w_topo"], distance_backend=backend,
                            ellphi_differentiable=True, topo_workers=1)
    topo8 = TopologicalLoss(weight=cfg["loss"]["w_topo"], distance_backend=backend,
                            ellphi_differentiable=True, topo_workers=8)

    # --- correctness on one batch ---
    d, c = batches[0]
    d, c = d.to(device), c.to(device)
    l1, g1 = run_once(model, trainer, topo1, d, c)
    l8, g8 = run_once(model, trainer, topo8, d, c)
    print(f"[correctness] loss w1={l1:.8f} w8={l8:.8f} |dloss|={abs(l1-l8):.3e}")
    print(f"[correctness] max|dgrad|={ (g1-g8).abs().max().item():.3e}  "
          f"max|grad|={g1.abs().max().item():.3e}")

    # --- speed: time a few full steps ---
    def timed(topo_fn, label):
        # warmup
        d0, c0 = batches[0]
        run_once(model, trainer, topo_fn, d0.to(device), c0.to(device))
        t0 = time.perf_counter()
        n = 0
        for d, c in batches:
            run_once(model, trainer, topo_fn, d.to(device), c.to(device))
            n += 1
        dt = (time.perf_counter() - t0) / n
        print(f"[speed] {label}: {dt:.2f} s/batch over {n} batches")
        return dt

    t1 = timed(topo1, "workers=1")
    t8 = timed(topo8, "workers=8")
    print(f"[speed] speedup x{t1/t8:.2f}  -> 30ep ETA "
          f"@w8 = {t8*71*30/3600:.1f} h (val negligible)")


if __name__ == "__main__":
    main()

import torch
import numpy as np
import os
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from tda_ml.checkpoint_io import extract_model_state_dict, load_torch_checkpoint
from tda_ml.data_loader import NoisyMNISTDataset, create_data_loader
from tda_ml.models import AnisotropicOutlierClassifier
from tda_ml.dbscan import apply_anisotropic_dbscan
from tda_ml.metrics import compute_recall_specificity_gmean_mcc

def calculate_metrics(labels_gt, labels_pred):
    return compute_recall_specificity_gmean_mcc(labels_gt, labels_pred)

def run_benchmark(model_path, seeds, device, output_dir):
    print(f"\nEvaluating Model: {model_path}")
    
    # Optimized Parameters
    EPS = 0.4
    MIN_SAMPLES = 4
    MIN_B = 0.3
    
    results = []
    
    # Load Model
    model = AnisotropicOutlierClassifier()
    checkpoint = load_torch_checkpoint(model_path, map_location=device)
    model.load_state_dict(extract_model_state_dict(checkpoint), strict=False)
    model.to(device)
    model.eval()

    for seed in seeds:
        print(f"Seed {seed}...")
        
        # Prepare Data: MNIST 150 inliers + 150 outliers (50/50 Ratio)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(10000, generator=generator)[:500] 
        
        test_dataset = NoisyMNISTDataset(
            root='./data',
            train=False,
            max_points=150,     # Slide actual condition?
            num_outliers=150,   # Slide actual condition?
            indices=indices,
            deterministic=True,
            noise_seed=seed,
            noise_std=0.01
        )
        test_loader = create_data_loader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        all_preds = []
        all_gt = []
        
        with torch.no_grad():
            for data, labels, _ in tqdm(test_loader, desc=f"Seed {seed}", leave=False):
                data = data.to(device)
                labels = labels.cpu().numpy().flatten()
                
                logits, params = model(data)
                
                points_np = data.squeeze(0).cpu().numpy()
                params_np = params.squeeze(0).cpu().numpy()
                
                # Apply Optimized Clamping
                params_np[:, 3] = np.maximum(params_np[:, 3], MIN_B)
                
                predicted_labels = apply_anisotropic_dbscan(
                    points_np, params_np, 
                    eps=EPS, 
                    min_samples=MIN_SAMPLES,
                    metric='max'
                )
                preds = (predicted_labels == -1).astype(int)
                
                all_preds.extend(preds)
                all_gt.extend(labels)
        
        r, s, g, m = calculate_metrics(all_gt, all_preds)
        results.append({'Seed': seed, 'Recall': r, 'Specificity': s, 'G-Mean': g, 'MCC': m})
        print(f"  MCC: {m:.4f}, G-Mean: {g:.4f}")

    df = pd.DataFrame(results)
    summary = df.agg({
        'Recall': ['mean', 'std'],
        'Specificity': ['mean', 'std'],
        'G-Mean': ['mean', 'std'],
        'MCC': ['mean', 'std']
    })
    return df, summary

def parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Non-official MNIST 50/50 benchmark. Requires a training run directory "
            "with best_model.pth and final_model.pth (not shipped in this repo)."
        )
    )
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Directory containing best_model.pth and final_model.pth.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/mnist_5050_benchmark",
        help="Directory for summary CSV files.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [0, 42, 123, 2024, 789]

    run_dir = args.run_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Reset to the TRUE slide condition determined by investigation
    max_pts = 50
    num_out = 50
    
    print(f"Benchmark Condition: Signal={max_pts}, Noise={num_out} (50/50 Ratio)")

    # Evaluate best and final checkpoints
    best_path = os.path.join(run_dir, "best_model.pth")
    final_path = os.path.join(run_dir, "final_model.pth")
    for path in (best_path, final_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Expected checkpoint: {path}")

    v73b_df, v73b_sum = run_benchmark_ext(best_path, seeds, device, max_pts, num_out)
    print("\n--- Best checkpoint statistics ---")
    print(v73b_sum)

    v73f_df, v73f_sum = run_benchmark_ext(final_path, seeds, device, max_pts, num_out)
    print("\n--- Final checkpoint statistics ---")
    print(v73f_sum)

    v73b_sum.to_csv(os.path.join(output_dir, "best_summary.csv"))
    v73f_sum.to_csv(os.path.join(output_dir, "final_summary.csv"))


def run_benchmark_ext(model_path, seeds, device, max_pts, num_out):
    print(f"\nEvaluating: {model_path}")
    # OPTIMIZED Hyperparams from tuner
    EPS_ELL = 0.4
    MS_ELL = 5 # Changed from 4
    MIN_B = 0.3
    USE_PROBS = True
    
    # Euclidean Hyperparams
    EUC_EPS_LIST = [0.1, 0.15, 0.2, 0.25, 0.3]
    MS_EUC = 4

    model = AnisotropicOutlierClassifier()
    checkpoint = load_torch_checkpoint(model_path, map_location=device)
    model.load_state_dict(extract_model_state_dict(checkpoint), strict=False)
    model.to(device)
    model.eval()

    results = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(10000, generator=generator)[:500]
        test_dataset = NoisyMNISTDataset(root='./data', train=False, max_points=max_pts, num_outliers=num_out, indices=indices, deterministic=True, noise_seed=seed)
        test_loader = create_data_loader(test_dataset, batch_size=1, shuffle=False)
        
        all_pts, all_gt, all_params, all_probs = [], [], [], []
        with torch.no_grad():
            for data, labels, _ in tqdm(test_loader, desc=f"Seed {seed}", leave=False):
                data = data.to(device)
                logits, params = model(data)
                
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().flatten()
                all_pts.append(data.squeeze(0).cpu().numpy())
                all_params.append(params.squeeze(0).cpu().numpy())
                all_probs.append(probs)
                all_gt.extend(labels.cpu().numpy().flatten())

        # 1. ADBSCAN (Elliptical + Prob Weighting)
        preds_ell = []
        for i in range(len(all_pts)):
            p_np = all_pts[i]
            param_np = all_params[i].copy()
            param_np[:, 3] = np.maximum(param_np[:, 3], MIN_B)
            prob_np = all_probs[i] if USE_PROBS else None
            
            lbls = apply_anisotropic_dbscan(points=p_np, params=param_np, eps=EPS_ELL, min_samples=MS_ELL, metric='max', probs=prob_np)
            preds_ell.extend((lbls == -1).astype(int))
        r_ell, s_ell, g_ell, m_ell = calculate_metrics(all_gt, preds_ell)
        
        # 2. Euclidean DBSCAN (Best over eps sweep)
        best_m_euc = -1
        best_stats_euc = (0,0,0,0)
        for eps_euc in EUC_EPS_LIST:
            preds_euc = []
            for i in range(len(all_pts)):
                db = DBSCAN(eps=eps_euc, min_samples=MS_EUC)
                lbls = db.fit_predict(all_pts[i])
                preds_euc.extend((lbls == -1).astype(int))
            metrics = calculate_metrics(all_gt, preds_euc)
            if metrics[3] > best_m_euc:
                best_m_euc = metrics[3]
                best_stats_euc = metrics
        
        results.append({
            'Seed': seed,
            'MCC_ADBSCAN': m_ell,
            'GMean_ADBSCAN': g_ell,
            'MCC_Euclidean': best_stats_euc[3],
            'GMean_Euclidean': best_stats_euc[2]
        })

    df = pd.DataFrame(results)
    summary = df.agg({
        'MCC_ADBSCAN': ['mean', 'std'],
        'GMean_ADBSCAN': ['mean', 'std'],
        'MCC_Euclidean': ['mean', 'std'],
        'GMean_Euclidean': ['mean', 'std']
    })
    return df, summary

if __name__ == "__main__":
    main()

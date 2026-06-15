import torch
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import DBSCAN

from tda_ml.checkpoint_io import extract_model_state_dict, load_torch_checkpoint
from tda_ml.data_loader import NoisyMNISTDataset, create_data_loader
from tda_ml.models import AnisotropicOutlierClassifier
from tda_ml.dbscan import calculate_anisotropic_distance_matrix
from tda_ml.metrics import (
    compute_recall_specificity_gmean_mcc,
    compute_recall_specificity_gmean_mcc_wdist,
)

def calculate_metrics(labels_gt, labels_pred):
    return compute_recall_specificity_gmean_mcc(labels_gt, labels_pred)

def apply_hybrid_adbscan(points, params, probs, eps, min_samples, prob_cutoff, scaling_power, min_b):
    params_clamped = params.copy()
    params_clamped[:, 3] = np.maximum(params_clamped[:, 3], min_b)
    
    mask = (probs > prob_cutoff)
    inlier_probs = 1.0 - probs
    inlier_probs = np.clip(inlier_probs, 1e-4, 1.0)
    scale_factors = (1.0 / inlier_probs) ** scaling_power
    scale_factors[mask] = 1e6 
    
    pseudo_probs = 1.0 - (1.0 / scale_factors)
    dist_matrix = calculate_anisotropic_distance_matrix(points, params_clamped, metric='max', probs=pseudo_probs)
    
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(dist_matrix)
    labels[mask] = -1
    return (labels == -1).astype(int)

def run_euclidean_baseline(loader, seeds, device):
    print("\nEvaluating: Euclidean DBSCAN (Baseline)")
    best_eps = 0.2
    min_samples = 4
    results = []
    
    for seed in seeds:
        all_preds, all_gt, all_wdists = [], [], []
        # Reset loader seed? In this loop we trust indices match
        for data, labels, clean_pc in tqdm(loader, desc=f"Seed {seed}", leave=False):
            pts = data.squeeze(0).cpu().numpy()
            db = DBSCAN(eps=best_eps, min_samples=min_samples)
            lbls = db.fit_predict(pts)
            mask_noise = (lbls == -1).astype(int)
            
            all_preds.extend(mask_noise)
            all_gt.extend(labels.cpu().numpy().flatten())
            
            gt_inliers = clean_pc.squeeze(0).cpu().numpy()
            gt_inliers = gt_inliers[np.any(gt_inliers != 0, axis=1)]
            _, _, _, _, wd = compute_recall_specificity_gmean_mcc_wdist(
                labels.cpu().numpy().flatten(),
                mask_noise,
                points=pts,
                gt_inliers=gt_inliers,
            )
            all_wdists.append(wd)
            
        r, s, g, m = calculate_metrics(all_gt, all_preds)
        w_mean = np.mean(all_wdists)
        results.append({'Seed': seed, 'Recall': r, 'Specificity': s, 'G-Mean': g, 'MCC': m, 'W-Dist': w_mean})
        
    df = pd.DataFrame(results)
    summary = df.agg({m: ['mean', 'std'] for m in ['Recall', 'Specificity', 'G-Mean', 'MCC', 'W-Dist']})
    print("\n--- EUCLIDEAN DBSCAN STATISTICS ---")
    print(summary)
    return summary

def parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Non-official hybrid ADBSCAN benchmark (requires --checkpoint)."
    )
    p.add_argument("--checkpoint", type=str, required=True, help="Path to final_model.pth or best_model.pth")
    p.add_argument("--output-dir", type=str, default="outputs/mnist_5050_hybrid_benchmark")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [0, 42, 123, 2024, 789]

    EPS = 0.4
    MS = 4
    CUTOFF = 0.5
    POWER = 1.8
    MIN_B = 0.3

    model_path = args.checkpoint
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    model = AnisotropicOutlierClassifier()
    checkpoint = load_torch_checkpoint(model_path, map_location=device)
    model.load_state_dict(extract_model_state_dict(checkpoint), strict=False)
    model.to(device)
    model.eval()

    results = []
    print(f"Final Benchmark [Hybrid ADBSCAN]: cutoff={CUTOFF}, power={POWER}")

    for seed in seeds:
        print(f"Seed {seed}...")
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(10000, generator=generator)[:1000] 
        dataset = NoisyMNISTDataset(root='./data', train=False, max_points=50, num_outliers=50, indices=indices, deterministic=True, noise_seed=seed)
        loader = create_data_loader(dataset, batch_size=1, shuffle=False)
        
        all_preds = []
        all_gt = []
        all_wdists = []
        with torch.no_grad():
            for data, labels, clean_pc in tqdm(loader, desc=f"Seed {seed}", leave=False):
                data = data.to(device)
                logits, params = model(data)
                
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().flatten()
                pts = data.squeeze(0).cpu().numpy()
                pms = params.squeeze(0).cpu().numpy()
                
                mask_noise = apply_hybrid_adbscan(pts, pms, probs, EPS, MS, CUTOFF, POWER, MIN_B)
                all_preds.extend(mask_noise)
                all_gt.extend(labels.cpu().numpy().flatten())
                
                gt_inliers = clean_pc.squeeze(0).cpu().numpy()
                gt_inliers = gt_inliers[np.any(gt_inliers != 0, axis=1)]
                _, _, _, _, wd = compute_recall_specificity_gmean_mcc_wdist(
                    labels.cpu().numpy().flatten(),
                    mask_noise,
                    points=pts,
                    gt_inliers=gt_inliers,
                )
                all_wdists.append(wd)
        
        r, s, g, m = calculate_metrics(all_gt, all_preds)
        w_mean = np.mean(all_wdists)
        results.append({'Seed': seed, 'Recall': r, 'Specificity': s, 'G-Mean': g, 'MCC': m, 'W-Dist': w_mean})

    df = pd.DataFrame(results)
    summary = df.agg({m: ['mean', 'std'] for m in ['Recall', 'Specificity', 'G-Mean', 'MCC', 'W-Dist']})
    print("\n--- HYBRID ADBSCAN FINAL STATISTICS ---")
    print(summary)
    summary.to_csv(os.path.join(output_dir, 'hybrid_final_summary.csv'))

    # Run Euclidean Baseline for direct comparison
    # Use the same seeds and a dummy loader with same settings
    # For simplicity, we can recreate the loader in a loop as above
    euc_results = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(10000, generator=generator)[:1000]
        dataset = NoisyMNISTDataset(root='./data', train=False, max_points=50, num_outliers=50, indices=indices, deterministic=True, noise_seed=seed)
        loader = create_data_loader(dataset, batch_size=1, shuffle=False)
        all_preds_e, all_gt_e, all_wdists_e = [], [], []
        for data, labels, clean_pc in tqdm(loader, desc=f"Euc-Seed {seed}", leave=False):
            pts = data.squeeze(0).cpu().numpy()
            db = DBSCAN(eps=0.2, min_samples=4)
            lbls = db.fit_predict(pts)
            mask_e = (lbls == -1).astype(int)
            all_preds_e.extend(mask_e)
            all_gt_e.extend(labels.cpu().numpy().flatten())
            gt_in_e = clean_pc.squeeze(0).cpu().numpy()
            gt_in_e = gt_in_e[np.any(gt_in_e != 0, axis=1)]
            _, _, _, _, wd_e = compute_recall_specificity_gmean_mcc_wdist(
                labels.cpu().numpy().flatten(),
                mask_e,
                points=pts,
                gt_inliers=gt_in_e,
            )
            all_wdists_e.append(wd_e)
        re, se, ge, me = calculate_metrics(all_gt_e, all_preds_e)
        we = np.mean(all_wdists_e)
        euc_results.append({'Seed': seed, 'Recall': re, 'Specificity': se, 'G-Mean': ge, 'MCC': me, 'W-Dist': we})
    
    df_e = pd.DataFrame(euc_results)
    summary_e = df_e.agg({m: ['mean', 'std'] for m in ['Recall', 'Specificity', 'G-Mean', 'MCC', 'W-Dist']})
    print("\n--- EUCLIDEAN DBSCAN FINAL STATISTICS ---")
    print(summary_e)
    summary_e.to_csv(os.path.join(output_dir, 'euclidean_final_summary.csv'))

if __name__ == "__main__":
    main()

"""
Non-official noise-sensitivity plot (sparse sine wave). Requires --model checkpoint.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tda_ml.checkpoint_io import extract_model_state_dict, load_torch_checkpoint
from tda_ml.models import AnisotropicOutlierClassifier
from sklearn.metrics import recall_score, matthews_corrcoef, confusion_matrix
from sklearn.cluster import DBSCAN
import seaborn as sns

def generate_sparse_sine_wave(n_signal=50, n_noise=50, noise_seed=None):
    if noise_seed is not None:
        np.random.seed(noise_seed)
        
    # Signal: One period of sine wave [0, 2pi]
    x_signal = np.linspace(0, 2*np.pi, n_signal)
    y_signal = np.sin(x_signal)
    signal = np.stack([x_signal, y_signal], axis=1)
    
    # Noise: Uniform in bounding box [0, 2pi] x [-1.5, 1.5]
    x_noise = np.random.uniform(0, 2*np.pi, n_noise)
    y_noise = np.random.uniform(-1.5, 1.5, n_noise)
    noise = np.stack([x_noise, y_noise], axis=1)
    
    data = np.concatenate([signal, noise])
    
    # Normalize to [-1, 1] for model input
    # Center and scale based on theoretical bounds approx
    # X: [0, 6.28] -> Center 3.14, Scale 3.14
    # Y: [-1.5, 1.5] -> Center 0, Scale 1.5
    data_norm = np.copy(data)
    data_norm[:, 0] = (data[:, 0] - np.pi) / np.pi
    data_norm[:, 1] = data[:, 1] / 1.5
    
    labels = np.zeros(len(data))
    labels[n_signal:] = 1 # 0: Inlier, 1: Outlier
    
    return data_norm, labels, data # Return original scale too for visualization if needed

def compute_elliptical_distance(data_np, ellipse_params_np):
    data_np.shape[0]
    
    # Extract params
    axes = ellipse_params_np[:, 2:4]
    angle = ellipse_params_np[:, 4]
    
    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    
    # Mahalanobis Metric Matrix M = R * S^-2 * R.T
    # We compute M elements m00, m01, m11
    # m00 = (cos/a)^2 + (sin/b)^2
    # m11 = (sin/a)^2 + (cos/b)^2
    # m01 = cos*sin*(1/a^2 - 1/b^2)
    
    inv_a2 = 1.0 / (axes[:, 0]**2)
    inv_b2 = 1.0 / (axes[:, 1]**2)
    
    m00 = cos_t**2 * inv_a2 + sin_t**2 * inv_b2
    m11 = sin_t**2 * inv_a2 + cos_t**2 * inv_b2
    m01 = cos_t * sin_t * (inv_a2 - inv_b2)
    
    # Compute pairwise distance
    # d^2(x, y) = (x-y)^T M_x (x-y)  [using M of point x]
    # Actually, the paper might define d(x,y) using M_x or max(M_x, M_y). 
    # Current codebase uses max(d(x->y), d(y->x))
    
    diff = data_np[:, np.newaxis, :] - data_np[np.newaxis, :, :] # (N, N, 2)
    dx = diff[:, :, 0]
    dy = diff[:, :, 1]
    
    # Dist from i (using M_i)
    # m00 is (N,), broadcast to (N, 1) or (N, N)
    dist_sq_from_i = (m00[:, np.newaxis] * dx**2 + 
                      2 * m01[:, np.newaxis] * dx * dy + 
                      m11[:, np.newaxis] * dy**2)
                      
    dist_sq_from_j = (m00[np.newaxis, :] * dx**2 +
                      2 * m01[np.newaxis, :] * dx * dy +
                      m11[np.newaxis, :] * dy**2)
                      
    dist_sq = np.maximum(dist_sq_from_i, dist_sq_from_j)
    return np.sqrt(np.maximum(dist_sq, 0))

def run_experiment(model_path, noise_levels=[50, 75, 100, 125, 150], n_trials=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = AnisotropicOutlierClassifier()
    checkpoint = load_torch_checkpoint(model_path, map_location=device)
    model.load_state_dict(extract_model_state_dict(checkpoint), strict=False)
    model.to(device)
    model.eval()
    
    results = []
    
    for n_noise in noise_levels:
        print(f"Testing Noise Level: {n_noise}")
        for t in range(n_trials):
            data_norm, gt_labels, _ = generate_sparse_sine_wave(n_signal=50, n_noise=n_noise, noise_seed=t)
            
            # 1. Elliptical DBSCAN
            data_tensor = torch.tensor(data_norm, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, params = model(data_tensor)
            
            ellipse_params_np = params.cpu().squeeze(0).numpy()
            
            # Compute Distance Matrix
            dist_matrix = compute_elliptical_distance(data_norm, ellipse_params_np)
            
            # DBSCAN
            # Paper likely uses eps=1.0 for normalized elliptical distance (Mahalanobis dist <= 1 is the ellipse boundary)
            # But let's verify. Code usually uses eps=0.2 for Euclidean?
            # For Elliptical: points on the boundary have distance 1.0.
            # So eps=1.0 is the natural choice if ellipse prediction is perfect.
            # Let's try eps=1.0
            db_ell = DBSCAN(eps=1.0, min_samples=3, metric='precomputed')
            clusters_ell = db_ell.fit_predict(dist_matrix)
            pred_ell = (clusters_ell == -1).astype(int)
            
            # Metrics Elliptical
            recall_score(gt_labels, pred_ell, pos_label=0) # Recall for INLIERS (0)
            # Wait, usually "Recall" in anomaly detection is for outliers (1).
            # But here users care about connecting inliers.
            # Let's stick to standard def: 0=Negative(Inlier), 1=Positive(Outlier)
            # User wants "Recall" of Inliers? "Signal Recall".
            # sklearn recall is for pos_label=1 by default.
            # Let's compute manually to be sure.
            
            tn, fp, fn, tp_outlier = confusion_matrix(gt_labels, pred_ell).ravel()
            # Inlier Recall (TP_inlier / Actual_Inlier) = TN / (TN + FP) -> This is Specificity for Outlier detection
            # Outlier Recall (TP_outlier / Actual_Outlier) = TP / (TP + FN) -> This is "Recall"
            
            # User previously used: Recall (of Outliers?), Specificity, MCC.
            # For the chart, let's store MCC.
            mcc_ell = matthews_corrcoef(gt_labels, pred_ell)
            
            results.append({
                'Method': 'Elliptical',
                'Noise': n_noise,
                'MCC': mcc_ell,
                'Trial': t
            })
            
            # 2. Euclidean DBSCAN
            # Need to tune eps for Euclidean? Or use fixed reasonable?
            # Signal dist ~ 0.12 (2pi/50).
            # If we use eps=0.15, we connect signal.
            # But noise density increases.
            db_euc = DBSCAN(eps=0.15, min_samples=3)
            clusters_euc = db_euc.fit_predict(data_norm)
            pred_euc = (clusters_euc == -1).astype(int)
            
            mcc_euc = matthews_corrcoef(gt_labels, pred_euc)
            
            results.append({
                'Method': 'Euclidean',
                'Noise': n_noise,
                'MCC': mcc_euc,
                'Trial': t
            })
            
    df = pd.DataFrame(results)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='Noise', y='MCC', hue='Method', marker='o')
    plt.title('Noise Sensitivity: Elliptical vs Euclidean DBSCAN\n(Signal=50, Sine Wave)')
    plt.ylabel('MCC')
    plt.xlabel('Number of Noise Points')
    plt.grid(True)
    plt.savefig('outputs/metrics_vs_noise_level.png')
    print("Saved outputs/metrics_vs_noise_level.png")
    
    # Save CSV
    df.to_csv('outputs/noise_sensitivity_results.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    
    run_experiment(args.model)

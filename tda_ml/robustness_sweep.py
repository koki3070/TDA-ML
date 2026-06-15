import argparse
import os
from pathlib import Path

import torch
import numpy as np
import datetime
import optuna
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import matthews_corrcoef
from joblib import Parallel, delayed

from tda_ml.checkpoint_io import extract_model_state_dict, load_torch_checkpoint
from tda_ml.config import load_config
from tda_ml.data_loader import NoisyMNISTDataset, create_data_loader
from tda_ml.models import AnisotropicOutlierClassifier
from tda_ml.dbscan import apply_anisotropic_dbscan
from tda_ml.metrics import compute_recall_specificity_gmean_mcc_wdist

_REPO_ROOT = Path(__file__).resolve().parents[1]

def calculate_metrics(labels_gt, labels_pred, points=None, gt_inliers=None):
    return compute_recall_specificity_gmean_mcc_wdist(
        labels_gt,
        labels_pred,
        points=points,
        gt_inliers=gt_inliers,
        empty_pred_wdist=9.99,
    )

def run_single_eval(item, method_name, params):
    points = item['points']
    labels_gt = item['labels']
    probs = item['probs']
    
    if method_name == 'DeepEllipticalPH':
        preds = (probs > params['threshold']).astype(int)
    elif method_name == 'ADBSCAN':
        p_np = item['params'].copy()
        if 'min_b' in params:
            p_np[:, 3] = np.maximum(p_np[:, 3], params['min_b'])
        db = apply_anisotropic_dbscan(points, p_np, eps=params['eps'], min_samples=params['min_samples'])
        preds = (db == -1).astype(int)
    elif method_name == 'StandardDBSCAN':
        db = DBSCAN(eps=params['eps'], min_samples=params['min_samples']).fit(points)
        preds = (db.labels_ == -1).astype(int)
    elif method_name == 'LOF':
        preds = (LocalOutlierFactor(n_neighbors=params['n_neighbors'], contamination=params['contamination']).fit_predict(points) == -1).astype(int)
    elif method_name == 'IsoForest':
        preds = (IsolationForest(contamination=params['contamination'], random_state=42).fit_predict(points) == -1).astype(int)
    elif method_name == 'Hybrid':
        mask = probs < params['threshold']
        if mask.sum() == 0:
            preds = np.ones(len(points))
        else:
            db = apply_anisotropic_dbscan(points[mask], item['params'][mask], eps=params['eps'], min_samples=params['min_samples'])
            preds = np.ones(len(points))
            preds[np.where(mask)[0][db >= 0]] = 0
    
    return preds, labels_gt

def run_evaluation_with_params(all_data, method_name, params):
    results = Parallel(n_jobs=-1)(delayed(run_single_eval)(item, method_name, params) for item in all_data)
    all_preds, all_gt = [], []
    for preds, labels_gt in results:
        all_preds.extend(preds)
        all_gt.extend(labels_gt)
    return matthews_corrcoef(all_gt, all_preds)

def get_data(seed, model, device, config, size, num_outliers, split='test'):
    max_points = 50
    
    generator = torch.Generator().manual_seed(seed)
    if split == 'val':
        indices = torch.arange(0, size)
    else:
        all_indices = torch.randperm(10000, generator=generator)
        indices = all_indices[all_indices >= 1000][:size]
        
    dataset = NoisyMNISTDataset(root='./data', train=False, max_points=max_points, num_outliers=num_outliers, indices=indices, deterministic=True, noise_seed=seed)
    loader = create_data_loader(dataset, batch_size=1, shuffle=False)
    
    all_data = []
    with torch.no_grad():
        for data, labels, clean_pc in loader:
            data = data.to(device)
            logits, params = model(data)
            
            all_data.append({
                'points': data.squeeze(0).cpu().numpy(),
                'params': params.squeeze(0).cpu().numpy(),
                'labels': labels.cpu().numpy().flatten(),
                'gt_inliers': clean_pc.squeeze(0).cpu().numpy(),
                'probs': torch.sigmoid(logits).squeeze(0).cpu().numpy().flatten()
            })
    return all_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--n_trials_opt', type=int, default=100)
    args = parser.parse_args()
    
    config = load_config(args.config, project_root=_REPO_ROOT)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AnisotropicOutlierClassifier()
    checkpoint = load_torch_checkpoint(args.model_path, map_location=device)
    model.load_state_dict(extract_model_state_dict(checkpoint), strict=False)
    model.to(device).eval()

    noise_levels = [50, 100, 200, 300]
    seeds = [42, 123, 2024, 789, 1024]
    methods = ['DeepEllipticalPH', 'StandardDBSCAN', 'ADBSCAN', 'IsoForest', 'LOF', 'Hybrid']
    
    all_results = []

    for n_noise in noise_levels:
        print(f"\n================ NOISE LEVEL: {n_noise} ================")
        val_data = get_data(42, model, device, config, 500, n_noise, split='val')
        
        optimized_params = {}
        for m in methods:
            print(f"Optimizing {m}...")
            def objective(trial):
                if m == 'DeepEllipticalPH':
                    p = {'threshold': trial.suggest_float('threshold', 0.01, 0.95)}
                elif m == 'ADBSCAN':
                    p = {
                        'eps': trial.suggest_float('eps', 0.05, 5.0, log=True),
                        'min_samples': trial.suggest_int('min_samples', 2, 10),
                        'min_b': trial.suggest_float('min_b', 0.01, 0.4),
                    }
                elif m == 'StandardDBSCAN':
                    p = {
                        'eps': trial.suggest_float('eps', 0.05, 5.0, log=True),
                        'min_samples': trial.suggest_int('min_samples', 2, 10),
                    }
                elif m == 'LOF':
                    p = {
                        'n_neighbors': trial.suggest_int('n_neighbors', 5, 50),
                        'contamination': trial.suggest_float('contamination', 0.05, 0.5),
                    }
                elif m == 'IsoForest':
                    p = {'contamination': trial.suggest_float('contamination', 0.05, 0.5)}
                elif m == 'Hybrid':
                    p = {
                        'threshold': trial.suggest_float('threshold', 0.01, 0.5),
                        'eps': trial.suggest_float('eps', 0.1, 2.0),
                        'min_samples': trial.suggest_int('min_samples', 2, 6),
                    }
                return run_evaluation_with_params(val_data, m, p)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=args.n_trials_opt)
            optimized_params[m] = study.best_params
            print(f"  -> Best MCC for {m}: {study.best_value:.4f}")

        for seed in seeds:
            print(f"Evaluating Seed {seed}...")
            test_data = get_data(seed, model, device, config, 500, n_noise, split='test')
            
            for m in methods:
                p_cfg = optimized_params[m]
                
                def eval_item(item):
                    p_np = item['params'].copy()
                    if m == 'DeepEllipticalPH':
                        preds = (item['probs'] > p_cfg['threshold']).astype(int)
                    elif m == 'ADBSCAN':
                        p_np[:, 3] = np.maximum(p_np[:, 3], p_cfg['min_b'])
                        db = apply_anisotropic_dbscan(
                            item['points'], p_np, eps=p_cfg['eps'], min_samples=p_cfg['min_samples']
                        )
                        preds = (db == -1).astype(int)
                    elif m == 'StandardDBSCAN':
                        db = DBSCAN(eps=p_cfg['eps'], min_samples=p_cfg['min_samples']).fit(item['points'])
                        preds = (db.labels_ == -1).astype(int)
                    elif m == 'LOF':
                        preds = (
                            LocalOutlierFactor(
                                n_neighbors=p_cfg['n_neighbors'],
                                contamination=p_cfg['contamination'],
                            ).fit_predict(item['points'])
                            == -1
                        ).astype(int)
                    elif m == 'IsoForest':
                        preds = (
                            IsolationForest(
                                contamination=p_cfg['contamination'], random_state=42
                            ).fit_predict(item['points'])
                            == -1
                        ).astype(int)
                    elif m == 'Hybrid':
                        # Hybrid: Filter by prob, then cluster
                        mask = item['probs'] < p_cfg['threshold']  # Potential inliers
                        if mask.sum() == 0:
                            preds = np.ones(len(item['points']))
                        else:
                            db = apply_anisotropic_dbscan(
                                item['points'][mask],
                                item['params'][mask],
                                eps=p_cfg['eps'],
                                min_samples=p_cfg['min_samples'],
                            )
                            preds = np.ones(len(item['points']))
                            preds[np.where(mask)[0][db >= 0]] = 0  # Cluster members are inliers
                    return calculate_metrics(item['labels'], preds, item['points'], item['gt_inliers'])

                results_seed = Parallel(n_jobs=-1)(delayed(eval_item)(item) for item in test_data)
                res_np = np.array(results_seed)
                
                all_results.append({
                    'Noise': n_noise,
                    'Seed': seed,
                    'Method': m,
                    'MCC': np.nanmean(res_np[:, 3]),
                    'GMean': np.nanmean(res_np[:, 2]),
                    'WDist': np.nanmean(res_np[:, 4])
                })

    # Save and Plot
    os.makedirs('important_results', exist_ok=True)
    df = pd.DataFrame(all_results)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f'important_results/robustness_sweep_v73_tuned_{timestamp}.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Plotting
    for metric in ['MCC', 'GMean', 'WDist']:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='Noise', y=metric, hue='Method', marker='o', err_style='band')
        plt.title(f'Robustness to Noise Levels: {metric}')
        plt.xlabel('Number of Noise Points (Inliers=50)')
        plt.ylabel(metric)
        plt.grid(True)
        plt.savefig(f'important_results/robustness_{metric.lower()}_{timestamp}.png')
        print(f"Saved important_results/robustness_{metric.lower()}_{timestamp}.png")

if __name__ == "__main__":
    main()

import torch
import pandas as pd
import os
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import datetime

from tda_ml.data_loader import NoisyMNISTDataset, create_data_loader
from tda_ml.metrics import compute_recall_specificity_gmean_mcc

def calculate_metrics(labels_gt, labels_pred):
    """
    Calculate Recall, Specificity, G-Mean, and MCC.
    labels_gt: 0=Inlier, 1=Outlier
    labels_pred: 0=Inlier, 1=Outlier
    """
    return compute_recall_specificity_gmean_mcc(labels_gt, labels_pred)

def run_lof(points, n_neighbors, contamination):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(points)
    preds = (y_pred == -1).astype(int)
    return preds

def run_iso_forest(points, contamination, seed):
    iso = IsolationForest(contamination=contamination, random_state=seed)
    y_pred = iso.fit_predict(points)
    preds = (y_pred == -1).astype(int)
    return preds

def evaluate_method(all_data, method_name, run_func, params, seed=None):
    all_preds = []
    all_gt = []
    
    for item in all_data:
        points = item['points']
        gt_labels = item['labels']
        
        if method_name == 'LOF':
            preds = run_func(points, params['n_neighbors'], params['contamination'])
        elif method_name == 'Isolation Forest':
            preds = run_func(points, params['contamination'], seed)
        
        all_preds.extend(preds)
        all_gt.extend(gt_labels)
        
    return calculate_metrics(all_gt, all_preds)

def main():
    seeds = [0, 42, 123, 2024, 2025]
    
    # Phase 1 Settings
    MAX_POINTS = 150
    NUM_OUTLIERS = 20
    NOISE_STD = 0.01
    TEST_SIZE = 1000
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"outputs/benchmark_baselines/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Starting Benchmark: LOF & Isolation Forest")
    print(f"Seeds: {seeds}")
    print(f"Output Directory: {output_dir}")
    print(f"Settings: N={MAX_POINTS}, Outliers={NUM_OUTLIERS}, Noise={NOISE_STD}")
    
    results = []
    
    for seed in seeds:
        print(f"\n--- Processing Seed {seed} ---")
        
        # Prepare Data
        generator = torch.Generator().manual_seed(seed)
        # Fix indices for consistency across methods within this seed
        # Note: NoisyMNISTDataset handles randomness via its own seed if deterministic=True
        # But we want to ensure we're testing on the 'test set' equivalent.
        # In compare_methods.py, it picks 1000 indices from 10000. 
        # Here we will just use 1000 random samples from the dataset as 'test'.
        # To strictly follow '5 trials (different random seeds)', we should change the data generation seed.
        
        # The prompt says: "5 trials (different random seeds)".
        # This usually means the dataset generation/noise is seeded differently, OR the method's PRNG is seeded differently.
        # For NoisyMNIST, the noise IS the dataset variation.
        
        # We will use 'seed' for both dataset generation noise and method PRNG (for IsoForest).
        
        # Generate random indices for this trial to simulate "different data realization" or just regular test set variation
        indices = torch.randperm(10000, generator=generator)[:TEST_SIZE]
        
        test_dataset = NoisyMNISTDataset(
            root='./data',
            train=False,
            max_points=MAX_POINTS,
            num_outliers=NUM_OUTLIERS,
            indices=indices,
            deterministic=True, # Ensure reproducibility for this specific seed run
            noise_seed=seed,
            noise_std=NOISE_STD
        )
        
        test_loader = create_data_loader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # Pre-load data
        print("Pre-loading data...")
        all_data = []
        for data, labels, _ in tqdm(test_loader, desc="Loading Data"):
            points = data.squeeze(0).cpu().numpy()
            gt_labels = labels.cpu().numpy().flatten()
            all_data.append({'points': points, 'labels': gt_labels})

        # --- LOF Grid Search ---
        print("Running LOF...")
        lof_neighbors = [10, 20, 30]
        lof_cont = [0.1, 0.15, 'auto']
        
        best_lof_mcc = -1
        best_lof_res = None
        best_lof_params = None
        
        for n_neigh in lof_neighbors:
            for cont in lof_cont:
                r, s, g, m = evaluate_method(all_data, 'LOF', run_lof, {'n_neighbors': n_neigh, 'contamination': cont})
                if m > best_lof_mcc:
                    best_lof_mcc = m
                    best_lof_res = (r, s, g, m)
                    best_lof_params = {'n_neighbors': n_neigh, 'contamination': cont}
        
        results.append({
            'Seed': seed,
            'Method': 'LOF',
            'Recall': best_lof_res[0],
            'Specificity': best_lof_res[1],
            'G-Mean': best_lof_res[2],
            'MCC': best_lof_res[3],
            'BestParams': str(best_lof_params)
        })
        print(f"LOF Best MCC: {best_lof_mcc:.4f}")

        # --- Isolation Forest Grid Search ---
        print("Running Isolation Forest...")
        iso_cont = [0.05, 0.1, 0.15, 'auto']
        
        best_iso_mcc = -1
        best_iso_res = None
        best_iso_params = None
        
        for cont in iso_cont:
            r, s, g, m = evaluate_method(all_data, 'Isolation Forest', run_iso_forest, {'contamination': cont}, seed=seed)
            if m > best_iso_mcc:
                best_iso_mcc = m
                best_iso_res = (r, s, g, m)
                best_iso_params = {'contamination': cont}
        
        results.append({
            'Seed': seed,
            'Method': 'Isolation Forest',
            'Recall': best_iso_res[0],
            'Specificity': best_iso_res[1],
            'G-Mean': best_iso_res[2],
            'MCC': best_iso_res[3],
            'BestParams': str(best_iso_params)
        })
        print(f"IsoForest Best MCC: {best_iso_mcc:.4f}")

    # --- Save Results ---
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/raw_results.csv", index=False)
    
    # Calculate Summary
    summary = df.groupby('Method').agg({
        'Recall': ['mean', 'std'],
        'Specificity': ['mean', 'std'],
        'G-Mean': ['mean', 'std'],
        'MCC': ['mean', 'std']
    })
    
    summary.to_csv(f"{output_dir}/summary_stats.csv")
    print("\nBenchmark Complete!")
    print(df)
    print("\nSummary Stats:")
    print(summary)

if __name__ == "__main__":
    main()


import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import homcloud.interface as hc
from scipy.spatial.distance import pdist, squareform

# Import from existing codebase
from tda_ml.models import AnisotropicOutlierClassifier
from tda_ml.trainer import Trainer
from tda_ml.data_loader import NoisyMNISTDataset, create_data_loader
from tda_ml.config import load_config

def get_animation_config():
    """Load the animation config and apply reproducibility-focused overrides."""
    config = load_config("reproduce")
    
    config['loss']['pos_weight'] = 2.0
    config['outputs']['base_dir'] = 'experiments/repro_pd_animation_final'
    config['outputs']['image_dir'] = 'experiments/repro_pd_animation_final/frames'
    config['outputs']['log_dir'] = 'experiments/repro_pd_animation_final/logs'
    config['training']['visualize_every'] = 100 # Disable Trainer's own viz to save time
    config['training']['epochs'] = 50
    return config

def compute_aniso_dist_matrix(points, params, eps_scale=0.7022):
    """
    Computes the anisotropic distance matrix used in TopologicalLoss.
    Adapted from ``tda_ml.losses`` (anisotropic pairwise distances).
    """
    # points: (N, 2), params: (N, 5)
    centers = points + params[:, 0:2]
    
    a = params[:, 2] + 1e-6
    b = params[:, 3] + 1e-6
    theta = params[:, 4]
    
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    
    inv_a2 = 1.0 / (a**2)
    inv_b2 = 1.0 / (b**2)
    
    m00 = cos_t**2 * inv_a2 + sin_t**2 * inv_b2
    m11 = sin_t**2 * inv_a2 + cos_t**2 * inv_b2
    m01 = cos_t * sin_t * (inv_a2 - inv_b2)
    
    # Pairwise diffs (N, N, 2)
    diff = points.unsqueeze(0) - centers.unsqueeze(1) # (N_center, N_point, 2)
    dx = diff[:, :, 0]
    dy = diff[:, :, 1]
    
    dist_sq_from_i = (m00.unsqueeze(1) * dx**2 + 
                      2 * m01.unsqueeze(1) * dx * dy + 
                      m11.unsqueeze(1) * dy**2)
                      
    dx_rev = -dx.t()
    dy_rev = -dy.t()
    
    dist_sq_from_j = (m00.unsqueeze(0) * dx_rev**2 + 
                      2 * m01.unsqueeze(0) * dx_rev * dy_rev + 
                      m11.unsqueeze(0) * dy_rev**2)
                      
    dist_sq = torch.max(dist_sq_from_i, dist_sq_from_j)
    D = torch.sqrt(torch.clamp(dist_sq, min=1e-8))
    
    return D * eps_scale

def visualize_pd_transition(model, sample, epoch, device, config):
    model.eval()
    data, labels, clean_pc = sample
    data = data.to(device)
    
    with torch.no_grad():
        if data.dim() == 2:
            batch = data.unsqueeze(0)
        else:
            batch = data
        logits, params = model(batch)
    
    points_np = data.squeeze().cpu().numpy()
    labels_np = labels.squeeze().cpu().numpy()
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    params_tensor = params.squeeze(0)
    points_tensor = data if data.dim() == 2 else data.squeeze(0)
    
    # 1. Compute Anisotropic Distance Matrix
    D = compute_aniso_dist_matrix(points_tensor, params_tensor, eps_scale=config['training'].get('topo_eps_scale', 0.7022))
    
    # 2. Compute PD using HomCloud
    D_np = D.cpu().numpy()
    # Explicitly use Ripser-based Rips filtration via HomCloud
    # Positional maxdim=1
    pdlist = hc.PDList.from_rips_filtration(D_np, 1)
    
    # 3. Compute GT PD (Before noise/filtering) using Rips Filtration (Consistent with Training)
    clean_pc_tensor = clean_pc.squeeze(0)
    mask = torch.abs(clean_pc_tensor).sum(dim=1) > 1e-6
    clean_pc_filtered = clean_pc_tensor[mask]
    
    pdlist_gt = None
    if len(clean_pc_filtered) >= 3:
        # Use Euclidean Rips for GT to match training units
        gt_dist_mat = squareform(pdist(clean_pc_filtered.cpu().numpy()))
        pdlist_gt = hc.PDList.from_rips_filtration(gt_dist_mat, 1)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Point Cloud with Predictions & GT Markers
    ax0 = axes[0]
    # Inliers (GT): circles, Outliers (GT): 'x'
    in_mask = labels_np == 0
    out_mask = labels_np == 1
    
    sc = ax0.scatter(points_np[in_mask, 0], points_np[in_mask, 1], c=probs[in_mask], 
                     cmap='coolwarm', vmin=0, vmax=1, label='GT Inlier (Circle)')
    ax0.scatter(points_np[out_mask, 0], points_np[out_mask, 1], c=probs[out_mask], 
                cmap='coolwarm', vmin=0, vmax=1, marker='x', label='GT Outlier (x)')
    
    ax0.set_title(f"Epoch {epoch}: Prediction (Sample Digit 0)")
    ax0.set_aspect('equal')
    ax0.set_xlim(-1.2, 1.2)
    ax0.set_ylim(-1.2, 1.2)
    ax0.legend(loc='upper right', fontsize='small')
    plt.colorbar(sc, ax=ax0)
    
    # Plot 2: Anisotropic PD (Current Learned State)
    ax1 = axes[1]
    if pdlist is not None:
        pd1 = pdlist[1] # Dimension 1 (Homology 1)
        ax1.scatter(pd1.births, pd1.deaths, s=15, c='darkred', alpha=0.6, label='H1')
        # Draw diagonal
        all_vals = np.concatenate([pd1.births, pd1.deaths])
        m_val = np.max(all_vals) if len(all_vals) > 0 else 1.0
        ax1.plot([0, m_val*1.1], [0, m_val*1.1], 'k-', alpha=0.3)
        ax1.set_xlim(0, m_val*1.1)
        ax1.set_ylim(0, m_val*1.1)
    ax1.set_title("Anisotropic PD (H1 - HomCloud)")
    ax1.set_xlabel("Birth")
    ax1.set_ylabel("Death")
    
    # Plot 3: Clean PD (Ground Truth)
    ax2 = axes[2]
    if pdlist_gt is not None:
        pd1_gt = pdlist_gt[1]
        ax2.scatter(pd1_gt.births, pd1_gt.deaths, s=15, c='darkblue', alpha=0.6, label='H1')
        # Draw diagonal
        all_vals_gt = np.concatenate([pd1_gt.births, pd1_gt.deaths])
        m_val_gt = np.max(all_vals_gt) if len(all_vals_gt) > 0 else 1.0
        ax2.plot([0, m_val_gt*1.1], [0, m_val_gt*1.1], 'k-', alpha=0.3)
        ax2.set_xlim(0, m_val_gt*1.1)
        ax2.set_ylim(0, m_val_gt*1.1)
        ax2.set_title("GT PD (H1 - Rips Complex)")
    else:
        ax2.text(0.5, 0.5, "No Clean PD (Too few pts)", ha='center')
        ax2.set_title("GT PD (N/A)")
    ax2.set_xlabel("Birth")
    ax2.set_ylabel("Death")
    
    plt.tight_layout()
    save_path = f"{config['outputs']['image_dir']}/pd_frame_{epoch:03d}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

def main():
    config = get_animation_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    dataset_train = NoisyMNISTDataset(
        root='./data', train=True, 
        max_points=config['data']['max_points'], 
        num_outliers=config['data']['num_outliers'],
        noise_seed=config['data']['seed']
    )
    train_loader = create_data_loader(dataset_train, batch_size=config['data']['batch_size'], shuffle=True)
    
    # Fixed sample for visualization
    dataset_val = NoisyMNISTDataset(
        root='./data', train=False, 
        max_points=300, # Use more points for viz Ground Truth to make it crisp
        num_outliers=config['data']['num_outliers'],
        indices=torch.tensor([3]) # Fixed Digit 0
    )
    # Important: shuffle=False for viz sample consistency!
    viz_loader = create_data_loader(dataset_val, batch_size=1, shuffle=False)
    viz_sample = next(iter(viz_loader))
    
    # Model
    model = AnisotropicOutlierClassifier(
        point_dim=config['model']['input_dim'],
        feature_dim=config['model']['hidden_dim'],
        ellipse_param_dim=config['model']['ellipse_param_dim']
    ).to(device)
    
    # Trainer
    trainer = Trainer(model, config, device=device)
    
    # Reproduction Training
    os.makedirs(config['outputs']['image_dir'], exist_ok=True)
    
    # Initial Frame (Epoch 0)
    visualize_pd_transition(model, viz_sample, 0, device, config)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        trainer.train_epoch(train_loader, epoch)
        visualize_pd_transition(model, viz_sample, epoch, device, config)

if __name__ == "__main__":
    main()

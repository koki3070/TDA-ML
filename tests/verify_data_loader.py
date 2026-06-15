import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tda_ml.data_loader import NoisyMNISTDataset

def verify_data_loader():
    print("Verifying Data Loader Fixes...")
    
    # Initialize dataset
    # Use a small max_points to force subsampling in some cases, and large for padding in others?
    # MNIST digits usually have ~100-200 points.
    # Let's use max_points=300 to force padding for most images.
    dataset = NoisyMNISTDataset(root='./data', train=True, num_samples=10, 
                                max_points=300, num_outliers=0, noise_std=0.1, deterministic=True)
    
    data, labels, clean_pc = dataset[0]
    
    # 1. Verify Clean PC has NO Noise
    # clean_pc should be different from data (inliers part) because data has noise
    # data[:300] are inliers (since num_outliers=0)
    
    # Find valid points in clean_pc (non-zero)
    valid_mask = torch.abs(clean_pc).sum(dim=1) > 1e-6
    clean_points = clean_pc[valid_mask]
    
    # Corresponding points in data (assuming alignment is preserved for the first N points)
    # Since we used zero padding for clean_pc, the valid points should correspond to the first N points of data
    # IF we didn't shuffle. But we DID shuffle at the end of __getitem__!
    # "data = data[perm]"
    
    # Ah, data is shuffled. clean_pc is NOT shuffled (it's created at the end).
    # But clean_pc is created from 'clean_pc_points'.
    # 'data' contains 'inliers' which are 'clean_pc_points' + noise.
    # But 'data' is shuffled. So we can't compare index-by-index easily.
    
    # However, we can check if clean_pc points are exactly integers (before normalization) or 
    # just check if they look "clean" (grid-like) vs "noisy".
    # MNIST points are on a grid (before normalization).
    # After normalization, they should still be on a grid.
    # Noisy points will NOT be on a grid.
    
    # Let's check if clean_pc points are close to the grid.
    # Grid steps: 2.0 / 27.0
    
    # Normalize back to [0, 27]
    clean_denorm = (clean_points + 1.0) / 2.0 * 27.0
    
    # Check if they are close to integers
    diff_from_int = torch.abs(clean_denorm - torch.round(clean_denorm))
    max_diff = diff_from_int.max()
    
    print(f"Max deviation from grid in clean_pc: {max_diff.item()}")
    if max_diff.item() < 1e-5:
        print("PASS: clean_pc is on the grid (clean).")
    else:
        print("FAIL: clean_pc is NOT on the grid (contains noise).")
        
    # 2. Verify Zero Padding
    # We used max_points=300. MNIST digits usually have < 300 points.
    # So clean_pc should have zeros at the end.
    
    num_zeros = (~valid_mask).sum().item()
    print(f"Number of zero-padded points in clean_pc: {num_zeros}")
    
    if num_zeros > 0:
        print("PASS: clean_pc is zero-padded.")
    else:
        print("WARNING: clean_pc is NOT zero-padded (might be a dense image or logic error).")
        
    # Check for duplicates in clean_pc (excluding zeros)
    # unique points
    unique_points = torch.unique(clean_points, dim=0)
    if unique_points.shape[0] == clean_points.shape[0]:
        print("PASS: No duplicates in clean_pc valid points.")
    else:
        print(f"FAIL: Found duplicates in clean_pc! ({clean_points.shape[0]} vs {unique_points.shape[0]} unique)")

if __name__ == "__main__":
    verify_data_loader()

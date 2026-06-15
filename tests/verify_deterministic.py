import torch
from tda_ml.data_loader import NoisyMNISTDataset

def test_determinism():
    print("Testing non-deterministic behavior (default)...")
    dataset = NoisyMNISTDataset(root='./data', train=False, num_samples=100, max_points=100, num_outliers=10)
    
    # Fetch item 0 twice with random operations in between
    torch.manual_seed(42)
    data1, _, _ = dataset[0]
    
    torch.rand(100) # Advance global RNG
    data2, _, _ = dataset[0]
    
    if not torch.allclose(data1, data2):
        print("Confirmed: Default behavior is non-deterministic.")
    else:
        print("Warning: Default behavior appears deterministic (unexpected).")

    print("\nTesting deterministic behavior (after fix)...")
    try:
        dataset_det = NoisyMNISTDataset(root='./data', train=False, num_samples=100, max_points=100, num_outliers=10, deterministic=True)
        
        torch.manual_seed(42)
        data3, _, _ = dataset_det[0]
        
        torch.rand(100) # Advance global RNG
        data4, _, _ = dataset_det[0]
        
        if torch.allclose(data3, data4):
            print("Success: Deterministic mode works as expected.")
        else:
            print("Failure: Deterministic mode is still non-deterministic.")
            
    except TypeError:
        print("Skipping deterministic test: 'deterministic' argument not yet implemented.")

if __name__ == "__main__":
    test_determinism()

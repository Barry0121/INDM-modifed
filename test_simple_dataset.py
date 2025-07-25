"""
Simple test of the ProteinContactMapDataset without torch dependencies.
"""
import numpy as np
import os
import sys

# Add path to import our dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dataset_logic():
    """Test the core dataset loading logic without torch/tensorflow dependencies."""
    print("=== Testing Dataset Loading Logic ===")
    
    # Test file discovery
    pdb_dir = 'pdb'
    all_files = [f for f in os.listdir(pdb_dir) if f.endswith('.npy')]
    print(f"Found {len(all_files)} .npy files")
    
    # Test train/val split logic
    max_samples = 5
    limited_files = all_files[:max_samples]
    print(f"Limited to {len(limited_files)} files")
    
    # Split logic
    train_ratio = 0.8
    seed = 42
    np.random.seed(seed)
    indices = np.arange(len(limited_files))
    np.random.shuffle(indices)
    num_train = int(len(limited_files) * train_ratio)
    train_indices = indices[:num_train]
    eval_indices = indices[num_train:]
    
    train_files = np.array(limited_files)[train_indices]
    eval_files = np.array(limited_files)[eval_indices]
    
    print(f"Train files: {len(train_files)}")
    print(f"Eval files: {len(eval_files)}")
    print(f"Train files: {train_files}")
    print(f"Eval files: {eval_files}")
    
    # Test loading and processing one file
    if len(train_files) > 0:
        test_file = train_files[0]
        file_path = os.path.join(pdb_dir, test_file)
        print(f"\nTesting file: {test_file}")
        
        # Load distance map
        distance_map = np.load(file_path)
        print(f"Original shape: {distance_map.shape}")
        print(f"Original range: [{np.min(distance_map):.3f}, {np.max(distance_map):.3f}]")
        
        # Test padding to 512x512
        max_length = 512
        current_length = distance_map.shape[0]
        
        if current_length <= max_length:
            # Pad with zeros
            output_matrix = np.zeros((max_length, max_length), dtype=np.float32)
            output_matrix[:current_length, :current_length] = distance_map.astype(np.float32)
            print(f"Padded shape: {output_matrix.shape}")
            
            # Test normalization
            minv, maxv = 288.645, 9422.372
            normalized = (output_matrix - minv) / (maxv - minv) * 2 - 1
            print(f"Normalized range: [{np.min(normalized):.3f}, {np.max(normalized):.3f}]")
            
            print("✅ File processing successful!")
        else:
            print(f"❌ File too large: {current_length} > {max_length}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_dataset_logic()
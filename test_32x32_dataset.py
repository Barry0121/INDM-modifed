#!/usr/bin/env python3
"""Test 32x32 dataset processing."""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from datasets import ProteinContactMapDataset

def test_32x32_processing():
    """Test the 32x32 dataset processing directly."""
    print("=== Testing 32x32 Dataset Processing ===")
    
    # Test loading and processing a single file
    sample_file = 'pdb/101m_1.npy'
    print(f"Loading sample file: {sample_file}")
    
    # Load original matrix
    original_matrix = np.load(sample_file)
    print(f"Original matrix shape: {original_matrix.shape}")
    print(f"Original matrix range: [{original_matrix.min():.3f}, {original_matrix.max():.3f}]")
    
    # Test 32x32 cropping
    cropped_matrix = original_matrix[:32, :32]
    print(f"Cropped matrix shape: {cropped_matrix.shape}")
    print(f"Cropped matrix range: [{cropped_matrix.min():.3f}, {cropped_matrix.max():.3f}]")
    
    # Test normalization (simple version)
    minv, maxv = 0.0, 5000.0
    normalized_matrix = (cropped_matrix - minv) / (maxv - minv) * 2 - 1
    print(f"Normalized matrix range: [{normalized_matrix.min():.3f}, {normalized_matrix.max():.3f}]")
    
    # Convert to tensor with channel dimension
    tensor_matrix = torch.from_numpy(normalized_matrix.astype(np.float32))
    tensor_matrix = tensor_matrix.unsqueeze(0)  # Add channel dimension
    print(f"Final tensor shape: {tensor_matrix.shape}")
    print(f"Final tensor dtype: {tensor_matrix.dtype}")
    
    print("\n=== Testing Dataset Creation (Direct) ===")
    
    # Create dataset with more specific parameters
    try:
        dataset = ProteinContactMapDataset(
            main_dir='pdb',
            max_length=32,
            train=True,
            train_ratio=0.8,
            seed=42,
            return_distance=True,
            minv=0.0,
            maxv=5000.0,
            pad_value=0.0,
            max_samples=5  # Small sample for testing
        )
        
        print(f"Dataset created successfully!")
        print(f"Dataset length: {len(dataset)}")
        
        if len(dataset) > 0:
            sample, filename = dataset[0]
            print(f"Sample shape: {sample.shape}")
            print(f"Sample dtype: {sample.dtype}")
            print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
            print(f"Filename: {filename}")
            print("✓ Dataset processing successful!")
        else:
            print("✗ Dataset is empty")
            
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_32x32_processing()
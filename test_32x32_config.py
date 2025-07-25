#!/usr/bin/env python3
"""
Test script to verify 32x32 protein configuration works correctly.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from configs.vp.PROTEIN.indm_fid import get_config

def test_32x32_config():
    """Test the 32x32 protein configuration."""
    print("=== Testing 32x32 Protein Configuration ===")
    
    # Get configuration
    config = get_config()
    print(f"Data image size: {config.data.image_size}")
    print(f"Data channels: {config.data.num_channels}")
    print(f"Flow image size: {config.flow.image_size}")
    print(f"Wolf config path: {config.flow.model_config}")
    
    # Test with dummy data matching our expected input shape
    batch_size = 2
    channels = config.data.num_channels
    height = width = config.data.image_size
    
    print(f"\nTesting with dummy data shape: [{batch_size}, {channels}, {height}, {width}]")
    
    # Create dummy protein distance matrix data
    dummy_data = torch.randn(batch_size, channels, height, width)
    print(f"Created dummy data with shape: {dummy_data.shape}")
    print(f"Dummy data range: [{dummy_data.min():.3f}, {dummy_data.max():.3f}]")
    
    # Test dataset creation
    print("\n=== Testing Dataset Creation ===")
    try:
        from datasets import ProteinContactMapDataset
        
        # Create a small test dataset
        test_dataset = ProteinContactMapDataset(
            main_dir='pdb',
            max_length=32,  # Use 32x32
            train=True,
            max_samples=3,  # Very small for testing
            return_distance=True,
            minv=0.0,
            maxv=5000.0
        )
        
        print(f"Dataset length: {len(test_dataset)}")
        
        if len(test_dataset) > 0:
            sample, filename = test_dataset[0]
            print(f"Sample shape: {sample.shape}")
            print(f"Sample dtype: {sample.dtype}")
            print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
            print(f"Sample filename: {filename}")
            
            # Verify shape is correct
            expected_shape = (1, 32, 32)  # [channels, height, width]
            if sample.shape == expected_shape:
                print("✓ Dataset output shape is correct!")
            else:
                print(f"✗ Dataset output shape mismatch. Expected {expected_shape}, got {sample.shape}")
        else:
            print("✗ Dataset is empty")
            
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
    
    # Test wolf config loading
    print("\n=== Testing Wolf Config Loading ===")
    try:
        import json
        wolf_config_path = config.flow.model_config
        
        if os.path.exists(wolf_config_path):
            with open(wolf_config_path, 'r') as f:
                wolf_config = json.load(f)
            
            print(f"✓ Wolf config loaded successfully from: {wolf_config_path}")
            
            # Check discriminator config
            disc_config = wolf_config['discriminator']
            encoder_config = disc_config['encoder']
            
            print(f"Encoder levels: {encoder_config['levels']}")
            print(f"Encoder in_planes: {encoder_config['in_planes']}")
            print(f"Encoder out_planes: {encoder_config['out_planes']}")
            print(f"Discriminator in_dim: {disc_config['in_dim']}")
            print(f"Discriminator dim: {disc_config['dim']}")
            
            # Verify dimensions
            expected_in_dim = encoder_config['out_planes'] * 4 * 4  # 4x4 after 3 levels from 32x32
            if disc_config['in_dim'] == expected_in_dim:
                print(f"✓ Discriminator in_dim ({disc_config['in_dim']}) matches encoder output ({expected_in_dim})")
            else:
                print(f"✗ Discriminator in_dim ({disc_config['in_dim']}) doesn't match encoder output ({expected_in_dim})")
                
        else:
            print(f"✗ Wolf config file not found: {wolf_config_path}")
            
    except Exception as e:
        print(f"✗ Wolf config loading failed: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_32x32_config()
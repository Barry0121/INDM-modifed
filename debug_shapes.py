#!/usr/bin/env python3
"""Debug script to check protein dataset shapes"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import ProteinContactMapDataset
from torch.utils.data import DataLoader
from configs.vp.PROTEIN.indm_nll import get_config

def main():
    # Get config
    config = get_config()
    
    # Print expected shapes
    print(f"Config expects:")
    print(f"  num_channels: {config.data.num_channels}")
    print(f"  image_size: {config.data.image_size}")
    print(f"  Expected batch shape: [batch_size, {config.data.num_channels}, {config.data.image_size}, {config.data.image_size}]")
    
    # Create a small test dataset
    # Use a simple test directory with dummy data
    test_dir = "/tmp/test_protein_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create some dummy protein data
    import numpy as np
    for i in range(2):
        dummy_map = np.random.rand(256, 256).astype(np.float32)
        np.save(f"{test_dir}/protein_{i:03d}.npy", dummy_map)
    
    try:
        # Create dataset
        dataset = ProteinContactMapDataset(
            main_dir=test_dir,
            max_length=config.data.image_size,
            contact_threshold=config.data.contact_threshold,
            return_distance=True,
            train=True
        )
        
        print(f"\nDataset loaded with {len(dataset)} samples")
        
        # Test single sample
        if len(dataset) > 0:
            sample_tensor, filename = dataset[0]
            print(f"\nSingle sample shape: {sample_tensor.shape}")
            print(f"Sample filename: {filename}")
            
            # Test DataLoader
            dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
            batch = next(iter(dataloader))
            print(f"\nDataLoader batch type: {type(batch)}")
            if isinstance(batch, (tuple, list)):
                print(f"Batch is tuple/list with {len(batch)} elements")
                print(f"Batch[0] (tensors) shape: {batch[0].shape}")
                print(f"Batch[1] (filenames): {batch[1]}")
            else:
                print(f"Batch shape: {batch.shape}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
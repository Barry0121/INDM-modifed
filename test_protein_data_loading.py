#!/usr/bin/env python3
"""Simple test script for PROTEIN data loading.

This script replicates the data loading logic from train() and evaluate()
functions in run_lib.py for testing with PROTEIN dataset configurations.
"""

import sys
import os
sys.path.append('/lambda/nfs/INDM/INDM')

import torch
import datasets
import logging

def test_protein_data_loading(config_path):
    """Test PROTEIN data loading with given configuration."""

    # Import the config
    sys.path.append(os.path.dirname(config_path))
    config_module = os.path.basename(config_path).replace('.py', '')
    config = __import__(config_module).get_config()

    print(f"Testing configuration: {config_path}")
    print(f"Dataset: {config.data.dataset}")
    print(f"Image size: {config.data.image_size}")
    print(f"Num channels: {config.data.num_channels}")
    print(f"SDE type: {config.training.sde}")
    print(f"Sampling method: {config.sampling.method}")
    print(f"Predictor: {config.sampling.predictor}")
    print(f"Corrector: {config.sampling.corrector}")
    print(f"Batch size: {config.training.batch_size}")
    print("-" * 50)

    try:
        # Replicate data loading from train() function
        print("Loading dataset...")
        train_ds, eval_ds = datasets.get_dataset(config)
        train_iter = iter(train_ds)

        # Get data scalers
        scaler = datasets.get_data_scaler(config)
        inverse_scaler = datasets.get_data_inverse_scaler(config)

        print("Dataset loaded successfully!")
        print(f"Train dataset type: {type(train_ds)}")
        print(f"Eval dataset type: {type(eval_ds)}")

        # Test getting a batch (replicate from train() function)
        print("Getting first batch...")
        batch, train_iter = datasets.get_batch(config, train_iter, train_ds)

        print(f"Batch shape: {batch.shape}")
        print(f"Batch dtype: {batch.dtype}")
        print(f"Batch min/max: {batch.min().item():.4f} / {batch.max().item():.4f}")

        # Apply the same preprocessing as in train()
        batch_preprocessed = (255. * batch + torch.rand_like(batch)) / 256.
        batch_scaled = scaler(batch_preprocessed)

        print(f"Preprocessed batch min/max: {batch_preprocessed.min().item():.4f} / {batch_preprocessed.max().item():.4f}")
        print(f"Scaled batch min/max: {batch_scaled.min().item():.4f} / {batch_scaled.max().item():.4f}")

        # Test a few more batches
        print("Testing 3 more batches...")
        for i in range(3):
            batch, train_iter = datasets.get_batch(config, train_iter, train_ds)
            print(f"  Batch {i+2}: shape {batch.shape}, min/max {batch.min().item():.4f}/{batch.max().item():.4f}")

        print("✅ Data loading test PASSED!")
        return True

    except Exception as e:
        print(f"❌ Data loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all three PROTEIN configurations."""

    # Configure basic logging
    logging.basicConfig(level=logging.INFO)

    configs = [
        '/lambda/nfs/INDM/INDM/configs/vp/PROTEIN/indm_fid_ode.py',
        '/lambda/nfs/INDM/INDM/configs/vp/PROTEIN/indm_fid_pc.py',
        '/lambda/nfs/INDM/INDM/configs/ve/PROTEIN/indm_fid_ode.py',
        '/lambda/nfs/INDM/INDM/configs/ve/PROTEIN/indm_fid_pc.py',
        '/lambda/nfs/INDM/INDM/configs/subvp/PROTEIN/indm_fid_ode.py',
        '/lambda/nfs/INDM/INDM/configs/subvp/PROTEIN/indm_fid_pc.py'
    ]

    print("=" * 60)
    print("PROTEIN Data Loading Test")
    print("=" * 60)

    results = {}

    for config_path in configs:
        sde_type = config_path.split('/')[-3]  # vp/ve/subvp
        solver_type = config_path.split('/')[-1].replace('indm_fid_', '').replace('.py', '')  # ode/pc
        config_name = f"{sde_type}/{solver_type}"
        print(f"\n🧪 Testing {config_name}...")

        success = test_protein_data_loading(config_path)
        results[config_name] = success

        print(f"Result: {'✅ PASSED' if success else '❌ FAILED'}")
        print()

    print("=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    for config_name, success in results.items():
        status = '✅ PASSED' if success else '❌ FAILED'
        print(f"{config_name:20} | {status}")

    all_passed = all(results.values())
    print(f"\nOverall result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")


if __name__ == '__main__':
    main()
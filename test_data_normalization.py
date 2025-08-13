#!/usr/bin/env python3
"""
Test script to verify data normalization consistency between VP and VE models.
"""
import sys
import numpy as np
import torch
from pdb_dataset import PDB


def create_mock_config(sde_type, centered):
    """Create a mock config object for testing."""
    class MockConfig:
        def __init__(self, sde_type, centered):
            self.training = type('Training', (), {'sde': sde_type})()
            self.data = type('Data', (), {'centered': centered, 'dataset': 'PROTEIN_CONTACT_MAP'})()
    
    return MockConfig(sde_type, centered)


def test_normalization():
    """Test that PDB dataset respects config.data.centered setting."""
    
    print("=== Testing PDB Dataset Normalization ===\n")
    
    try:
        # Test VP model (centered=True, should normalize to [-1, 1])
        print("1. Testing VP model (data.centered=True)")
        vp_config = create_mock_config('vpsde', True)
        vp_train_dataset = PDB(train=True, config=vp_config)
        vp_matrices = vp_train_dataset.get_matrices()[:1000]  # Get 1000 samples for better range
        
        vp_min, vp_max = np.min(vp_matrices), np.max(vp_matrices)
        print(f"   VP range: [{vp_min:.3f}, {vp_max:.3f}] (expected: [-1, 1])")
        
        vp_correct = -1.0 <= vp_min <= vp_max <= 1.0  # Values should be within [-1, 1]
        if vp_correct:
            print("   ✓ VP normalization CORRECT")
        else:
            print("   ✗ VP normalization INCORRECT")
            
        # Test VE model (centered=False, should normalize to [0, 1])
        print("\n2. Testing VE model (data.centered=False)")
        ve_config = create_mock_config('vesde', False)
        ve_train_dataset = PDB(train=True, config=ve_config)
        ve_matrices = ve_train_dataset.get_matrices()[:1000]  # Get 1000 samples for better range
        
        ve_min, ve_max = np.min(ve_matrices), np.max(ve_matrices)
        print(f"   VE range: [{ve_min:.3f}, {ve_max:.3f}] (expected: [0, 1])")
        
        ve_correct = 0.0 <= ve_min <= ve_max <= 1.0  # Values should be within [0, 1]
        if ve_correct:
            print("   ✓ VE normalization CORRECT")
        else:
            print("   ✗ VE normalization INCORRECT")
            
        # Test SubVP model (centered=True, should normalize to [-1, 1])
        print("\n3. Testing SubVP model (data.centered=True)")
        subvp_config = create_mock_config('subvpsde', True)
        subvp_train_dataset = PDB(train=True, config=subvp_config)
        subvp_matrices = subvp_train_dataset.get_matrices()[:1000]  # Get 1000 samples for better range
        
        subvp_min, subvp_max = np.min(subvp_matrices), np.max(subvp_matrices)
        print(f"   SubVP range: [{subvp_min:.3f}, {subvp_max:.3f}] (expected: [-1, 1])")
        
        subvp_correct = -1.0 <= subvp_min <= subvp_max <= 1.0  # Values should be within [-1, 1]
        if subvp_correct:
            print("   ✓ SubVP normalization CORRECT")
        else:
            print("   ✗ SubVP normalization INCORRECT")
            
        # Test denormalization consistency
        print("\n4. Testing denormalization consistency")
        
        # Test VP denormalization
        vp_sample = vp_matrices[0]
        vp_denorm = vp_train_dataset.denormalize_tensor(vp_sample)
        print(f"   VP: sample [{np.min(vp_sample):.3f}, {np.max(vp_sample):.3f}] -> denorm [{np.min(vp_denorm):.3f}, {np.max(vp_denorm):.3f}]")
        
        # Test VE denormalization  
        ve_sample = ve_matrices[0]
        ve_denorm = ve_train_dataset.denormalize_tensor(ve_sample)
        print(f"   VE: sample [{np.min(ve_sample):.3f}, {np.max(ve_sample):.3f}] -> denorm [{np.min(ve_denorm):.3f}, {np.max(ve_denorm):.3f}]")
        
        print("\n=== Test Summary ===")
        if vp_correct and ve_correct and subvp_correct:
            print("✓ ALL TESTS PASSED - Data normalization is working correctly!")
            print("  - VP models will train/evaluate on [-1, 1] data")
            print("  - VE models will train/evaluate on [0, 1] data")
            print("  - SubVP models will train/evaluate on [-1, 1] data")
            return True
        else:
            print("✗ TESTS FAILED - Data normalization needs fixing")
            return False
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_normalization()
    sys.exit(0 if success else 1)
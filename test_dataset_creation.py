#!/usr/bin/env python3
"""
Test script for the modified dataset creation that truncates instead of skips.
"""
import os
import sys
import numpy as np
from create_dataset import process_distance_matrix, scan_and_process_files

def test_truncation_behavior():
    """Test the modified dataset creation behavior."""
    print("=== Testing Dataset Creation with Truncation ===")
    
    # Test individual matrix processing
    print("\n1. Testing individual matrix processing:")
    
    # Test with different sized sample files
    test_files = ['pdb/101m_1.npy', 'pdb/102l_1.npy', 'pdb/103l_1.npy']
    target_size = 32  # Test with 32x32 for our protein use case
    
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"\nTesting {test_file}:")
            
            # Check original size
            original = np.load(test_file)
            print(f"  Original shape: {original.shape}")
            
            # Process with our function
            processed = process_distance_matrix(test_file, target_size)
            
            if processed is not None:
                print(f"  Processed shape: {processed.shape}")
                print(f"  Non-zero elements: {np.count_nonzero(processed)}")
                print(f"  Value range: [{processed.min():.3f}, {processed.max():.3f}]")
                
                # Verify it's the expected size
                if processed.shape == (target_size, target_size):
                    print("  ✓ Output size is correct")
                else:
                    print("  ✗ Output size is incorrect")
                    
                # Check if it was truncated (original was larger)
                if original.shape[0] > target_size:
                    print("  ✓ Large matrix was truncated")
                    # Verify truncation correctness
                    expected_portion = original[:target_size, :target_size]
                    actual_portion = processed[:original.shape[0], :original.shape[1]]
                    if np.allclose(expected_portion.astype(np.float32), actual_portion[:expected_portion.shape[0], :expected_portion.shape[1]]):
                        print("  ✓ Truncation is correct")
                    else:
                        print("  ✗ Truncation has errors")
                elif original.shape[0] < target_size:
                    print("  ✓ Small matrix was padded")
                else:
                    print("  ✓ Matrix was used as-is")
            else:
                print("  ✗ Processing failed")
        else:
            print(f"  ⚠ File {test_file} not found")
    
    print(f"\n2. Testing batch processing:")
    
    # Test batch processing with small sample
    try:
        all_matrices, global_min, global_max = scan_and_process_files(
            pdb_dir='pdb',
            batch_size=100,
            target_size=target_size,
            max_samples=5  # Small sample for testing
        )
        
        print(f"✓ Batch processing completed successfully")
        print(f"  Total matrices processed: {len(all_matrices)}")
        print(f"  Global min/max: {global_min:.3f}/{global_max:.3f}")
        
        if len(all_matrices) > 0:
            print(f"  Matrix shapes: {[m.shape for m in all_matrices[:3]]}")  # Show first 3
            print("  ✓ All matrices have consistent shape")
        else:
            print("  ✗ No matrices were processed")
            
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_truncation_behavior()
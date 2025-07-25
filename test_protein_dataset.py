"""
Test the modified ProteinContactMapDataset with pre-loading.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import ProteinContactMapDataset
import time

def test_protein_dataset():
    print("=== Testing Modified ProteinContactMapDataset ===")
    
    # Test with small sample
    start_time = time.time()
    
    print("\nCreating training dataset...")
    train_dataset = ProteinContactMapDataset(
        main_dir='pdb',
        max_length=512,
        train=True,
        max_samples=5,  # Small test
        return_distance=True,
        minv=288.645,  # Use values from our test
        maxv=9422.372
    )
    
    init_time = time.time() - start_time
    print(f"Initialization took: {init_time:.2f} seconds")
    
    print(f"\nDataset length: {len(train_dataset)}")
    
    if len(train_dataset) > 0:
        print("\nTesting data access...")
        access_start = time.time()
        
        # Test multiple access to same item (should be fast)
        for i in range(3):
            tensor_map, filename = train_dataset[0]
            print(f"Access {i+1}: Shape: {tensor_map.shape}, Filename: {filename}")
        
        access_time = time.time() - access_start
        print(f"3 accesses took: {access_time:.4f} seconds")
        
        # Test tensor properties
        sample_tensor, sample_filename = train_dataset[0]
        print(f"\nTensor properties:")
        print(f"  Shape: {sample_tensor.shape}")
        print(f"  Type: {sample_tensor.dtype}")
        print(f"  Range: [{sample_tensor.min():.3f}, {sample_tensor.max():.3f}]")
        print(f"  Filename: {sample_filename}")
        
        # Test evaluation dataset
        print("\nCreating evaluation dataset...")
        eval_dataset = ProteinContactMapDataset(
            main_dir='pdb',
            max_length=512,
            train=False,
            max_samples=5,
            return_distance=True,
            minv=288.645,
            maxv=9422.372
        )
        
        print(f"Eval dataset length: {len(eval_dataset)}")
        
        if len(eval_dataset) > 0:
            eval_tensor, eval_filename = eval_dataset[0]
            print(f"Eval sample shape: {eval_tensor.shape}")
            print(f"Eval filename: {eval_filename}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_protein_dataset()
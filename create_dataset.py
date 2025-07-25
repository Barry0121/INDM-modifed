"""
Data consolidation script for PDB distance matrices.
Converts individual .npy files to consolidated 512x512 dataset with train/val split.
"""
import os
import numpy as np
from tqdm import tqdm
import argparse
import warnings

def process_distance_matrix(filepath, target_size=512):
    """
    Load and process a single distance matrix file.
    
    Args:
        filepath: Path to .npy file
        target_size: Target matrix size (512x512)
    
    Returns:
        Processed matrix of shape (512, 512) or None if too large
    """
    try:
        matrix = np.load(filepath)
        
        # Skip if matrix is larger than target size
        if matrix.shape[0] > target_size or matrix.shape[1] > target_size:
            return None
            
        # Create padded matrix filled with zeros
        padded_matrix = np.zeros((target_size, target_size), dtype=np.float32)
        
        # Copy original matrix to top-left corner
        rows, cols = matrix.shape
        padded_matrix[:rows, :cols] = matrix.astype(np.float32)
        
        return padded_matrix
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def scan_and_process_files(pdb_dir, batch_size=1000, target_size=512, max_samples=None):
    """
    Scan all .npy files and process them in batches.
    
    Args:
        pdb_dir: Directory containing .npy files
        batch_size: Number of files to process in each batch
        target_size: Target matrix size
        max_samples: Maximum number of samples to process
    
    Returns:
        all_matrices: List of processed matrices
        global_min: Global minimum value
        global_max: Global maximum value
    """
    # Get all .npy files
    npy_files = [f for f in os.listdir(pdb_dir) if f.endswith('.npy')]
    print(f"Found {len(npy_files)} .npy files")
    
    # Limit files if max_samples specified
    if max_samples:
        npy_files = npy_files[:max_samples]
        print(f"Limited to {len(npy_files)} files for processing")
    
    all_matrices = []
    global_min = float('inf')
    global_max = float('-inf')
    processed_count = 0
    skipped_count = 0
    
    # Process files in batches
    for i in tqdm(range(0, len(npy_files), batch_size), desc="Processing batches"):
        batch_files = npy_files[i:i+batch_size]
        batch_matrices = []
        
        for filename in batch_files:
            filepath = os.path.join(pdb_dir, filename)
            matrix = process_distance_matrix(filepath, target_size)
            
            if matrix is not None:
                batch_matrices.append(matrix)
                
                # Update global min/max (only for non-zero values to ignore padding)
                non_zero_mask = matrix > 0
                if np.any(non_zero_mask):
                    batch_min = np.min(matrix[non_zero_mask])
                    batch_max = np.max(matrix[non_zero_mask])
                    global_min = min(global_min, batch_min)
                    global_max = max(global_max, batch_max)
                
                processed_count += 1
            else:
                skipped_count += 1
        
        # Add batch to all matrices
        all_matrices.extend(batch_matrices)
        
        # Print progress
        if i % (batch_size * 10) == 0:
            print(f"Processed: {processed_count}, Skipped: {skipped_count}, "
                  f"Current global min/max: {global_min:.3f}/{global_max:.3f}")
    
    print(f"\nFinal stats:")
    print(f"Total processed: {processed_count}")
    print(f"Total skipped (too large): {skipped_count}")
    print(f"Global min/max: {global_min:.3f}/{global_max:.3f}")
    
    return all_matrices, global_min, global_max

def save_consolidated_dataset(matrices, global_min, global_max, output_path):
    """
    Save consolidated dataset to .npz format.
    
    Args:
        matrices: List of processed matrices
        global_min: Global minimum value
        global_max: Global maximum value
        output_path: Path to save .npz file
    """
    print(f"Converting {len(matrices)} matrices to numpy array...")
    
    # Convert to numpy array
    distance_matrices = np.array(matrices, dtype=np.float32)
    
    print(f"Final dataset shape: {distance_matrices.shape}")
    print(f"Final dataset size: {distance_matrices.nbytes / (1024**3):.2f} GB")
    
    # Save to .npz format
    print(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        distance_matrices=distance_matrices,
        global_min=global_min,
        global_max=global_max,
        num_samples=len(matrices)
    )
    
    print(f"Dataset saved successfully!")

def main():
    parser = argparse.ArgumentParser(description="Create consolidated PDB dataset")
    parser.add_argument("--pdb_dir", default="pdb", help="Directory containing .npy files")
    parser.add_argument("--output", default="data/pdb/distance_matrices.npz", help="Output .npz file path")
    parser.add_argument("--target_size", type=int, default=512, help="Target matrix size")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to process (for testing)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("=== PDB Dataset Consolidation ===")
    print(f"Input directory: {args.pdb_dir}")
    print(f"Output file: {args.output}")
    print(f"Target size: {args.target_size}x{args.target_size}")
    print(f"Batch size: {args.batch_size}")
    
    # Process files
    all_matrices, global_min, global_max = scan_and_process_files(
        args.pdb_dir, args.batch_size, args.target_size, args.max_samples
    )
    
    # Note: max_samples is already handled in scan_and_process_files
    
    # Save consolidated dataset
    save_consolidated_dataset(all_matrices, global_min, global_max, args.output)

if __name__ == "__main__":
    main()
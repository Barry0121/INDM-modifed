#!/usr/bin/env python3
import numpy as np
import os

def check_protein_files():
    """Check sizes of protein files to see which ones are >= 32 residues."""
    pdb_dir = 'pdb'
    files = os.listdir(pdb_dir)[:10]  # Check first 10 files
    
    print("Checking protein file sizes:")
    valid_count = 0
    
    for f in files:
        try:
            matrix = np.load(os.path.join(pdb_dir, f))
            is_valid = matrix.shape[0] >= 32
            valid_count += is_valid
            status = "✓ >= 32" if is_valid else "✗ < 32"
            print(f'{f}: {matrix.shape} - {status}')
        except Exception as e:
            print(f'{f}: Error - {e}')
    
    print(f'\nValid files (>= 32 residues): {valid_count}/{len(files)}')

if __name__ == "__main__":
    check_protein_files()
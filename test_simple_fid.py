#!/usr/bin/env python3
"""Simple test for protein FID functions without full imports."""

import numpy as np
from scipy import linalg
import tempfile
import os

def protein_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute the Frechet Distance between two multivariate Gaussians for protein data."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, 'Mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Covariance matrices have different dimensions'
    
    diff = mu1 - mu2
    
    # Add regularization to avoid numerical issues with large covariance matrices
    reg_eps = max(eps, 1e-4)  # Minimum regularization for high-dim matrices
    sigma1_reg = sigma1 + np.eye(sigma1.shape[0]) * reg_eps
    sigma2_reg = sigma2 + np.eye(sigma2.shape[0]) * reg_eps
    
    # Use eigenvalue decomposition for more stable computation
    try:
        # Try standard approach first
        product = sigma1_reg.dot(sigma2_reg)
        covmean, _ = linalg.sqrtm(product, disp=False)
        
        # Check for numerical issues
        if not np.isfinite(covmean).all() or np.iscomplexobj(covmean):
            raise np.linalg.LinAlgError("Matrix square root failed")
            
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: use eigenvalue decomposition for numerical stability
        print("Warning: Using eigenvalue decomposition fallback for matrix square root")
        
        # Compute eigendecomposition of both matrices
        eigvals1, eigvecs1 = np.linalg.eigh(sigma1_reg)
        eigvals2, eigvecs2 = np.linalg.eigh(sigma2_reg)
        
        # Ensure positive eigenvalues
        eigvals1 = np.maximum(eigvals1, reg_eps)
        eigvals2 = np.maximum(eigvals2, reg_eps)
        
        # Reconstruct matrices with regularized eigenvalues
        sigma1_stable = eigvecs1 @ np.diag(eigvals1) @ eigvecs1.T
        sigma2_stable = eigvecs2 @ np.diag(eigvals2) @ eigvecs2.T
        
        # Compute trace of geometric mean using eigenvalue approach
        product_stable = sigma1_stable @ sigma2_stable
        prod_eigvals = np.linalg.eigvals(product_stable)
        prod_eigvals = np.maximum(np.real(prod_eigvals), 0)  # Ensure non-negative
        tr_covmean = np.sum(np.sqrt(prod_eigvals))
    else:
        # Standard path - remove any tiny imaginary components
        if np.iscomplexobj(covmean):
            if np.allclose(covmean.imag, 0, atol=1e-3):
                covmean = covmean.real
            else:
                # Large imaginary components - use fallback
                print(f"Warning: Large imaginary component detected: {np.max(np.abs(covmean.imag))}, using fallback")
                # Use eigenvalue approach as fallback
                eigvals1, eigvecs1 = np.linalg.eigh(sigma1_reg)
                eigvals2, eigvecs2 = np.linalg.eigh(sigma2_reg)
                eigvals1 = np.maximum(eigvals1, reg_eps)
                eigvals2 = np.maximum(eigvals2, reg_eps)
                sigma1_stable = eigvecs1 @ np.diag(eigvals1) @ eigvecs1.T
                sigma2_stable = eigvecs2 @ np.diag(eigvals2) @ eigvecs2.T
                product_stable = sigma1_stable @ sigma2_stable
                prod_eigvals = np.linalg.eigvals(product_stable)
                prod_eigvals = np.maximum(np.real(prod_eigvals), 0)
                tr_covmean = np.sum(np.sqrt(prod_eigvals))
        else:
            tr_covmean = np.trace(covmean)
    
    # Compute final FID
    mean_diff = diff.dot(diff)
    trace1 = np.trace(sigma1_reg)
    trace2 = np.trace(sigma2_reg)
    
    fid = mean_diff + trace1 + trace2 - 2 * tr_covmean
    
    # Ensure non-negative result (FID should always be >= 0)
    fid = max(0.0, fid)
    
    return fid

def test_protein_frechet_distance():
    """Test the frechet distance calculation."""
    print("Testing protein_frechet_distance function...")
    
    # Create two simple 2D distributions
    mu1 = np.array([0.5, 0.3])
    sigma1 = np.array([[0.1, 0.02], [0.02, 0.1]])
    
    mu2 = np.array([0.4, 0.4])  
    sigma2 = np.array([[0.15, 0.01], [0.01, 0.12]])
    
    fid = protein_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"FID between two different distributions: {fid:.6f}")
    
    # Test identical distributions (should be 0)
    fid_identical = protein_frechet_distance(mu1, sigma1, mu1, sigma1)
    print(f"FID between identical distributions: {fid_identical:.10f}")
    
    assert abs(fid_identical) < 1e-10, f"Identical distributions should have FID ≈ 0, got {fid_identical}"
    
    print("✓ Frechet distance test passed")

def test_protein_data_shapes():
    """Test with realistic protein data shapes (32x32x1 flattened)."""
    print("Testing with protein distance map dimensions...")
    
    # Simulate 100 protein distance maps of size 32x32x1, flattened to 1024 dimensions
    n_samples = 100
    dim = 32 * 32 * 1
    
    # Generate fake protein data in [0,1] range
    np.random.seed(42)  # For reproducible results
    
    # Reference data (slightly different distribution)
    ref_data = np.random.beta(2, 3, (n_samples, dim))  # Beta distribution biased toward lower values
    ref_mu = np.mean(ref_data, axis=0)
    ref_sigma = np.cov(ref_data, rowvar=False)
    
    # Generated data (different distribution)
    gen_data = np.random.beta(3, 2, (n_samples, dim))  # Beta distribution biased toward higher values  
    gen_mu = np.mean(gen_data, axis=0)
    gen_sigma = np.cov(gen_data, rowvar=False)
    
    print(f"Reference data range: [{ref_data.min():.3f}, {ref_data.max():.3f}]")
    print(f"Generated data range: [{gen_data.min():.3f}, {gen_data.max():.3f}]")
    print(f"Data shape: {ref_data.shape}")
    print(f"Covariance matrix shape: {ref_sigma.shape}")
    
    # Compute FID
    fid_score = protein_frechet_distance(ref_mu, ref_sigma, gen_mu, gen_sigma)
    print(f"FID between reference and generated protein data: {fid_score:.6f}")
    
    # Test with identical data (should be near 0)
    fid_identical = protein_frechet_distance(ref_mu, ref_sigma, ref_mu, ref_sigma)
    print(f"FID between identical reference data: {fid_identical:.10f}")
    
    assert abs(fid_identical) < 1e-5, f"Identical data should have FID ≈ 0, got {fid_identical}"
    assert fid_score > 0, f"Different distributions should have FID > 0, got {fid_score}"
    
    print("✓ Protein data shape test passed")

def test_sample_file_creation():
    """Test creating and loading sample files like the actual pipeline does."""
    print("Testing sample file creation and loading...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_dir = temp_dir
        
        # Create dummy protein samples (simulating sampling_lib output)
        for i in range(2):  # Create 2 sample files
            samples_per_file = 50
            samples = np.random.rand(samples_per_file, 32, 32, 1).astype(np.float32)
            
            # Save as uint8 (simulating actual pipeline behavior)
            samples_uint8 = (samples * 255).astype(np.uint8)
            filename = f"sample_{i}.npz"
            np.savez_compressed(os.path.join(sample_dir, filename), samples=samples_uint8)
            
        print(f"Created sample files in {sample_dir}")
        
        # Load samples back (simulating load_protein_samples function)
        loaded_samples = []
        for i in range(2):
            filename = f"sample_{i}.npz"
            filepath = os.path.join(sample_dir, filename)
            
            data = np.load(filepath)['samples']
            
            # Convert from uint8 back to [0,1] float range
            if data.dtype == np.uint8:
                data = data.astype(np.float32) / 255.0
            
            # Flatten spatial dimensions
            data_flat = data.reshape(data.shape[0], -1)
            loaded_samples.append(data_flat)
            
        all_samples = np.concatenate(loaded_samples, axis=0)
        print(f"Loaded samples shape: {all_samples.shape}")
        print(f"Sample range: [{all_samples.min():.3f}, {all_samples.max():.3f}]")
        
        assert all_samples.shape == (100, 32*32*1), f"Expected shape (100, 1024), got {all_samples.shape}"
        assert 0 <= all_samples.min() and all_samples.max() <= 1, "Samples should be in [0,1] range"
        
        print("✓ Sample file creation test passed")

if __name__ == "__main__":
    print("Simple Protein FID Function Tests")
    print("=" * 40)
    
    try:
        test_protein_frechet_distance()
        print()
        
        test_protein_data_shapes()
        print()
        
        test_sample_file_creation()
        print()
        
        print("✅ All simple tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
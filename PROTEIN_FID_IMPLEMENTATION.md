# Protein FID Implementation

## Overview

This implementation provides a domain-appropriate FID (Fréchet Inception Distance) evaluation for protein distance maps that operates directly on [0,1] normalized data without using InceptionV3 features.

## Key Features

- **Direct comparison**: Computes FID directly on [0,1] normalized protein distance maps
- **PyTorch-based**: Pure PyTorch implementation without cleanfid dependency for protein data
- **Backward compatible**: Maintains compatibility with existing `compute_fid_and_is()` function
- **Statistics caching**: Automatically caches reference dataset statistics to avoid recomputation
- **Proper normalization**: Handles data normalization consistently with the training pipeline

## Implementation Details

### Core Functions

1. **`protein_frechet_distance(mu1, sigma1, mu2, sigma2)`**
   - Robust Fréchet distance calculation with numerical stability for high-dimensional matrices
   - Uses regularization (minimum 1e-4) for covariance matrices
   - Falls back to eigenvalue decomposition when matrix square root fails
   - Handles complex eigenvalues and ensures non-negative FID scores
   - Returns scalar FID score

2. **`get_protein_dataset_stats(config, num_data)`**
   - Loads reference statistics from cached file or computes from dataset
   - Uses same data normalization as training pipeline ([0,1] range)
   - Caches results in `assets/stats/protein_stats_{image_size}_{num_data}.npz`

3. **`load_protein_samples(sample_dir, num_data)`**
   - Loads generated samples from sample directory
   - Handles uint8 to float32 conversion properly
   - Ensures samples are in [0,1] range

4. **`compute_protein_fid(config, assetdir, ckpt, name, sample_dir, num_data)`**
   - Main function that computes protein FID
   - Routes automatically when `config.data.dataset == 'PROTEIN'`
   - Saves results in compatible format

### Integration

The implementation integrates seamlessly with the existing evaluation pipeline:

- `compute_fid_and_is()` automatically routes PROTEIN datasets to the new implementation
- IS (Inception Score) returns None for protein data (not applicable)
- Results are saved in the same format as existing FID evaluation

### Data Flow

1. **Reference data**: Loaded from dataset → normalized to [0,1] → flattened → statistics computed
2. **Generated samples**: Loaded from sample files → converted to [0,1] → flattened → statistics computed  
3. **FID computation**: Direct Fréchet distance between the two distributions

## Usage

The implementation is transparent to users - simply run FID evaluation on PROTEIN datasets as usual:

```python
from evaluation import compute_fid_and_is

# This will automatically use protein-specific FID for PROTEIN datasets
compute_fid_and_is(config, score_model, flow_model, sampling_fn, 
                   step, sample_dir, assetdir, num_data)
```

## Validation

The implementation has been tested with:

- ✅ Basic Fréchet distance calculation
- ✅ Realistic protein data dimensions (32×32×1 = 1024D)
- ✅ Sample file loading and conversion
- ✅ Numerical stability with large covariance matrices
- ✅ Data range preservation ([0,1])

## File Changes

### Modified Files:
- `evaluation.py`: Added protein-specific FID functions and routing
- `compute_fid_and_is_()`: Added PROTEIN dataset case

### Created Files:
- `test_simple_fid.py`: Standalone validation tests
- `PROTEIN_FID_IMPLEMENTATION.md`: This documentation

### Preserved:
- All existing functionality for other datasets (CIFAR10, CELEBA, etc.)
- Function signatures and return formats
- Statistics caching patterns

## Technical Notes

- **Data range**: Protein samples are kept in [0,1] range (meaningful distance values)
- **Memory efficiency**: Samples are flattened only when needed for covariance computation
- **Numerical stability**: Uses regularized covariance matrices (minimum 1e-4 regularization)
- **Robust computation**: Falls back to eigenvalue decomposition for matrix square root failures  
- **Tolerance**: Handles numerical precision issues and ensures non-negative FID scores

This implementation provides a scientifically meaningful evaluation metric for protein distance maps that measures structural similarity rather than perceptual similarity.
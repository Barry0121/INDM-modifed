# INDM Project Memory

## Training Configuration Strategy

### Training Steps Reduction (2025-08-13)
- **Issue**: Training runs for 70000 steps due to early stopping, which is too long for testing phases
- **Observation**: Successful runs typically show lower FID within 20000 steps (~2 sampling occasions)
- **Solution**: Reduced training steps to 20005 for faster testing and performance approximation
- **Rationale**: This allows getting good performance estimates without full training pipeline overhead
- **IMPORTANT TODO**: After debugging phase is complete, restore `training.n_iters` back to 1,000,000 in all PROTEIN config files for full training runs

### Configuration Files to Modify
- All config files in configs/ directory need training.n_iters updated from 70000 to 20005
- This affects both flow model and non-flow model configurations

## Project Structure
- Main training code: run_lib.py
- Evaluation code: evaluation.py (recently fixed indentation and syntax errors)
- Dataset handling: pdb_dataset.py, datasets.py
- Configuration files: configs/ directory with subdirectories for different model types

## Recent Fixes
- Fixed evaluation.py syntax errors including indentation issues and undefined variable 'mode'

## Troubleshooting Changes (2025-08-13)

### 1. Data Normalization Updates
- **VP and SubVP**: Modified to use data normalized to [-1,1] and output [-1,1] data
- **VE**: Modified to normalize input to [0,1] and output [0,1] data
- **Impact**: Different SDE types now have proper normalization schemes

### 2. Evaluation Metrics Issues - RESOLVED ✅
- **Problem**: Density, Coverage, and IS (Inception Score) metrics returning problematic values
- **Symptoms**: 
  - Results of 0.0 (indicating computation failure)
  - Results close to 1.0 (potentially indicating trivial/degenerate cases)
- **Root Cause**: Inconsistent normalization between real and fake data for PRDC computation
- **Solution Implemented**: SDE-specific normalization in `compute_protein_diversity_coverage()` function:
  - **VP/SubVP SDEs**: Use mean/std normalization (preserves working approach)
  - **VE SDE**: Use min-max normalization to [0,1] range (matches VE's [0,1] data range)
- **Implementation**: Modified `evaluation.py:654-728` to accept config parameter and apply conditional normalization based on `config.training.sde`
- **Status**: Fixed for VP and SubVP, VE normalization ready for testing

### 3. Data Shape Issues
- **Problem**: Made changes to data shapes that might be breaking flow + diffusion model integration
- **Impact**: Flow model + diffusion model combination may not work properly
- **Current Goal**: Focus on evaluating diffusion model WITHOUT flow using ODE and PC methods on protein contact map dataset
- **Priority**: Get basic diffusion model working first, then troubleshoot flow integration later

### 4. Flow Model Performance Issues (2025-08-14)
- **Problem**: Flow model continues to have issues with higher FID compared to diffusion-only models
- **Status**: Ongoing issue requiring investigation
- **Impact**: Flow + diffusion model combination underperforming relative to pure diffusion approaches
- **Next Steps**: Focus on diffusion-only models until flow integration issues are resolved

## Success Criteria

### FID Performance Standard
- **Target**: FID should reach single digits for successful training
- **Scope**: This standard applies across all SDE types:
  - **VP (Variance Preserving)**
  - **SubVP (Sub-Variance Preserving)** 
  - **VE (Variance Exploding)**
- **Note**: VP, SubVP, and VE differ only in convergence speed, not final performance
- **Expectation**: All three methods should achieve single-digit FID when properly configured
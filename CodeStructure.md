# INDM Codebase Structure

## Overview
This repository implements the **Implicit Nonlinear Diffusion Model (INDM)** from the NeurIPS 2022 paper "Maximum Likelihood Training of Implicit Nonlinear Diffusion Model". The codebase combines normalizing flows with diffusion processes to learn nonlinear diffusion models.

## Project Structure

### Core Components

#### Entry Points
- **`main.py`** - Main entry point with GPU configuration and command-line interface
  - Supports `train` and `eval` modes
  - Handles TensorFlow/PyTorch GPU memory management
- **`run_lib.py`** - High-level training and evaluation orchestration

#### Model Architecture (`src/models/`)
- **Diffusion Models**:
  - `ddpm.py` - Denoising Diffusion Probabilistic Models
  - `ncsnv2.py` - Noise Conditional Score Networks v2 (multiple variants)
  - `ncsnpp.py` - NCSN++ architecture
  - `vdm.py` - Variational Diffusion Models
- **Building Blocks**:
  - `layers.py` - Basic neural network layers and blocks
  - `layerspp.py` - Advanced layers with Fourier projections
  - `normalization.py` - Various normalization layers (conditional/unconditional)
  - `up_or_down_sampling.py` - Upsampling/downsampling operations
- **Utilities**:
  - `ema.py` - Exponential Moving Average for model parameters
  - `utils.py` - Model creation and management utilities

#### Flow Models (`src/flow_models/`)
- **`flow_model.py`** - Main flow model interface
- **ResFlow** (`resflow/`) - Residual flow implementation
  - Invertible ResNet blocks
  - Custom layers for normalizing flows
- **WOLF** (`wolf/`) - Another flow architecture
  - Glow and MaCow implementations
  - Comprehensive module system for encoders/decoders/discriminators

#### Training & Sampling
- **`losses.py`** - Loss functions for diffusion models
  - SDE-based losses
  - SMLD/DDPM specific losses
- **`sampling.py`** - Sampling algorithms
  - Various predictors (Euler-Maruyama, Reverse Diffusion, Ancestral)
  - Correctors (Langevin dynamics)
  - PC sampler and ODE sampler implementations
- **`sampling_lib.py`** - High-level sampling utilities

#### SDE Framework
- **`sde_lib.py`** - Stochastic Differential Equation abstractions
  - VPSDE (Variance Preserving)
  - VESDE (Variance Exploding)
  - GeometricVPSDE
  - Forward/reverse SDE implementations

#### Evaluation
- **`evaluation.py`** - Metrics computation
  - FID (Fréchet Inception Distance)
  - IS (Inception Score)
  - NLL/NELBO evaluation
- **`modern_metrics.py`** - Modern metric implementations
  - KID (Kernel Inception Distance)
  - Improved FID calculations
- **`likelihood.py`** - Likelihood computation for diffusion models

#### Data & Configuration
- **`datasets.py`** - Dataset loading and preprocessing
  - CIFAR-10, CelebA, ImageNet support
  - Data transformations and scalers
- **`configs/`** - Configuration files
  - Default configs for CIFAR-10 and CelebA
  - Separate configs for VE/VP SDEs
  - FID vs NLL optimization variants

### Additional Components

#### Utilities
- **`utils.py`** - General utilities for checkpointing and model loading
- **`cleanfid/`** - Clean-FID implementation for consistent evaluation

#### Pretrained Models
- **`checkpoints_cifar10/`** - CIFAR-10 checkpoints
- **`checkpoints_celebA/`** - CelebA checkpoints
- Organized by SDE type (VE/VP) and optimization target (FID/NLL)

#### Custom Operations
- **`op/`** - CUDA kernels for optimized operations
  - Fused bias activation
  - Upfirdn2d operations

## Key Features

1. **Dual Framework**: Combines diffusion models with normalizing flows
2. **Multiple SDEs**: Supports VE, VP, and Geometric VP SDEs
3. **Flexible Architecture**: Multiple model architectures (DDPM, NCSN, VDM)
4. **Comprehensive Evaluation**: FID, IS, NLL/NELBO metrics
5. **Advanced Sampling**: PC sampling, ODE sampling with various predictors/correctors

## Configuration System
- ML Collections-based configuration
- Hierarchical config structure
- Separate configs for different datasets and SDE types
- Easy hyperparameter tuning

## Training Pipeline
1. Data loading with configurable transformations
2. Model initialization (score model + flow model)
3. Loss computation based on SDE type
4. EMA updates for stable training
5. Periodic evaluation and checkpointing

## Evaluation Pipeline
1. Load pretrained checkpoints
2. Generate samples using configured sampler
3. Compute metrics (FID, IS, NLL)
4. Save results and generated samples

# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training NCSN++ on Protein Contact Maps with VE SDE."""

import torch
from configs.default_celeba_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'vesde'
  training.continuous = True
  training.reduce_mean = True
  training.likelihood_weighting = True
  training.importance_sampling = True
  training.batch_size = 64  # Very small due to 32x32 matrices
  training.n_iters = 2005  # Short run for PC flow testing
  training.snapshot_freq = 1000
  training.log_freq = 100
  training.eval_freq = 1000
  training.snapshot_freq_for_preemption = 1000

  # sampling
  sampling = config.sampling
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.16

  # data - Modified for protein distance maps
  data = config.data
  data.dataset = 'PROTEIN_CONTACT_MAP'  # Custom dataset name
  data.image_size = 32  # Our actual matrix size (32x32 - first 32 residues)
  data.num_channels = 1  # Distance maps are single channel
  data.centered = True  # Center the data around 0 (we normalize to [-1,1])
  data.uniform_dequantization = False  # Not needed for continuous distance maps
  data.num_epochs = None  # Infinite epochs
  data.cache = False  # Data is pre-loaded in memory
  data.random_flip = False  # Proteins have specific orientation

  # Protein dataset parameters
  data.pdb_dir = 'pdb'  # Directory containing .npy files
  data.max_samples = 1000  # Limit for testing (None for full dataset)
  data.global_min = 0.0  # Will be calculated from actual data
  data.global_max = 50000.0  # Will be calculated from actual data
  data.contact_threshold = 8.0  # Angstrom threshold for contact definition (if needed)
  data.return_distance = True  # Return distance maps instead of binary contacts

  # Training data statistics (will be updated automatically)
  training.num_train_data = 800  # Placeholder - updated by dataset

  # Evaluation data statistics (will be updated automatically)
  eval_config = config.eval
  eval_config.batch_size = 50  # Batch size for evaluation
  eval_config.num_samples = 1000  # Number of samples to generate for evaluation
  eval_config.num_test_data = 200  # Placeholder - updated by dataset
  eval_config.enable_bpd = False  # Disable likelihood (NLL/NELBO) calculation
  eval_config.enable_loss = False  # Disable loss evaluation
  eval_config.num_nelbo = 1  # Disable NELBO evaluations

  # model - Adjusted for protein data with VE-specific settings
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = True  # Important for VE SDE
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128  # Base number of filters
  model.ch_mult = (1, 2, 2, 2)  # Channel multipliers for each resolution
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)  # Add attention at 16x16 resolution
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3
  model.dropout = 0.1  # Add dropout for regularization
  
  # VE SDE parameters adjusted for protein data in [0,1] range
  model.sigma_max = 1.0  # Reduced from 90.0 - appropriate for [0,1] data
  model.sigma_min = 0.01  # Keep existing value, appropriate for [0,1] scale

  # Protein-specific model parameters
  model.symmetric_loss = True  # Enforce symmetry in protein contact maps
  model.contact_map_loss_weight = 1.0  # Weight for contact map specific losses
  model.distance_loss_type = 'mse'  # Loss type for distance prediction

  # Optimizer settings
  optim = config.optim
  optim.weight_decay = 1e-4
  optim.optimizer = 'Adam'
  optim.lr = 2e-4  # Learning rate
  optim.beta1 = 0.9
  optim.beta2 = 0.999
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.0

  # flow - Keep similar but adjust for protein data
  flow = config.flow
  flow.model = 'wolf'
  flow.lr = 1e-3
  flow.ema_rate = 0.999
  flow.optim_reset = False
  flow.nblocks = '16-16'
  flow.intermediate_dim = 512
  flow.resblock_type = 'resflow'
  flow.squeeze = False  # Disable squeeze to keep 1-channel input
  flow.model_config = 'flow_models/wolf/wolf_configs/protein/32x32/glow/resflow-gaussian-uni.json'
  flow.rank = 1
  flow.local_rank = 0
  flow.batch_size = 16  # Reduced for protein maps
  flow.eval_batch_size = 4
  flow.batch_steps = 1
  flow.init_batch_size = 32  # Reduced
  flow.epochs = 500
  flow.valid_epochs = 1
  flow.seed = 65537
  flow.train_k = 1
  flow.log_interval = 10
  flow.warmup_steps = 500
  flow.lr_decay = 0.999997
  flow.beta1 = 0.9
  flow.beta2 = 0.999
  flow.eps = 1e-8
  flow.weight_decay = 0
  flow.amsgrad = True
  flow.grad_clip = 0
  flow.dataset = 'protein_contact_map'
  flow.category = None
  flow.image_size = 32  # Match data.image_size since squeeze is disabled
  flow.workers = 4
  flow.n_bits = 8
  flow.recover = -1

  # Device configuration
  config.device = torch.device('cuda') if config.training.batch_size > 1 else torch.device('cpu')

  # Protein-specific evaluation metrics
  config.protein_eval = type('ProteinEval', (), {})()
  config.protein_eval.compute_contact_precision = True
  config.protein_eval.compute_distance_mae = True
  config.protein_eval.compute_symmetry_loss = True
  config.protein_eval.contact_thresholds = [6.0, 8.0, 10.0, 12.0]  # Multiple thresholds
  config.protein_eval.precision_thresholds = ['L/10', 'L/5', 'L/2', 'L']  # Top L/x contacts

  # Checkpointing
  config.checkpoint_dir = './checkpoints/protein_contact_maps_ve/'
  config.checkpoint_freq = 10000
  config.keep_checkpoint_max = 5

  # Logging
  config.log_dir = './logs/protein_contact_maps_ve/'
  config.wandb_project = 'protein-diffusion-ve'  # For Weights & Biases logging
  config.wandb_entity = None  # Your W&B username/team

  return config


def get_protein_small_config():
  """Smaller config for testing or limited computational resources."""
  config = get_config()

  # Reduce model size
  config.model.nf = 64
  config.model.ch_mult = (1, 2, 2)
  config.model.num_res_blocks = 2
  config.model.attn_resolutions = (16,)

  # Reduce data size
  config.data.image_size = 128
  config.data.max_sequence_length = 128

  # Reduce batch sizes
  config.training.batch_size = 8
  config.eval.batch_size = 4
  config.flow.batch_size = 8

  # Reduce training iterations
  config.training.n_iters = 500000

  return config


def get_protein_large_config():
  """Larger config for high-resolution protein maps."""
  config = get_config()

  # Increase model capacity
  config.model.nf = 256
  config.model.ch_mult = (1, 2, 2, 4, 4)
  config.model.num_res_blocks = 6
  config.model.attn_resolutions = (32, 16, 8)

  # Increase data size
  config.data.image_size = 512
  config.data.max_sequence_length = 512

  # Adjust batch sizes (may need to reduce based on GPU memory)
  config.training.batch_size = 4
  config.eval.batch_size = 2
  config.flow.batch_size = 4

  # Increase training iterations
  config.training.n_iters = 2000000

  return config
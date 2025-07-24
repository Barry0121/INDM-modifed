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
"""Training NCSNv3 on Protein Contact Maps with continuous sigmas for NLL evaluation."""

from configs.default_celeba_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'vpsde'
  training.continuous = True
  training.reduce_mean = True

  # sampling
  sampling = config.sampling
  sampling.method = 'ode'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'

  # data - Modified for protein contact maps
  data = config.data
  data.dataset = 'PROTEIN_CONTACT_MAP'  # Custom dataset name
  data.centered = True
  data.image_size = 256  # Typical protein map size
  data.num_channels = 1  # Contact/distance maps are single channel
  data.uniform_dequantization = False  # Not needed for continuous distance maps
  data.random_flip = False  # Proteins have specific orientation

  # Protein-specific data parameters for NLL evaluation
  data.contact_threshold = 8.0  # Angstrom threshold for contact definition
  data.return_distance = True  # Return distance maps instead of binary contacts
  data.max_sequence_length = 256  # Maximum protein length to consider
  data.min_sequence_length = 50   # Minimum protein length to consider
  data.normalize_distances = True  # Normalize distances to [0,1]
  data.max_distance_cutoff = 50.0  # Maximum distance to consider (Angstroms)
  data.pad_value = 1.0  # Value to use for padding (large distance)

  # Data paths for protein contact maps
  data.train_data_path = '/path/to/protein/train/maps/'
  data.eval_data_path = '/path/to/protein/eval/maps/'

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = False
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'none'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.embedding_type = 'positional'
  model.fourier_scale = 16
  model.conv_size = 3

  # flow - Adapted for protein contact maps NLL evaluation
  flow = config.flow
  flow.model = 'wolf'
  flow.lr = 1e-3
  flow.ema_rate = 0.999
  flow.optim_reset = False
  flow.nblocks = '16-16'
  flow.intermediate_dim = 512
  flow.resblock_type = 'resflow'

  # Flow model config path for protein contact maps
  flow.model_config = 'flow_models/wolf/wolf_configs/protein/256x256/glow/resflow-gaussian-uni.json'
  flow.rank = 1
  flow.local_rank = 0
  flow.batch_size = 64  # Reduced from 512 for protein maps
  flow.eval_batch_size = 4
  flow.batch_steps = 1
  flow.init_batch_size = 128  # Reduced from 1024 for protein maps
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
  flow.dataset = 'protein_contact_map'  # Changed from 'celeba'
  flow.category = None
  flow.image_size = 256  # Changed from 64 to match protein maps
  flow.workers = 4
  flow.n_bits = 8
  flow.recover = -1

  # NLL-specific evaluation parameters for proteins
  config.nll_eval = type('NLLEval', (), {})()
  config.nll_eval.likelihood_weighting = True  # Important for accurate NLL
  config.nll_eval.importance_sampling = True   # Helps with NLL estimation
  config.nll_eval.rtol = 1e-5  # Relative tolerance for ODE solver
  config.nll_eval.atol = 1e-5  # Absolute tolerance for ODE solver
  config.nll_eval.method = 'RK45'  # ODE solver method for likelihood computation
  config.nll_eval.eps = 1e-5  # Small epsilon for numerical stability

  # Protein-specific NLL evaluation
  config.nll_eval.compute_bpd = True  # Bits per dimension
  config.nll_eval.compute_elbo = True  # Evidence lower bound
  config.nll_eval.protein_nll_metrics = True  # Enable protein-specific NLL metrics
  config.nll_eval.contact_conditional_nll = True  # Conditional NLL for contacts
  config.nll_eval.distance_conditional_nll = True  # Conditional NLL for distances
  config.nll_eval.symmetry_regularization = 0.1  # Weight for symmetry in likelihood

  return config


def get_protein_nll_small_config():
  """Smaller config for NLL evaluation with limited computational resources."""
  config = get_config()

  # Reduce model size
  config.model.nf = 64
  config.model.ch_mult = (1, 2, 2)
  config.model.num_res_blocks = 2
  config.model.attn_resolutions = (16,)

  # Reduce data size
  config.data.image_size = 128
  config.data.max_sequence_length = 128

  # Reduce batch sizes for NLL computation
  config.flow.batch_size = 32
  config.flow.eval_batch_size = 2
  config.flow.init_batch_size = 64
  config.flow.image_size = 128

  # Adjust NLL evaluation parameters for smaller model
  config.nll_eval.rtol = 1e-4
  config.nll_eval.atol = 1e-4

  return config


def get_protein_nll_large_config():
  """Larger config for high-precision NLL evaluation on large protein maps."""
  config = get_config()

  # Increase model capacity for better likelihood estimation
  config.model.nf = 256
  config.model.ch_mult = (1, 2, 2, 4, 4)
  config.model.num_res_blocks = 6
  config.model.attn_resolutions = (32, 16, 8)

  # Increase data size
  config.data.image_size = 512
  config.data.max_sequence_length = 512

  # Adjust batch sizes (may need further reduction based on GPU memory)
  config.flow.batch_size = 16  # Much smaller due to memory requirements
  config.flow.eval_batch_size = 1
  config.flow.init_batch_size = 32
  config.flow.image_size = 512

  # Higher precision for NLL evaluation on large models
  config.nll_eval.rtol = 1e-6
  config.nll_eval.atol = 1e-6
  config.nll_eval.method = 'DOP853'  # Higher-order method for better precision

  return config


def get_protein_nll_fast_config():
  """Fast config for approximate NLL evaluation during training."""
  config = get_config()

  # Use faster but less precise settings
  config.nll_eval.rtol = 1e-3
  config.nll_eval.atol = 1e-3
  config.nll_eval.method = 'RK23'  # Faster lower-order method
  config.nll_eval.likelihood_weighting = False  # Disable for speed
  config.nll_eval.importance_sampling = False   # Disable for speed

  # Smaller batch sizes for faster evaluation
  config.flow.batch_size = 32
  config.flow.eval_batch_size = 8

  # Reduce some protein-specific computations
  config.nll_eval.contact_conditional_nll = False
  config.nll_eval.distance_conditional_nll = False

  return config
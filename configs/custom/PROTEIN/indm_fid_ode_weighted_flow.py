# coding=utf-8
# Custom training configuration: Full dataset with weighted flow output

from configs.default_cifar10_configs import get_default_configs

def get_config():
  config = get_default_configs()
  
  # training - Use default dataset size (no limit)
  training = config.training
  training.sde = 'vpsde'
  training.continuous = True
  training.reduce_mean = True
  training.likelihood_weighting = False
  training.importance_sampling = False
  # training.num_train_data = 50000  # Keep default from base config

  # sampling
  sampling = config.sampling
  sampling.method = 'ode'
  sampling.predictor = 'euler_maruyama'
  sampling.corrector = 'none'

  # data
  data = config.data
  data.dataset = 'PROTEIN'
  data.image_size = 32
  data.num_channels = 1
  data.centered = True

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

  # flow - CUSTOM: Add flow output weight parameter
  flow = config.flow
  flow.model = 'wolf'
  flow.lr = 1e-3
  flow.ema_rate = 0.999
  flow.optim_reset = False
  flow.nblocks = '16-16'
  flow.intermediate_dim = 512
  flow.resblock_type = 'resflow'
  flow.output_weight = 1e-3  # CUSTOM: Flow output weight to reduce value spread

  flow.model_config = 'flow_models/wolf/wolf_configs/protein/resflow-gaussian-uni.json'
  flow.rank = 1
  flow.local_rank = 0
  flow.batch_size = 128
  flow.eval_batch_size = 1
  flow.batch_steps = 1
  flow.init_batch_size = 256
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
  flow.dataset = 'protein'
  flow.category = None
  flow.image_size = 32
  flow.workers = 4
  flow.n_bits = 8
  flow.recover = -1

  # evaluation
  evaluate = config.eval
  evaluate.enable_bpd = False

  return config
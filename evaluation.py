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

"""Utility functions for computing FID/Inception scores."""

import six
import logging
import os
import io
import torch
import numpy as np
import socket
import datasets
import sampling_lib
from scipy.linalg import sqrtm
import torch.nn.functional as F

from cleanfid import fid as fid_calculator

try:
    from prdc import compute_prdc
    PRDC_AVAILABLE = True
except ImportError:
    PRDC_AVAILABLE = False
    logging.warning("prdc package not available. Sample diversity and coverage metrics will be skipped.")

import tensorflow as tf
import tensorflow_gan as tfgan
import tensorflow_hub as tfhub

INCEPTION_TFHUB = 'https://tfhub.dev/tensorflow/tfgan/eval/inception/1'
INCEPTION_OUTPUT = 'logits'
INCEPTION_FINAL_POOL = 'pool_3'
_DEFAULT_DTYPES = {
  INCEPTION_OUTPUT: tf.float32,
  INCEPTION_FINAL_POOL: tf.float32
}
INCEPTION_DEFAULT_IMAGE_SIZE = 299


def get_inception_model(inceptionv3=False):
  if inceptionv3:
    return tfhub.load(
      'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/4')
  else:
    return tfhub.load(INCEPTION_TFHUB)


def load_dataset_stats(config, assetdir):
  """Load the pre-computed dataset statistics."""
  if config.data.dataset == 'CIFAR10':
    filename = assetdir + '/cifar10_stats.npz'
  elif config.data.dataset == 'IMAGENET32':
    filename = assetdir + '/imagenet32_stats.npz'
  elif config.data.dataset == 'CELEBA':
    filename = assetdir + '/celeba_stats.npz'
  elif config.data.dataset == 'LSUN':
    filename = assetdir + f'/LSUN_{config.data.category}_{config.data.image_size}_stats.npz'
  elif config.data.dataset == 'PROTEIN_CONTACT_MAP':
    filename = assetdir + '/protein_contact_map_stats.npz'
  else:
    raise ValueError(f'Dataset {config.data.dataset} stats not found.')

  with tf.io.gfile.GFile(filename, 'rb') as fin:
    stats = np.load(fin)
    return stats


def classifier_fn_from_tfhub(output_fields, inception_model,
                             return_tensor=False):
  """Returns a function that can be as a classifier function.

  Copied from tfgan but avoid loading the model each time calling _classifier_fn

  Args:
    output_fields: A string, list, or `None`. If present, assume the module
      outputs a dictionary, and select this field.
    inception_model: A model loaded from TFHub.
    return_tensor: If `True`, return a single tensor instead of a dictionary.

  Returns:
    A one-argument function that takes an image Tensor and returns outputs.
  """
  if isinstance(output_fields, six.string_types):
    output_fields = [output_fields]

  def _classifier_fn(images):
    output = inception_model(images)
    if output_fields is not None:
      output = {x: output[x] for x in output_fields}
    if return_tensor:
      assert len(output) == 1
      output = list(output.values())[0]
    return tf.nest.map_structure(tf.compat.v1.layers.flatten, output)

  return _classifier_fn


@tf.function
def run_inception_jit(inputs,
                      inception_model,
                      num_batches=1,
                      inceptionv3=False):
  """Running the inception network. Assuming input is within [0, 255]."""
  if not inceptionv3:
    inputs = (tf.cast(inputs, tf.float32) - 127.5) / 127.5
  else:
    inputs = tf.cast(inputs, tf.float32) / 255.

  return tfgan.eval.run_classifier_fn(
    inputs,
    num_batches=num_batches,
    classifier_fn=classifier_fn_from_tfhub(None, inception_model),
    dtypes=_DEFAULT_DTYPES)


@tf.function
def run_inception_distributed(input_tensor,
                              inception_model,
                              num_batches=1,
                              inceptionv3=False):
  """Distribute the inception network computation to all available TPUs.

  Args:
    input_tensor: The input images. Assumed to be within [0, 255].
    inception_model: The inception network model obtained from `tfhub`.
    num_batches: The number of batches used for dividing the input.
    inceptionv3: If `True`, use InceptionV3, otherwise use InceptionV1.

  Returns:
    A dictionary with key `pool_3` and `logits`, representing the pool_3 and
      logits of the inception network respectively.
  """
  num_tpus = torch.cuda.device_count()
  input_tensors = tf.split(input_tensor, num_tpus, axis=0)
  pool3 = []
  logits = [] if not inceptionv3 else None
  device_format = '/GPU:{}'
  for i, tensor in enumerate(input_tensors):
    with tf.device(device_format.format(i)):
      tensor_on_device = tf.identity(tensor)
      res = run_inception_jit(
        tensor_on_device, inception_model, num_batches=num_batches,
        inceptionv3=inceptionv3)

      if not inceptionv3:
        pool3.append(res['pool_3'])
        logits.append(res['logits'])  # pytype: disable=attribute-error
      else:
        pool3.append(res)

  with tf.device('/CPU'):
    return {
      'pool_3': tf.concat(pool3, axis=0),
      'logits': tf.concat(logits, axis=0) if not inceptionv3 else None
    }

def compute_fid_and_is(config, score_model, flow_model, sampling_fn, step, sample_dir, assetdir, num_data, eval=False,
                       inverse_scaler=None, this_sample_dir=None, scaler=None, data_mean=None):
  ip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  ip.connect(("8.8.8.8", 80))
  ip = ip.getsockname()[0]

  inceptionv3 = config.data.image_size >= 256
  if config.data.dataset in ['CIFAR10', 'IMAGENET32']:
    samples_dir = tf.io.gfile.glob(os.path.join(sample_dir, "sample*.npz"))
    # Use inceptionV3 for images with resolution higher than 256.

    for sample_name in samples_dir:
      sampling_idx = int(sample_name.split('/')[-1].split('_')[1].split('.')[0])
      # sampling_idx = 0
      _, samples = sampling_lib.get_samples(config, score_model, flow_model, sampling_fn, step, sampling_idx,
                                            sample_dir, temperature=config.sampling.temperature, this_sample_dir=this_sample_dir,
                                            inverse_scaler=inverse_scaler, scaler=scaler, data_mean=data_mean)
    inception_model = get_inception_model(inceptionv3=inceptionv3)
    samples_dir = tf.io.gfile.glob(os.path.join(this_sample_dir, "sample*.npz"))
    for sample_name in samples_dir:
      sampling_idx = int(sample_name.split('/')[-1].split('_')[1].split('.')[0])
      # sampling_idx = 0
      _, samples = sampling_lib.get_samples(config, score_model, flow_model, sampling_fn, step, sampling_idx,
                                            sample_dir, temperature=config.sampling.temperature, this_sample_dir=this_sample_dir,
                                            inverse_scaler=inverse_scaler, scaler=scaler, data_mean=data_mean)
      latents = sampling_lib.get_latents(config, samples, inception_model, inceptionv3, step, sampling_idx, this_sample_dir)
      sampling_lib.save_statistics(config, latents, inceptionv3, step, sampling_idx, this_sample_dir)
    del inception_model
  elif config.data.dataset == 'PROTEIN_CONTACT_MAP':
    # For PROTEIN_CONTACT_MAP dataset, we skip inception model computation since it's not image data
    # The samples should already be generated by the sampling process
    # logging.info('Skipping inception model computation for PROTEIN_CONTACT_MAP dataset')
    # Skip FID computation entirely and use protein-specific metrics only
    compute_protein_metrics(config, this_sample_dir, assetdir, step, num_data)
  tf.keras.backend.clear_session()
  torch.cuda.empty_cache()

def compute_fid_and_is_(config, assetdir, inceptionv3, ckpt, dataset, name='/0', sample_dir='', latents='',
                       num_data=1000):
  # Special handling for protein data - should not reach here but add safeguard
  if config.data.dataset == 'PROTEIN_CONTACT_MAP':
    logging.info("PROTEIN_CONTACT_MAP detected in compute_fid_and_is_ - using protein-specific metrics")
    compute_protein_metrics(config, sample_dir, assetdir, ckpt, num_data)
    return

  if config.data.dataset in ['CIFAR10', 'IMAGENET32']:
    compute_fid_and_is_cifar10(config, assetdir, inceptionv3, ckpt, name=name, sample_dir=sample_dir, latents=latents,
                               num_data=num_data)
  elif config.data.dataset in ['FFHQ', 'LSUN', 'CelebAHQ', 'CELEBA', 'STL10', 'IMAGENET64', 'CIFAR100']:
    compute_fid_(config, assetdir, inceptionv3, ckpt, dataset, name=name, sample_dir=sample_dir, latents=latents,
                 num_data=num_data)
    if config.data.dataset == 'STL10':
      compute_is_stl10(config, inceptionv3, ckpt, name=name, sample_dir=sample_dir, latents=latents)
  else:
    raise NotImplementedError


def normalize_generated_samples_for_fid(config, sample_dir):
  """
  Normalize generated samples to proper range for FID computation.
  This ensures generated samples have the same value range as real data.
  """
  logging.info(f"Normalizing generated samples in {sample_dir} for FID computation...")

  # Get all sample files
  sample_files = tf.io.gfile.glob(os.path.join(sample_dir, "sample*.npy")) + \
                 tf.io.gfile.glob(os.path.join(sample_dir, "sample*.npz"))

  if not sample_files:
    logging.warning(f"No sample files found in {sample_dir} for normalization")
    return

  for sample_file in sample_files:
    try:
      # Load the sample
      if sample_file.endswith('.npy'):
        with tf.io.gfile.GFile(sample_file, 'rb') as f:
          samples = np.load(f)
      else:  # .npz file
        with tf.io.gfile.GFile(sample_file, 'rb') as f:
          data = np.load(f)
          if 'samples' in data:
            samples = data['samples']
          else:
            samples = data[list(data.keys())[0]]

      original_min, original_max = np.min(samples), np.max(samples)
      logging.info(f"Sample {os.path.basename(sample_file)}: original range [{original_min:.4f}, {original_max:.4f}]")

      # Normalize samples to [0, 255] range expected by FID computation
      # First normalize to [0, 1], then scale to [0, 255]
      if original_max > original_min:
        # Normalize to [0, 1]
        samples_normalized = (samples - original_min) / (original_max - original_min)
        # Scale to [0, 255]
        samples_normalized = samples_normalized * 255.0
      else:
        # Handle edge case where all values are the same
        samples_normalized = np.ones_like(samples) * 127.5  # Middle value

      # Ensure values are in proper format
      samples_normalized = samples_normalized.astype(np.uint8).astype(np.float32)

      new_min, new_max = np.min(samples_normalized), np.max(samples_normalized)
      logging.info(f"Sample {os.path.basename(sample_file)}: normalized range [{new_min:.4f}, {new_max:.4f}]")

      # Save the normalized samples back
      if sample_file.endswith('.npy'):
        with tf.io.gfile.GFile(sample_file, 'wb') as f:
          np.save(f, samples_normalized)
      else:  # .npz file
        with tf.io.gfile.GFile(sample_file, 'wb') as f:
          if 'samples' in data:
            np.savez_compressed(f, samples=samples_normalized, **{k: v for k, v in data.items() if k != 'samples'})
          else:
            key_name = list(data.keys())[0]
            np.savez_compressed(f, **{key_name: samples_normalized, **{k: v for k, v in data.items() if k != key_name}})

    except Exception as e:
      logging.error(f"Error normalizing sample file {sample_file}: {e}")
      continue

  logging.info(f"Completed normalization of {len(sample_files)} sample files")


def compute_fid_(config, assetdir, inceptionv3, ckpt, dataset, name='/0', sample_dir='', latents='', num_data=1000):
  # Skip cleanfid computation for protein data since it's not compatible with image-based FID
  if config.data.dataset == 'PROTEIN_CONTACT_MAP':
    # logging.info(f"Skipping cleanfid computation for PROTEIN_CONTACT_MAP dataset - using protein-specific metrics only")
    # Only compute protein-specific metrics
    compute_protein_metrics(config, sample_dir, assetdir, ckpt, num_data)
    return

  # For other datasets, use regular cleanfid computation
  # if config.data.dataset == 'LSUN':
  #    fids = fid_ttur.calculate_fid_given_paths([sample_dir, assetdir + f'/LSUN_{config.data.category}_{config.data.image_size}_stats.npz'], './', low_profile=False)
  # elif config.data.dataset == 'FFHQ':

  # Normalize generated samples before FID computation (skip for protein data as it's already correctly ranged)
  if config.data.dataset != 'PROTEIN_CONTACT_MAP':
    normalize_generated_samples_for_fid(config, sample_dir)

  # Mine
  fids = fid_calculator.compute_fid(config=config, mode='clean', fdir1=sample_dir, sigma_min=config.model.sigma_min,
                                    dataset_name=config.data.dataset, assetdir=assetdir, dataset=dataset,
                                    dequantization=True,
                                    num_data=num_data)

  # else:
  #    raise NotImplementedError
  print(fids)
  logging.info(f"{sample_dir}_ckpt-%d_{name} --- FID: {fids}" % (ckpt))

  if len(name.split('.')) == 1:
    name = f'report_{name}.npz'
  else:
    name = f'report_{name.split(".")[0]}.npz'
  if not os.path.join(sample_dir, name):
    with tf.io.gfile.GFile(os.path.join(sample_dir, name),
                           "wb") as f:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, fids=fids)
      f.write(io_buffer.getvalue())


def compute_is_stl10(config, inceptionv3, ckpt, name='/0', sample_dir='', latents=''):
  all_logits = []
  all_pools = []
  if latents == '':
    stats = tf.io.gfile.glob(os.path.join(sample_dir, "statistics_*.npz"))
    for stat_file in stats:
      with tf.io.gfile.GFile(stat_file, "rb") as fin:
        stat = np.load(fin)
        print("stat : ", stat)
        if not inceptionv3:
          all_logits.append(stat["logits"])
        all_pools.append(stat["pool_3"])
  else:
    if not inceptionv3:
      all_logits.append(latents["logits"])
    all_pools.append(latents["pool_3"])
  if not inceptionv3:
    all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
  all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

  all_use = False
  if all_use:
    inception_score = calculate_inception_score_CDSM(config, all_logits, inceptionv3)
  else:
    inception_score = calculate_inception_score_styleGAN(all_logits, inceptionv3, ckpt, name, sample_dir)

  logging.info(f'Inception score: {np.mean(inception_score)}')
  if len(name.split('.')) == 1:
    name = f'report_{name}.npz'
  else:
    name = f'report_{name.split(".")[0]}.npz'
  if not os.path.join(sample_dir, name):
    with tf.io.gfile.GFile(os.path.join(sample_dir, name),
                           "wb") as f:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, IS=inception_score)
      f.write(io_buffer.getvalue())


def compute_fid_and_is_cifar10(config, assetdir, inceptionv3, ckpt, name='/0', sample_dir='', latents='',
                               num_data=None):
  # Compute inception scores, FIDs and KIDs.
  # Load all statistics that have been previously computed and saved for each host
  all_logits = []
  all_pools = []

  if latents == '':
    if config.sampling.pc_denoise:
      stats = tf.io.gfile.glob(os.path.join(sample_dir, f"statistics_denoise_{config.sampling.pc_denoise_time}_*.npz"))
    elif config.sampling.more_step:
      stats = tf.io.gfile.glob(os.path.join(sample_dir, f"statistics_more_step_*.npz"))
    else:
      stats = tf.io.gfile.glob(os.path.join(sample_dir, "statistics_*.npz"))
    logging.info(f'sample_dir: {sample_dir}')
    for stat_file in stats:
      with tf.io.gfile.GFile(stat_file, "rb") as fin:
        stat = np.load(fin)
        print("stat : ", stat)
        if not inceptionv3:
          all_logits.append(stat["logits"])
        all_pools.append(stat["pool_3"])
  else:
    if not inceptionv3:
      all_logits.append(latents["logits"])
    all_pools.append(latents["pool_3"])

  if num_data != None:
    num_samples = num_data
  else:
    num_samples = 50000

  if not inceptionv3:
    all_logits = np.concatenate(all_logits, axis=0)
  all_pools = np.concatenate(all_pools, axis=0)

  for k in range(len(all_logits) // num_samples):

    # Load pre-computed dataset statistics.
    print(f"assetdir: {assetdir}")
    data_stats = load_dataset_stats(config, assetdir)
    data_pools = data_stats["pool_3"]

    all_use = True
    if all_use:
      inception_score = calculate_inception_score_CDSM(all_logits[k * num_samples: (k + 1) * num_samples],
                                                       inceptionv3)
    else:
      inception_score = calculate_inception_score_styleGAN(all_logits[k * num_samples: (k + 1) * num_samples],
                                                           inceptionv3, ckpt, name, sample_dir)

    fid = tfgan.eval.frechet_classifier_distance_from_activations(
      data_pools, all_pools[k * num_samples: (k + 1) * num_samples])
    # Hack to get tfgan KID work for eager execution.
    tf_data_pools = tf.convert_to_tensor(data_pools)
    tf_all_pools = tf.convert_to_tensor(all_pools[k * num_samples: (k + 1) * num_samples])
    kid = tfgan.eval.kernel_classifier_distance_from_activations(
      tf_data_pools, tf_all_pools).numpy()
    del tf_data_pools, tf_all_pools
    name = name.split('/')[-1]

    logging.info(
      f"{sample_dir}_ckpt-%d_{name}_num_data-{num_data} --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
        ckpt, np.mean(inception_score), fid, kid))

  if len(name.split('.')) == 1:
    name = f'report_{name}.npz'
  else:

    name = f'report_{name.split(".")[0]}.npz'
  if not os.path.join(sample_dir, name):
    with tf.io.gfile.GFile(os.path.join(sample_dir, name),
                           "wb") as f:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
      f.write(io_buffer.getvalue())


def calculate_inception_score_styleGAN(all_logits, inceptionv3, ckpt, name, sample_dir):
  inception_scores = []
  for k in range(10):
    # indices = np.arange(k * 5000,(k+1) * 5000)

    if not inceptionv3:
      all_logit = all_logits[k * 5000: (k + 1) * 5000]

    print("all logits length : ", len(all_logit))
    assert len(all_logit) == 5000

    # Compute FID/KID/IS on all samples together.
    if not inceptionv3:
      inception_score = tfgan.eval.classifier_score_from_logits(all_logit)
      inception_scores.append(inception_score)
    else:
      inception_score = -1

    logging.info(
      f"{sample_dir}_ckpt-%d_{name} --- inception_score: %.6e" % (
        ckpt, inception_score))

  return inception_scores

def calculate_inception_score_CDSM(all_logits, inceptionv3):
  print("all logits length : ", len(all_logits))
  # assert len(all_logits) == config.eval.num_samples

  # Compute FID/KID/IS on all samples together.
  if not inceptionv3:
    inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
  else:
    inception_score = -1

  return inception_score

def get_bpd(config, eval_ds, scaler, nelbo_fn, nll_fn, score_model, flow_model=None, step=0, eval=False):
  with torch.no_grad():
    if config.flow.model != 'identity':
      flow_model.eval()
    if eval:
      num_data = min(5000, config.eval.num_test_data)  # Cap at 5000 samples for evaluation mode
    else:
      num_data = 10000

    # num_data = 1000

    # nelbo
    nelbo_bpds_full = []
    nelbo_residual_bpds_full = []
    for _ in range(config.eval.num_nelbo):
      nelbo_bpds = []
      nelbo_residual_bpds = []
      bpd_iter = iter(eval_ds)
      for batch_id in range((num_data - 1) // config.eval.batch_size + 1):
        eval_batch, _ = datasets.get_batch(config, bpd_iter, eval_ds)

        # Apply dequantization only for image data, not continuous protein data
        if config.data.dataset != 'PROTEIN_CONTACT_MAP':
          eval_batch = (255. * eval_batch + torch.rand_like(eval_batch)) / 256.
        logdet = None
        eval_batch = scaler(eval_batch)
        nelbo_bpd, nelbo_residual_bpd = nelbo_fn(score_model, flow_model, eval_batch, logdet)
        nelbo_bpd = nelbo_bpd.detach().cpu().numpy().reshape(-1)
        nelbo_bpds.extend(nelbo_bpd)

        nelbo_residual_bpd = nelbo_residual_bpd.detach().cpu().numpy().reshape(-1)
        nelbo_residual_bpds.extend(nelbo_residual_bpd)
      nelbo_bpds_full.append(np.mean(nelbo_bpds))
      nelbo_residual_bpds_full.append(np.mean(nelbo_residual_bpds))
      logging.info("step: %d, num samples: %d, mean nelbo bpd: %.5e, std nelbo bpd: %.5e" % (
      step, len(nelbo_bpds), np.mean(nelbo_bpds), np.std(nelbo_bpds)))
      logging.info("step: %d, num samples: %d, mean nelbo_residual bpd: %.5e, std nelbo_residual bpd: %.5e" % (
      step, len(nelbo_residual_bpds), np.mean(nelbo_residual_bpds), np.std(nelbo_residual_bpds)))
    logging.info("step: %d, average nelbo bpd out of %d evaluations: %.5e" % (
      step, len(nelbo_bpds_full), np.mean(nelbo_bpds_full)))
    logging.info("step: %d, average nelbo bpd out of %d evaluations: %.5e" % (
      step, len(nelbo_residual_bpds_full), np.mean(nelbo_residual_bpds_full)))

    if not eval:
      num_data = num_data // 10

    # nll wrong with eps=1e-5
    if config.eval.skip_nll_wrong:
      pass
    else:
      for _ in range(1):
        if config.eval.truncation_time == -1.:
          eps_bpd = 1e-5
        else:
          eps_bpd = config.eval.truncation_time
        nll_bpds = []
        bpd_iter = iter(eval_ds)
        for batch_id in range((num_data - 1) // config.eval.batch_size + 1):
          eval_batch, _ = datasets.get_batch(config, bpd_iter, eval_ds)
          # Apply dequantization only for image data, not continuous protein data
          if config.data.dataset != 'PROTEIN_CONTACT_MAP':
            eval_batch = (255. * eval_batch + torch.rand_like(eval_batch)) / 256.
          logdet = None
          eval_batch = scaler(eval_batch)
          nll_bpd = nll_fn(score_model, flow_model, eval_batch, logdet, residual=False, eps_bpd=eps_bpd)[0].detach().cpu().numpy().reshape(-1)
          nll_bpds.extend(nll_bpd)
          if eval:
            logging.info("step: %d, [NLL WRONG w/ eps=%.1e] num samples: %d, mean nll bpd: %.5e, std nll bpd: %.5e" % (
              step, eps_bpd, len(nll_bpds), np.mean(nll_bpds), np.std(nll_bpds)))
        logging.info("step: %d, [NLL WRONG w/ eps=%.1e] num samples: %d, mean nll bpd: %.5e, std nll bpd: %.5e" % (
        step, eps_bpd, len(nll_bpds), np.mean(nll_bpds), np.std(nll_bpds)))

    # nll correct with eps=1e-5
    for _ in range(1):
      if config.eval.truncation_time == -1.:
        eps_bpd = 1e-5
      else:
        eps_bpd = config.eval.truncation_time
      nll_bpds = []
      bpd_iter = iter(eval_ds)
      for batch_id in range((num_data - 1) // config.eval.batch_size + 1):
        eval_batch, _ = datasets.get_batch(config, bpd_iter, eval_ds)
        # Apply dequantization only for image data, not continuous protein data
        if config.data.dataset != 'PROTEIN_CONTACT_MAP':
          eval_batch = (255. * eval_batch + torch.rand_like(eval_batch)) / 256.
        logdet = None
        eval_batch = scaler(eval_batch)
        nll_bpd = nll_fn(score_model, flow_model, eval_batch, logdet, residual=True, eps_bpd=eps_bpd)[0].detach().cpu().numpy().reshape(-1)
        nll_bpds.extend(nll_bpd)
        if eval:
          logging.info("step: %d, [NLL CORRECT w/ eps=%.1e] num samples: %d, mean nll bpd: %.5e, std nll bpd: %.5e" % (
            step, eps_bpd, len(nll_bpds), np.mean(nll_bpds), np.std(nll_bpds)))
      logging.info("step: %d, [NLL CORRECT w/ eps=%.1e] num samples: %d, mean nll bpd: %.5e, std nll bpd: %.5e" % (
      step, eps_bpd, len(nll_bpds), np.mean(nll_bpds), np.std(nll_bpds)))

    # nll correct with their eps if eps != 1e-5
    if config.training.truncation_time != 1e-5:
      for _ in range(1):
        nll_bpds = []
        bpd_iter = iter(eval_ds)
        for batch_id in range((num_data - 1) // config.eval.batch_size + 1):
          eval_batch, _ = datasets.get_batch(config, bpd_iter, eval_ds)
          # Apply dequantization only for image data, not continuous protein data
          if config.data.dataset != 'PROTEIN_CONTACT_MAP':
            eval_batch = (255. * eval_batch + torch.rand_like(eval_batch)) / 256.
          logdet = None
          eval_batch = scaler(eval_batch)
          nll_bpd = nll_fn(score_model, flow_model, eval_batch, logdet, residual=True, eps_bpd=config.training.truncation_time)[0].detach().cpu().numpy().reshape(-1)
          nll_bpds.extend(nll_bpd)
          if eval:
            logging.info("step: %d, [NLL CORRECT w/ eps=eps] num samples: %d, mean nll bpd: %.5e, std nll bpd: %.5e" % (
              step, len(nll_bpds), np.mean(nll_bpds), np.std(nll_bpds)))
        logging.info("step: %d, [NLL CORRECT w/ eps=eps] num samples: %d, mean nll bpd: %.5e, std nll bpd: %.5e" % (
        step, len(nll_bpds), np.mean(nll_bpds), np.std(nll_bpds)))
    if config.flow.model != 'identity':
      flow_model.train()


def compute_protein_inception_score(data, batch_size=32):
  """
  Compute Inception Score for protein data using a simple approach.
  For protein data, we treat each sample as a feature vector and compute diversity.

  Args:
    data: numpy array of shape (N, ...) where N is number of samples
    batch_size: batch size for processing

  Returns:
    inception_score: computed IS value
  """
  if len(data.shape) > 2:
    data = data.reshape(data.shape[0], -1)

  # Normalize data
  data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-8)

  # Compute predictions using a simple classifier approach
  # For protein data, we use the data variance as a proxy for class predictions
  n_samples = data.shape[0]
  n_features = data.shape[1]

  # Create pseudo-predictions based on data variance and mean
  predictions = np.zeros((n_samples, min(10, n_features)))  # Assume 10 classes max

  for i in range(predictions.shape[1]):
    # Use different feature subsets to create pseudo-class probabilities
    start_idx = i * (n_features // predictions.shape[1])
    end_idx = (i + 1) * (n_features // predictions.shape[1])
    feature_subset = data[:, start_idx:end_idx]
    predictions[:, i] = np.mean(np.abs(feature_subset), axis=1)

  # Apply softmax to get probabilities
  predictions = F.softmax(torch.tensor(predictions), dim=1).numpy()

  # Compute IS = exp(E[KL(p(y|x) || p(y))])
  marginal = np.mean(predictions, axis=0)
  kl_divs = []

  for i in range(n_samples):
    kl_div = np.sum(predictions[i] * np.log(predictions[i] / (marginal + 1e-8) + 1e-8))
    kl_divs.append(kl_div)

  inception_score = np.exp(np.mean(kl_divs))
  return inception_score


def compute_protein_diversity_coverage(real_data, fake_data, config, nearest_k=5):
  """
  Compute sample diversity and coverage for protein data using PRDC.

  Args:
    real_data: numpy array of real samples (N_real, ...)
    fake_data: numpy array of fake samples (N_fake, ...)
    config: configuration object with SDE type information
    nearest_k: k for k-nearest neighbors

  Returns:
    dict with precision, recall, density, coverage metrics
  """
  if not PRDC_AVAILABLE:
    logging.warning("PRDC not available, returning dummy metrics")
    return {"precision": 0.0, "recall": 0.0, "density": 0.0, "coverage": 0.0}

  # Flatten data if needed and ensure 2D shape
  if len(real_data.shape) > 2:
    real_data = real_data.reshape(real_data.shape[0], -1)
  elif len(real_data.shape) == 1:
    real_data = real_data.reshape(-1, 1)

  if len(fake_data.shape) > 2:
    fake_data = fake_data.reshape(fake_data.shape[0], -1)
  elif len(fake_data.shape) == 1:
    fake_data = fake_data.reshape(-1, 1)

  # Check dimensional compatibility
  if real_data.shape[1] != fake_data.shape[1]:
    logging.warning(f"Dimensional mismatch: real_data has {real_data.shape[1]} features, fake_data has {fake_data.shape[1]} features")

    # Align dimensions by taking the minimum or using projection
    min_features = min(real_data.shape[1], fake_data.shape[1])
    max_features = max(real_data.shape[1], fake_data.shape[1])

    if min_features == 1 and max_features > 1:
      # If one dataset has only 1 feature and the other has many, repeat the single feature
      if real_data.shape[1] == 1:
        logging.info(f"Expanding real_data from {real_data.shape[1]} to {fake_data.shape[1]} features by replication")
        real_data = np.repeat(real_data, fake_data.shape[1], axis=1)
      else:
        logging.info(f"Expanding fake_data from {fake_data.shape[1]} to {real_data.shape[1]} features by replication")
        fake_data = np.repeat(fake_data, real_data.shape[1], axis=1)
    else:
      # Truncate to smaller dimension
      logging.info(f"Truncating both datasets to {min_features} features")
      real_data = real_data[:, :min_features]
      fake_data = fake_data[:, :min_features]

  # Conditional normalization based on SDE type
  if config.training.sde == 'vesde':  # VE
    logging.info("Using min-max normalization to [0,1] for VE SDE")
    # Use min-max normalization to [0,1] for VE
    real_min, real_max = np.min(real_data), np.max(real_data)
    fake_min, fake_max = np.min(fake_data), np.max(fake_data)
    
    if real_max > real_min:
      real_data_norm = (real_data - real_min) / (real_max - real_min)
    else:
      real_data_norm = np.ones_like(real_data) * 0.5
      
    if fake_max > fake_min:
      fake_data_norm = (fake_data - fake_min) / (fake_max - fake_min)
    else:
      fake_data_norm = np.ones_like(fake_data) * 0.5
  else:  # VP and SubVP
    logging.info(f"Using mean/std normalization for {config.training.sde} SDE")
    # Use mean/std normalization (current working approach)
    real_mean = np.mean(real_data, axis=0)
    real_std = np.std(real_data, axis=0) + 1e-8
    fake_mean = np.mean(fake_data, axis=0)
    fake_std = np.std(fake_data, axis=0) + 1e-8
    real_data_norm = (real_data - real_mean) / real_std
    fake_data_norm = (fake_data - fake_mean) / fake_std

  try:
    logging.debug(f"Computing PRDC with real_data shape: {real_data_norm.shape}, fake_data shape: {fake_data_norm.shape}")

    metrics = compute_prdc(
      real_features=real_data_norm,
      fake_features=fake_data_norm,
      nearest_k=nearest_k
    )
    return metrics
  except Exception as e:
    logging.warning(f"PRDC computation failed: {e}")
    return {"precision": 0.0, "recall": 0.0, "density": 0.0, "coverage": 0.0}


def compute_protein_fid_simple(real_data, fake_data):
  """
  Compute a simple FID-like metric for protein data without using Inception network.

  Args:
    real_data: numpy array of real samples (N_real, ...)
    fake_data: numpy array of fake samples (N_fake, ...)

  Returns:
    fid_score: computed FID-like score
  """
  # Flatten data if needed and ensure 2D shape
  if len(real_data.shape) > 2:
    real_data = real_data.reshape(real_data.shape[0], -1)
  elif len(real_data.shape) == 1:
    real_data = real_data.reshape(-1, 1)

  if len(fake_data.shape) > 2:
    fake_data = fake_data.reshape(fake_data.shape[0], -1)
  elif len(fake_data.shape) == 1:
    fake_data = fake_data.reshape(-1, 1)

  # Check dimensional compatibility
  if real_data.shape[1] != fake_data.shape[1]:
    logging.warning(f"FID: Dimensional mismatch: real_data has {real_data.shape[1]} features, fake_data has {fake_data.shape[1]} features")

    # Align dimensions by taking the minimum or using projection
    min_features = min(real_data.shape[1], fake_data.shape[1])
    max_features = max(real_data.shape[1], fake_data.shape[1])

    if min_features == 1 and max_features > 1:
      # If one dataset has only 1 feature and the other has many, repeat the single feature
      if real_data.shape[1] == 1:
        logging.info(f"FID: Expanding real_data from {real_data.shape[1]} to {fake_data.shape[1]} features by replication")
        real_data = np.repeat(real_data, fake_data.shape[1], axis=1)
      else:
        logging.info(f"FID: Expanding fake_data from {fake_data.shape[1]} to {real_data.shape[1]} features by replication")
        fake_data = np.repeat(fake_data, real_data.shape[1], axis=1)
    else:
      # Truncate to smaller dimension
      logging.info(f"FID: Truncating both datasets to {min_features} features")
      real_data = real_data[:, :min_features]
      fake_data = fake_data[:, :min_features]

  # Compute statistics
  mu_real = np.mean(real_data, axis=0)
  mu_fake = np.mean(fake_data, axis=0)

  sigma_real = np.cov(real_data, rowvar=False)
  sigma_fake = np.cov(fake_data, rowvar=False)

  # Handle case where covariance is scalar (single feature or single sample)
  if np.ndim(sigma_real) == 0:
    sigma_real = np.array([[sigma_real]])
  elif np.ndim(sigma_real) == 1:
    sigma_real = np.diag(sigma_real)

  if np.ndim(sigma_fake) == 0:
    sigma_fake = np.array([[sigma_fake]])
  elif np.ndim(sigma_fake) == 1:
    sigma_fake = np.diag(sigma_fake)

  # Add small epsilon to diagonal for numerical stability
  sigma_real += 1e-6 * np.eye(sigma_real.shape[0])
  sigma_fake += 1e-6 * np.eye(sigma_fake.shape[0])

  # Verify covariance matrix dimensions match
  if sigma_real.shape != sigma_fake.shape:
    logging.error(f"Covariance matrix shape mismatch: sigma_real {sigma_real.shape} vs sigma_fake {sigma_fake.shape}")
    return float('inf')  # Return infinite FID to indicate failure

  # Compute FID
  ssdiff = np.sum((mu_real - mu_fake) ** 2.0)
  try:
    covmean = sqrtm(sigma_real.dot(sigma_fake))
  except Exception as e:
    logging.error(f"Failed to compute matrix square root in FID: {e}")
    return float('inf')

  if np.iscomplexobj(covmean):
    covmean = covmean.real

  fid = ssdiff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
  return fid


def normalize_to_range(data, target_min=-1.0, target_max=1.0):
  """
  Normalize data from its actual range to target range (default [-1, 1]) using min-max normalization.

  NOTE: This function should no longer be needed for protein data since the sampling
  pipeline has been fixed to generate samples in the correct [-1, 1] range.

  Args:
    data: numpy array with any numeric values
    target_min: minimum value of target range (default: -1.0)
    target_max: maximum value of target range (default: 1.0)

  Returns:
    numpy array normalized to target range
  """
  # Convert to float32 if needed
  if data.dtype == np.uint8:
    data = data.astype(np.float32)

  # Find actual data range
  data_min = np.min(data)
  data_max = np.max(data)

  # Handle edge case where all values are the same
  if data_max == data_min:
    # Return middle value of target range
    return np.full_like(data, (target_max + target_min) / 2.0, dtype=np.float32)

  # Normalize to [0, 1] based on actual data range
  normalized_01 = (data - data_min) / (data_max - data_min)

  # Scale to target range
  return normalized_01 * (target_max - target_min) + target_min


def compute_protein_metrics(config, sample_dir, assetdir, step, num_data=1000):
  """
  Compute IS, sample diversity, and sample coverage for PROTEIN dataset.

  Args:
    config: configuration object
    sample_dir: directory containing generated samples
    assetdir: directory containing real data assets
    step: current training step
    num_data: number of samples to use
  """
  logging.info(f"Computing PROTEIN metrics for step {step}")

  try:
    # Load generated samples
    sample_files = tf.io.gfile.glob(os.path.join(sample_dir, "sample*.npz"))
    if not sample_files:
      logging.warning(f"No sample files found in {sample_dir}")
      return

    fake_samples = []
    for sample_file in sample_files:
      with tf.io.gfile.GFile(sample_file, "rb") as f:
        data = np.load(f)
        if 'samples' in data:
          fake_samples.append(data['samples'])
        else:
          # Take the first array in the file
          fake_samples.append(data[list(data.keys())[0]])

    if not fake_samples:
      logging.warning("No valid samples found in sample files")
      return

    fake_data = np.concatenate(fake_samples, axis=0)[:num_data]
    logging.info(f"Loaded {fake_data.shape[0]} fake samples with shape {fake_data.shape}")
    logging.info(f"Fake data dtype: {fake_data.dtype}, min: {np.min(fake_data):.4f}, max: {np.max(fake_data):.4f}")

    # Log fake data range and determine expected range based on config
    fake_min, fake_max = np.min(fake_data), np.max(fake_data)
    logging.info(f"Fake data range: [{fake_min:.4f}, {fake_max:.4f}]")

    # Determine expected range based on config
    if config.data.centered:
        expected_range = "[-1, 1]"
        range_min, range_max = -1.1, 1.1
    else:
        expected_range = "[0, 1]"
        range_min, range_max = -0.1, 1.1

    # Warn if data is outside expected range
    if fake_min < range_min or fake_max > range_max:
        logging.warning(f"WARNING: Fake data range [{fake_min:.4f}, {fake_max:.4f}] is outside expected {expected_range} range!")
        logging.warning("This indicates an issue with the sampling pipeline that should be investigated.")

    print(f"Num fake samples: {len(fake_data)}")

    # Load real data using the PDB dataset class to get proper contact maps
    try:
      from pdb_dataset import PDB

      logging.info("Loading real training data using PDB dataset class...")
      # Load the training split of the PDB dataset with config for proper normalization
      train_dataset = PDB(train=True, config=config)

      # Get the training matrices (normalized according to config.data.centered)
      train_matrices = train_dataset.get_matrices()[:num_data]

      # Convert to the same format as fake_data (add channel dimension if needed)
      if len(fake_data.shape) == 4 and fake_data.shape[-1] == 1:
        # Fake data has shape (N, 32, 32, 1), so expand real data
        real_data = np.expand_dims(train_matrices, axis=-1)
      else:
        # Keep original shape
        real_data = train_matrices

      logging.info(f"Loaded {real_data.shape[0]} real samples with shape {real_data.shape}")
      real_min, real_max = np.min(real_data), np.max(real_data)
      logging.info(f"Real data dtype: {real_data.dtype}, min: {real_min:.4f}, max: {real_max:.4f}")
      logging.info(f"Real data range: [{real_min:.4f}, {real_max:.4f}] (expected: {expected_range})")

      # Verify real and fake data are in the same range
      if abs(real_min - fake_min) > 0.2 or abs(real_max - fake_max) > 0.2:
        logging.warning(f"Real and fake data ranges differ significantly!")
        logging.warning(f"Real: [{real_min:.4f}, {real_max:.4f}], Fake: [{fake_min:.4f}, {fake_max:.4f}]")

      print(f"Num real samples: {len(real_data)}, Num fake samples: {len(fake_data)}")

    except Exception as e:
      logging.error(f"Failed to load real data from PDB dataset: {e}")
      logging.warning("Falling back to loading from assetdir (stats file)")

      # Fallback to original approach
      real_data_files = tf.io.gfile.glob(os.path.join(assetdir, "*protein*.npz"))
      if not real_data_files:
        real_data_files = tf.io.gfile.glob(os.path.join(assetdir, "*.npz"))

      if real_data_files:
        with tf.io.gfile.GFile(real_data_files[0], "rb") as f:
          real_data_dict = np.load(f)
          if 'data' in real_data_dict:
            real_data = real_data_dict['data'][:num_data]
          else:
            real_data = real_data_dict[list(real_data_dict.keys())[0]][:num_data]
        logging.info(f"Loaded {real_data.shape[0]} real samples with shape {real_data.shape}")
        logging.info(f"Real data dtype: {real_data.dtype}, min: {np.min(real_data):.4f}, max: {np.max(real_data):.4f}")
        print(f"Num real samples: {len(real_data)}, Num fake samples: {len(fake_data)}")
      else:
        logging.warning(f"No real data files found in {assetdir}, using synthetic reference")
        # Create synthetic reference data with similar statistics
        real_data = np.random.randn(*fake_data.shape) * np.std(fake_data) + np.mean(fake_data)
        logging.info(f"Generated synthetic real data with shape {real_data.shape}")

    # Add debugging info for dimension compatibility
    logging.info(f"Before metrics computation - Real: {real_data.shape}, Fake: {fake_data.shape}")

    # Compute metrics

    # 1. Inception Score
    is_score = compute_protein_inception_score(fake_data)
    logging.info(f"Step {step} - Protein Inception Score: {is_score:.6f}")

    # 2. Sample Diversity and Coverage (PRDC)
    prdc_metrics = compute_protein_diversity_coverage(real_data, fake_data, config)
    logging.info(f"Step {step} - PRDC Precision: {prdc_metrics['precision']:.6f}")
    logging.info(f"Step {step} - PRDC Recall: {prdc_metrics['recall']:.6f}")
    logging.info(f"Step {step} - PRDC Density: {prdc_metrics['density']:.6f}")
    logging.info(f"Step {step} - PRDC Coverage: {prdc_metrics['coverage']:.6f}")

    # 3. Simple FID for comparison
    simple_fid = compute_protein_fid_simple(real_data, fake_data)
    logging.info(f"Step {step} - Protein Simple FID: {simple_fid:.6f}")

    # Optional: Add tensorboard logging if available
    try:
      try:
        from torch.utils.tensorboard import SummaryWriter
        tensorboard_available = True
      except ImportError:
        try:
          from tensorboardX import SummaryWriter
          tensorboard_available = True
        except ImportError:
          tensorboard_available = False

      if tensorboard_available:
        import glob
        tb_dirs = glob.glob(os.path.join(os.path.dirname(sample_dir), "../tensorboard"))
        if tb_dirs:
          tb_writer = SummaryWriter(tb_dirs[0])
          tb_writer.add_scalar('Evaluation/Protein_Inception_Score', is_score, step)
          tb_writer.add_scalar('Evaluation/PRDC_Precision', prdc_metrics['precision'], step)
          tb_writer.add_scalar('Evaluation/PRDC_Recall', prdc_metrics['recall'], step)
          tb_writer.add_scalar('Evaluation/PRDC_Density', prdc_metrics['density'], step)
          tb_writer.add_scalar('Evaluation/PRDC_Coverage', prdc_metrics['coverage'], step)
          tb_writer.add_scalar('Evaluation/Protein_Simple_FID', simple_fid, step)
          tb_writer.close()
    except Exception as e:
      logging.debug(f"Tensorboard logging failed: {e}")

    # Save metrics
    metrics = {
      'inception_score': is_score,
      'precision': prdc_metrics['precision'],
      'recall': prdc_metrics['recall'],
      'density': prdc_metrics['density'],
      'coverage': prdc_metrics['coverage'],
      'simple_fid': simple_fid,
      'step': step
    }

    report_path = os.path.join(sample_dir, f"protein_metrics_step_{step}.npz")
    with tf.io.gfile.GFile(report_path, "wb") as f:
      io_buffer = io.BytesIO()
      np.savez_compressed(io_buffer, **metrics)
      f.write(io_buffer.getvalue())

    logging.info(f"Saved PROTEIN metrics to {report_path}")

  except Exception as e:
    logging.error(f"Error computing PROTEIN metrics: {e}")
    import traceback
    logging.error(traceback.format_exc())
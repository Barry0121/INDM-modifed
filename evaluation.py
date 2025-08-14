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
import tensorflow as tf
from scipy import linalg

from cleanfid import fid as fid_calculator

# Import original functions that are still needed for non-protein datasets
try:
    import evaluation_original
    get_inception_model = evaluation_original.get_inception_model
    compute_fid_and_is_cifar10 = evaluation_original.compute_fid_and_is_cifar10
    compute_is_stl10 = evaluation_original.compute_is_stl10
except ImportError:
    logging.warning("Could not import evaluation_original module; non-PROTEIN dataset can't be processed.")

# def load_dataset_stats(config, assetdir):
#   """Load the pre-computed dataset statistics."""
#   if config.data.dataset == 'CIFAR10':
#     filename = assetdir + '/cifar10_stats.npz'
#   elif config.data.dataset == 'IMAGENET32':
#     filename = assetdir + '/imagenet32_stats.npz'
#   elif config.data.dataset == 'CELEBA':
#     filename = assetdir + '/celeba_stats.npz'
#   elif config.data.dataset == 'LSUN':
#     filename = assetdir + f'/LSUN_{config.data.category}_{config.data.image_size}_{mode}_stats.npz'
#   else:
#     raise ValueError(f'Dataset {config.data.dataset} stats not found.')

#   with tf.io.gfile.GFile(filename, 'rb') as fin:
#     stats = np.load(fin)
#     return stats


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
  compute_fid_and_is_(config, assetdir, config.data.image_size >= 256, step, [],
                                  sample_dir=this_sample_dir, num_data=num_data)
  tf.keras.backend.clear_session()
  torch.cuda.empty_cache()

def compute_fid_and_is_(config, assetdir, inceptionv3, ckpt, dataset, name='/0', sample_dir='', latents='',
                       num_data=1000):
  if config.data.dataset in ['CIFAR10', 'IMAGENET32']:
    compute_fid_and_is_cifar10(config, assetdir, inceptionv3, ckpt, name=name, sample_dir=sample_dir, latents=latents,
                               num_data=num_data)
  elif config.data.dataset == 'PROTEIN':
    compute_protein_fid(config, assetdir, ckpt, name=name, sample_dir=sample_dir, num_data=num_data)
  elif config.data.dataset in ['FFHQ', 'LSUN', 'CelebAHQ', 'CELEBA', 'STL10', 'IMAGENET64', 'CIFAR100']:
    compute_fid_(config, assetdir, inceptionv3, ckpt, dataset, name=name, sample_dir=sample_dir, latents=latents,
                 num_data=num_data)
    if config.data.dataset == 'STL10':
      compute_is_stl10(config, inceptionv3, ckpt, name=name, sample_dir=sample_dir, latents=latents)
  else:
    raise NotImplementedError


def compute_fid_(config, assetdir, inceptionv3, ckpt, dataset, name='/0', sample_dir='', latents='', num_data=1000):
  # if config.data.dataset == 'LSUN':
  #    fids = fid_ttur.calculate_fid_given_paths([sample_dir, assetdir + f'/LSUN_{config.data.category}_{config.data.image_size}_stats.npz'], './', low_profile=False)
  # elif config.data.dataset == 'FFHQ':

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


def get_bpd(config, eval_ds, scaler, nelbo_fn, nll_fn, score_model, flow_model=None, step=0, eval=False):
  with torch.no_grad():
    if config.flow.model != 'identity':
      flow_model.eval()
    if eval:
      num_data = config.eval.num_test_data
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


# ============================================================================
# PROTEIN-SPECIFIC FID IMPLEMENTATION
# ============================================================================

def protein_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  """Compute the Frechet Distance between two multivariate Gaussians for protein data.

  Args:
    mu1: Mean of first distribution
    sigma1: Covariance matrix of first distribution
    mu2: Mean of second distribution
    sigma2: Covariance matrix of second distribution
    eps: Small value to add to diagonal for numerical stability

  Returns:
    Frechet distance as a scalar
  """
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
    logging.warning("Using eigenvalue decomposition fallback for matrix square root")
    
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
    # tr(sqrt(sigma1 * sigma2)) ≈ sum(sqrt(eigvals of sigma1 * sigma2))
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
        # Large imaginary components indicate numerical failure - use fallback
        logging.warning(f"Large imaginary component detected: {np.max(np.abs(covmean.imag))}, using fallback")
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


def get_protein_dataset_stats(config, num_data=1000):
  """Load or compute reference statistics for protein dataset.

  Args:
    config: Configuration object
    num_data: Number of samples to use for statistics

  Returns:
    (mu, sigma): Mean and covariance matrix of reference data
  """
  # Cache file path
  cache_dir = os.path.join(os.path.dirname(__file__), 'assets', 'stats')
  os.makedirs(cache_dir, exist_ok=True)
  cache_file = os.path.join(cache_dir, f'protein_stats_{config.data.image_size}_{num_data}.npz')

  # Try to load cached statistics
  if os.path.exists(cache_file):
    logging.info(f'Loading cached protein statistics from {cache_file}')
    stats = np.load(cache_file)
    return stats['mu'], stats['sigma']

  logging.info(f'Computing protein reference statistics for {num_data} samples')

  # Load protein dataset
  train_ds, eval_ds = datasets.get_dataset(config)

  # Collect reference samples
  ref_samples = []
  data_iter = iter(train_ds)
  samples_collected = 0

  while samples_collected < num_data:
    try:
      batch, _ = datasets.get_batch(config, data_iter, train_ds)
      # batch is already normalized to [0,1] by the dataset loader
      batch_np = batch.cpu().numpy()

      # Flatten spatial dimensions for each sample
      batch_flat = batch_np.reshape(batch_np.shape[0], -1)
      ref_samples.append(batch_flat)

      samples_collected += batch_np.shape[0]
      if samples_collected >= num_data:
        break
    except StopIteration:
      logging.warning(f'Only collected {samples_collected} samples (requested {num_data})')
      break

  # Concatenate all samples
  ref_samples = np.concatenate(ref_samples, axis=0)[:num_data]

  # Compute statistics
  mu = np.mean(ref_samples, axis=0)
  sigma = np.cov(ref_samples, rowvar=False)

  # Cache the results
  np.savez_compressed(cache_file, mu=mu, sigma=sigma)
  logging.info(f'Saved protein statistics to {cache_file}')

  return mu, sigma


def load_protein_samples(sample_dir, num_data=1000):
  """Load generated protein samples from sample directory.

  Args:
    sample_dir: Directory containing generated samples
    num_data: Maximum number of samples to load

  Returns:
    Generated samples as numpy array, shape (N, H*W*C)
  """
  # Find all sample files
  sample_files = tf.io.gfile.glob(os.path.join(sample_dir, "sample*.npz"))
  if not sample_files:
    raise ValueError(f'No sample files found in {sample_dir}')

  logging.info(f'Found {len(sample_files)} sample files in {sample_dir}')

  gen_samples = []
  samples_loaded = 0

  for sample_file in sorted(sample_files):
    if samples_loaded >= num_data:
      break

    with tf.io.gfile.GFile(sample_file, "rb") as f:
      data = np.load(f)['samples']

    # Data should be in [0,1] range for protein, convert from uint8 if needed
    if data.dtype == np.uint8:
      data = data.astype(np.float32) / 255.0
    else:
      data = data.astype(np.float32)

    # Ensure data is in [0,1] range
    data = np.clip(data, 0.0, 1.0)

    # Flatten spatial dimensions
    data_flat = data.reshape(data.shape[0], -1)
    gen_samples.append(data_flat)

    samples_loaded += data.shape[0]

  # Concatenate and limit to num_data
  gen_samples = np.concatenate(gen_samples, axis=0)[:num_data]

  logging.info(f'Loaded {gen_samples.shape[0]} generated protein samples')
  return gen_samples


def compute_protein_fid(config, assetdir, ckpt, name='/0', sample_dir='', num_data=1000):
  """Compute FID for protein distance maps using direct comparison.

  Args:
    config: Configuration object
    assetdir: Asset directory (for compatibility, not used)
    ckpt: Checkpoint step
    name: Name identifier for results
    sample_dir: Directory containing generated samples
    num_data: Number of samples to use for evaluation
  """
  logging.info(f'Computing protein FID for {num_data} samples')

  # Load reference statistics
  ref_mu, ref_sigma = get_protein_dataset_stats(config, num_data=num_data)

  # Load generated samples
  gen_samples = load_protein_samples(sample_dir, num_data=num_data)

  # Compute statistics for generated samples
  gen_mu = np.mean(gen_samples, axis=0)
  gen_sigma = np.cov(gen_samples, rowvar=False)

  # Compute FID
  fid_score = protein_frechet_distance(gen_mu, gen_sigma, ref_mu, ref_sigma)

  logging.info(f'{sample_dir}_ckpt-{ckpt}_{name} --- Protein FID: {fid_score:.6f}')

  # Save results (compatible with existing format)
  if len(name.split('.')) == 1:
    result_name = f'report_{name}.npz'
  else:
    result_name = f'report_{name.split(".")[0]}.npz'

  result_path = os.path.join(sample_dir, result_name)
  if not os.path.exists(result_path):
    with tf.io.gfile.GFile(result_path, "wb") as f:
      io_buffer = io.BytesIO()
      # Save FID score with same key as original for compatibility
      np.savez_compressed(io_buffer, fids=fid_score)
      f.write(io_buffer.getvalue())
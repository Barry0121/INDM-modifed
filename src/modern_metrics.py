"""
Modern implementations of evaluation metrics to replace tensorflow-gan.
Implements FID, Inception Score, and KID using PyTorch and modern libraries.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
import logging
from typing import Tuple, Optional, Union, Dict
import warnings


class ModernMetrics:
    """Modern implementation of evaluation metrics."""

    def __init__(self, device: str = 'cuda', feature_dim: int = 2048):
        self.device = device
        self.feature_dim = feature_dim

        # Initialize torchmetrics
        self.fid_metric = FrechetInceptionDistance(feature=feature_dim, normalize=True).to(device)
        self.is_metric = InceptionScore(normalize=True).to(device)
        self.kid_metric = KernelInceptionDistance(feature=feature_dim, normalize=True).to(device)

    def calculate_fid_from_activations(self,
                                     real_activations: np.ndarray,
                                     fake_activations: np.ndarray) -> float:
        """
        Calculate FID from pre-computed activations.

        Args:
            real_activations: Real image features [N, feature_dim]
            fake_activations: Generated image features [N, feature_dim]

        Returns:
            FID score

        Raises:
            ValueError: If input shapes are incompatible
        """
        # Validate inputs
        if real_activations.shape[1] != fake_activations.shape[1]:
            raise ValueError(f"Feature dimensions must match. Got {real_activations.shape[1]} and {fake_activations.shape[1]}")

        if real_activations.shape[0] < 2 or fake_activations.shape[0] < 2:
            raise ValueError("Need at least 2 samples to compute FID")

        # Convert to torch tensors
        real_features = torch.from_numpy(real_activations).float()
        fake_features = torch.from_numpy(fake_activations).float()

        # Calculate statistics
        mu1, sigma1 = real_features.mean(0), torch_cov(real_features)
        mu2, sigma2 = fake_features.mean(0), torch_cov(fake_features)

        # Calculate FID
        fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fid.item()

    def calculate_inception_score_from_logits(self,
                                            logits: np.ndarray,
                                            splits: int = 10) -> Tuple[float, float]:
        """
        Calculate Inception Score from logits.

        Args:
            logits: Logits from inception model [N, num_classes]
            splits: Number of splits for computing IS

        Returns:
            (mean_is, std_is)

        Raises:
            ValueError: If logits shape is invalid or splits > N
        """
        if len(logits.shape) != 2:
            raise ValueError(f"Logits must be 2D array, got shape {logits.shape}")

        if splits > logits.shape[0]:
            warnings.warn(f"Number of splits ({splits}) is greater than number of samples ({logits.shape[0]}). Using {logits.shape[0]} splits.")
            splits = logits.shape[0]

        logits_tensor = torch.from_numpy(logits).float()

        # Convert logits to probabilities
        probs = F.softmax(logits_tensor, dim=1)

        # Calculate IS for each split
        N = probs.shape[0]
        split_size = N // splits
        scores = []

        for i in range(splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < splits - 1 else N
            split_probs = probs[start_idx:end_idx]

            # Calculate IS for this split
            py = split_probs.mean(dim=0)
            kl_div = split_probs * (torch.log(split_probs + 1e-16) - torch.log(py + 1e-16))
            is_score = torch.exp(kl_div.sum(dim=1).mean())
            scores.append(is_score.item())

        return np.mean(scores), np.std(scores)

    def calculate_kid_from_activations(self,
                                     real_activations: np.ndarray,
                                     fake_activations: np.ndarray,
                                     max_subset_size: int = 1000) -> float:
        """
        Calculate KID from pre-computed activations.

        Args:
            real_activations: Real image features [N, feature_dim]
            fake_activations: Generated image features [N, feature_dim]
            max_subset_size: Maximum subset size for KID computation

        Returns:
            KID score

        Raises:
            ValueError: If input shapes are incompatible
        """
        # Validate inputs
        if real_activations.shape[1] != fake_activations.shape[1]:
            raise ValueError(f"Feature dimensions must match. Got {real_activations.shape[1]} and {fake_activations.shape[1]}")

        if max_subset_size < 1:
            raise ValueError(f"max_subset_size must be positive, got {max_subset_size}")

        real_features = torch.from_numpy(real_activations).float()
        fake_features = torch.from_numpy(fake_activations).float()

        # Subsample if necessary
        if real_features.shape[0] > max_subset_size:
            indices = torch.randperm(real_features.shape[0])[:max_subset_size]
            real_features = real_features[indices]

        if fake_features.shape[0] > max_subset_size:
            indices = torch.randperm(fake_features.shape[0])[:max_subset_size]
            fake_features = fake_features[indices]

        # Calculate KID using polynomial kernel
        kid_score = polynomial_mmd(real_features, fake_features)
        return kid_score.item()

    def run_classifier_fn_replacement(self,
                                    images: Union[torch.Tensor, np.ndarray],
                                    inception_model,
                                    batch_size: int = 50) -> Dict[str, np.ndarray]:
        """
        Replacement for tfgan.eval.run_classifier_fn.

        Args:
            images: Input images [N, C, H, W] or [N, H, W, C]
            inception_model: TensorFlow Hub inception model
            batch_size: Batch size for processing

        Returns:
            Dictionary with 'pool_3' and optionally 'logits' keys

        Raises:
            ValueError: If images have invalid shape
        """
        import tensorflow as tf

        # Validate input
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()

        if len(images.shape) != 4:
            raise ValueError(f"Images must be 4D array, got shape {images.shape}")

        # Determine if we need to transpose (channels first to channels last)
        if images.shape[-1] != 3 and images.shape[1] == 3:
            images = np.transpose(images, (0, 2, 3, 1))
        elif images.shape[-1] != 3:
            raise ValueError(f"Images must have 3 channels, got shape {images.shape}")

        all_pool3 = []
        all_logits = []

        # Process in batches to manage memory
        num_batches = (images.shape[0] + batch_size - 1) // batch_size

        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i + batch_size]

            # Normalize to [-1, 1] if needed
            if batch.max() > 1.0:
                batch = (batch - 127.5) / 127.5

            # Clear any existing gradients to save memory
            if hasattr(tf, 'keras'):
                tf.keras.backend.clear_session()

            batch_tf = tf.convert_to_tensor(batch, dtype=tf.float32)

            # Run through inception model
            outputs = inception_model(batch_tf)

            if isinstance(outputs, dict):
                all_pool3.append(outputs['pool_3'].numpy())
                if 'logits' in outputs:
                    all_logits.append(outputs['logits'].numpy())
            else:
                # For InceptionV3 feature vector
                all_pool3.append(outputs.numpy())

            # Log progress for large batches
            if num_batches > 10 and (i // batch_size + 1) % 10 == 0:
                logging.info(f"Processed {i + batch_size}/{images.shape[0]} images")

        result = {
            'pool_3': np.concatenate(all_pool3, axis=0)
        }

        if all_logits:
            result['logits'] = np.concatenate(all_logits, axis=0)

        return result


def torch_cov(tensor: torch.Tensor) -> torch.Tensor:
    """Calculate covariance matrix of a tensor.

    Args:
        tensor: Input tensor [N, D]

    Returns:
        Covariance matrix [D, D]
    """
    if tensor.shape[0] <= 1:
        raise ValueError("Need at least 2 samples to compute covariance")

    tensor_centered = tensor - tensor.mean(dim=0, keepdim=True)
    return torch.mm(tensor_centered.t(), tensor_centered) / (tensor.shape[0] - 1)


def calculate_frechet_distance(mu1: torch.Tensor,
                             sigma1: torch.Tensor,
                             mu2: torch.Tensor,
                             sigma2: torch.Tensor) -> torch.Tensor:
    """Calculate Frechet distance between two multivariate Gaussians.

    Args:
        mu1: Mean of first distribution [D]
        sigma1: Covariance of first distribution [D, D]
        mu2: Mean of second distribution [D]
        sigma2: Covariance of second distribution [D, D]

    Returns:
        Frechet distance
    """
    # Convert to numpy for scipy
    mu1_np = mu1.cpu().numpy()
    mu2_np = mu2.cpu().numpy()
    sigma1_np = sigma1.cpu().numpy()
    sigma2_np = sigma2.cpu().numpy()

    # Calculate squared difference of means
    diff = mu1_np - mu2_np

    # Calculate sqrt of product of covariances
    covmean, _ = linalg.sqrtm(sigma1_np.dot(sigma2_np), disp=False)

    # Handle numerical issues
    if not np.isfinite(covmean).all():
        msg = "FID calculation produces singular product; adding epsilon to diagonal."
        warnings.warn(msg)
        offset = np.eye(sigma1_np.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1_np + offset).dot(sigma2_np + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            warnings.warn(f"Imaginary component {m} in FID calculation")
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1_np + sigma2_np - 2 * covmean)
    return torch.tensor(fid)


def polynomial_mmd(X: torch.Tensor, Y: torch.Tensor, degree: int = 3, gamma: float = 1.0) -> torch.Tensor:
    """
    Calculate Maximum Mean Discrepancy with polynomial kernel (for KID).

    Args:
        X: First sample [N, D]
        Y: Second sample [M, D]
        degree: Polynomial kernel degree
        gamma: Kernel coefficient

    Returns:
        MMD estimate

    Raises:
        ValueError: If inputs have incompatible shapes or too few samples
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"Feature dimensions must match. Got {X.shape[1]} and {Y.shape[1]}")

    if X.shape[0] < 2 or Y.shape[0] < 2:
        raise ValueError("Need at least 2 samples in each set for MMD")

    # Polynomial kernel
    def poly_kernel(x, y):
        return (gamma * torch.mm(x, y.t()) + 1) ** degree

    # Calculate kernel matrices
    Kxx = poly_kernel(X, X)
    Kyy = poly_kernel(Y, Y)
    Kxy = poly_kernel(X, Y)

    # Calculate MMD
    m, n = X.shape[0], Y.shape[0]

    # Avoid division by zero
    if m == 1 or n == 1:
        raise ValueError("Need at least 2 samples in each set for MMD")

    mmd = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1))
    mmd += (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1))
    mmd -= 2 * Kxy.mean()

    return mmd


# Global instance for easy access
_metrics_instance = None

def get_metrics_instance(device: str = 'cuda') -> ModernMetrics:
    """Get global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = ModernMetrics(device=device)
    return _metrics_instance


# Convenience functions that match the old tfgan API
def frechet_classifier_distance_from_activations(real_activations: np.ndarray,
                                               fake_activations: np.ndarray) -> float:
    """Drop-in replacement for tfgan.eval.frechet_classifier_distance_from_activations."""
    metrics = get_metrics_instance()
    return metrics.calculate_fid_from_activations(real_activations, fake_activations)


def classifier_score_from_logits(logits: np.ndarray) -> float:
    """Drop-in replacement for tfgan.eval.classifier_score_from_logits."""
    metrics = get_metrics_instance()
    mean_is, _ = metrics.calculate_inception_score_from_logits(logits)
    return mean_is


def kernel_classifier_distance_from_activations(real_activations: np.ndarray,
                                              fake_activations: np.ndarray) -> float:
    """Drop-in replacement for tfgan.eval.kernel_classifier_distance_from_activations."""
    metrics = get_metrics_instance()
    return metrics.calculate_kid_from_activations(real_activations, fake_activations)


def run_classifier_fn(images: Union[torch.Tensor, np.ndarray],
                     classifier_fn,
                     num_batches: int = 1,
                     **kwargs) -> dict:
    """Drop-in replacement for tfgan.eval.run_classifier_fn."""
    metrics = get_metrics_instance()
    batch_size = images.shape[0] // num_batches if num_batches > 0 else 50
    return metrics.run_classifier_fn_replacement(images, classifier_fn, batch_size)

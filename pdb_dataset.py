"""
PDB Dataset implementation for protein distance matrices.
Compatible with torchvision dataset structure with train/val split.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

def to_tensor(x):
    """Convert numpy array to torch tensor."""
    if isinstance(x, np.ndarray):
        # Use torch.tensor to create a copy and avoid NumPy compatibility issues
        return torch.tensor(x.tolist(), dtype=torch.float32)
    return x

def scale_img(x):
    """Scale image tensor (identity function for distance matrices)."""
    return x

class TVData(Dataset):
    """torchvision dataset adapter"""

    def __init__(self, constructor, *args, cache_dir='cache_datasets', **kwargs):
        super().__init__()
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        self.dset_train = constructor(root=cache_dir, **kwargs)
        self.__N_train = len(self.dset_train)
        self.classes = getattr(self.dset_train, 'classes', [])

    def __len__(self):
        return self.__N_train

    def __getitem__(self, idx):
        x, y = self.dset_train[idx]
        x = to_tensor(x)
        return scale_img(x), y

class PDB(TVData):
    """
    PDB Dataset for protein distance matrices.
    
    Args:
        train: Whether to use training split (True) or validation split (False)
        cache_dir: Directory for caching (unused, for compatibility)
        config: Configuration object to determine data normalization range
        **kwargs: Additional arguments (unused, for compatibility)
    """
    def __init__(self, *args, cache_dir='cache_datasets', config=None, **kwargs):
        # Dataset configuration
        self.file_path = 'data/pdb/distance_matrices.npz'
        self.train_ratio = 0.8
        self.seed = 42
        self.train = kwargs.get('train', True)
        self.config = config
        
        # Load dataset
        print(f"Loading PDB dataset from {self.file_path}...")
        data = np.load(self.file_path)
        self.distance_matrices = data["distance_matrices"].astype(np.float32)
        stored_global_min = float(data["global_min"])
        stored_global_max = float(data["global_max"])
        num_samples = int(data["num_samples"])
        
        # Use actual min and robust max (99.95% percentile) for better normalization
        # This avoids outliers skewing the normalization range
        self.global_min = float(np.min(self.distance_matrices))
        self.global_max = float(np.percentile(self.distance_matrices, 99.95))
        
        print(f"Loaded {num_samples} distance matrices of shape {self.distance_matrices.shape[1:]}")
        print(f"Stored min/max: {stored_global_min:.3f}/{stored_global_max:.3f}")
        print(f"Actual min/max: {self.global_min:.3f}/{self.global_max:.3f}")
        
        # Create train/val split
        np.random.seed(self.seed)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        num_train = int(num_samples * self.train_ratio)
        train_indices = indices[:num_train]
        eval_indices = indices[num_train:]
        
        # Split matrices
        self.train_matrices = self.distance_matrices[train_indices]
        self.eval_matrices = self.distance_matrices[eval_indices]
        
        # Normalize matrices based on config settings
        self.train_matrices = self.normalize_tensor(self.train_matrices)
        self.eval_matrices = self.normalize_tensor(self.eval_matrices)
        
        print(f"Train samples: {len(self.train_matrices)}")
        print(f"Eval samples: {len(self.eval_matrices)}")
        
        # Determine target range based on config
        if self.config and hasattr(self.config.data, 'centered') and not self.config.data.centered:
            target_range = "[0, 1]"
        else:
            target_range = "[-1, 1]"
        print(f"Normalization range: [{np.min(self.train_matrices):.3f}, {np.max(self.train_matrices):.3f}] (target: {target_range})")
        
        # Set classes for compatibility
        self.classes = [0]  # Single class for unsupervised learning
    
    def normalize_tensor(self, tensor):
        """Normalize tensor based on config settings."""
        # First clip outliers, then normalize to [0, 1] using global min/max
        clipped_tensor = np.clip(tensor, self.global_min, self.global_max)
        normalized_01 = (clipped_tensor - self.global_min) / (self.global_max - self.global_min)
        
        # Then scale based on config.data.centered setting
        if self.config and hasattr(self.config.data, 'centered') and not self.config.data.centered:
            # VE models: use [0, 1] range
            return normalized_01
        else:
            # VP/SubVP models: use [-1, 1] range  
            return normalized_01 * 2 - 1

    def denormalize_tensor(self, normalized_tensor):
        """Denormalize tensor back to original distance range."""
        if self.config and hasattr(self.config.data, 'centered') and not self.config.data.centered:
            # VE models: from [0, 1] to original range
            return normalized_tensor * (self.global_max - self.global_min) + self.global_min
        else:
            # VP/SubVP models: from [-1, 1] to original range
            return (normalized_tensor + 1) / 2 * (self.global_max - self.global_min) + self.global_min

    def get_matrices(self):
        """Get the matrices for current split (train or eval)."""
        return self.train_matrices if self.train else self.eval_matrices
        
    def __len__(self):
        """Return length of current split."""
        return len(self.train_matrices) if self.train else len(self.eval_matrices)

    def __getitem__(self, idx):
        """
        Get item from dataset.
        
        Args:
            idx: Index of sample
            
        Returns:
            tuple: (distance_matrix, label) where label is always 0
        """
        matrices = self.train_matrices if self.train else self.eval_matrices
        ds_size = len(matrices)
        x = matrices[idx % ds_size]
        x = to_tensor(x)
        return x, 0  # Return label 0 for unsupervised learning

# Test the dataset
if __name__ == "__main__":
    print("=== Testing PDB Dataset ===")
    
    # Test train dataset
    train_dataset = PDB(train=True)
    print(f"Train dataset length: {len(train_dataset)}")
    
    # Test eval dataset  
    eval_dataset = PDB(train=False)
    print(f"Eval dataset length: {len(eval_dataset)}")
    
    # Test data loading
    train_sample, train_label = train_dataset[0]
    eval_sample, eval_label = eval_dataset[0]
    
    print(f"Train sample shape: {train_sample.shape}")
    print(f"Train sample type: {type(train_sample)}")
    print(f"Train sample range: [{train_sample.min():.3f}, {train_sample.max():.3f}]")
    print(f"Train label: {train_label}")
    
    print(f"Eval sample shape: {eval_sample.shape}")
    print(f"Eval sample type: {type(eval_sample)}")
    print(f"Eval sample range: [{eval_sample.min():.3f}, {eval_sample.max():.3f}]")
    print(f"Eval label: {eval_label}")
    
    # Test denormalization
    original_train = train_dataset.denormalize_tensor(train_sample.numpy())
    print(f"Denormalized train sample range: [{original_train.min():.3f}, {original_train.max():.3f}]")
    
    print("=== Dataset Test Complete ===")
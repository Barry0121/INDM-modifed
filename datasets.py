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

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import torch
import torch.utils.data
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from typing import Optional, Callable, Tuple, Any
import logging
import socket, os, natsort
import torchvision.transforms as transforms
from PIL import Image

def _data_transforms_generic(size):
  train_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
  ])

  valid_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
  ])

  return train_transform, valid_transform

class ImagenetDataSet(torch.utils.data.Dataset):
  def __init__(self, main_dir, transform):
    self.main_dir = main_dir
    self.transform = transform
    all_imgs = os.listdir(main_dir)
    self.total_imgs = natsort.natsorted(all_imgs)

  def __len__(self):
    return len(self.total_imgs)

  def __getitem__(self, idx):
    img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
    image = Image.open(img_loc).convert("RGB")
    tensor_image = self.transform(image)
    return tensor_image

class ProteinContactMapDataset(torch.utils.data.Dataset):
    """
    Dataset class for protein contact maps stored as .npy files.
    Loads all data into memory during initialization for faster training.

    Args:
        main_dir (str): Directory containing .npy files with distance/contact maps
        transform (callable, optional): Optional transform to be applied on a sample
        max_length (int, optional): Maximum sequence length to pad/crop to (same as image_size in PDB class)
        contact_threshold (float, optional): Distance threshold for contact definition (Angstroms)
        return_distance (bool): If True, return distance maps; if False, return binary contact maps
        file_extension (str): File extension to look for (default: '.npy')
        train (bool): If True, use training split; if False, use evaluation split
        train_ratio (float): Ratio of data to use for training (default: 0.8)
        seed (int): Random seed for reproducible splits (default: 42)
        minv (float): Minimum value for normalization (default: 0)
        maxv (float): Maximum value for normalization (default: 19266)
        pad_value (float): Value to use for padding (default: 0.0)
        max_samples (int, optional): Maximum number of samples to load (for testing)
    """

    def __init__(
        self,
        main_dir: str,
        transform: Optional[Callable] = None,
        max_length: Optional[int] = 512,
        contact_threshold: float = 8.0,
        return_distance: bool = True,
        file_extension: str = '.npy',
        train: bool = True,
        train_ratio: float = 0.8,
        seed: int = 42,
        minv: float = 0.0,
        maxv: float = 19266.0,
        pad_value: float = 0.0,
        max_samples: Optional[int] = None
    ):
        from tqdm import tqdm
        
        self.main_dir = main_dir
        self.transform = transform
        self.max_length = max_length
        self.contact_threshold = contact_threshold
        self.return_distance = return_distance
        self.file_extension = file_extension
        self.train = train
        self.train_ratio = train_ratio
        self.seed = seed
        self.minv = minv
        self.maxv = maxv
        self.pad_value = pad_value

        # Get all .npy files and sort them naturally
        all_files = [f for f in os.listdir(main_dir) if f.endswith(file_extension)]
        self.total_files = natsort.natsorted(all_files)

        if len(self.total_files) == 0:
            raise ValueError(f"No {file_extension} files found in {main_dir}")

        # Limit files if max_samples specified
        if max_samples:
            self.total_files = self.total_files[:max_samples]
            print(f"Limited to {len(self.total_files)} files for processing")

        # Create train/validation split
        num_samples = len(self.total_files)
        np.random.seed(self.seed)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        num_train = int(num_samples * self.train_ratio)
        train_indices = indices[:num_train]
        eval_indices = indices[num_train:]

        train_files = np.array(self.total_files)[train_indices]
        eval_files = np.array(self.total_files)[eval_indices]

        print(f"Found {len(self.total_files)} protein contact map files")
        print(f"Training samples: {len(train_files)}, Evaluation samples: {len(eval_files)}")

        # Pre-load all distance maps into memory
        print("Loading distance maps into memory...")
        self.train_matrices = []
        self.train_filenames = []
        self.eval_matrices = []
        self.eval_filenames = []

        # Load training files
        if len(train_files) > 0:
            print("Loading training files...")
            for filename in tqdm(train_files, desc="Loading train data"):
                file_path = os.path.join(main_dir, filename)
                try:
                    processed_matrix = self._load_and_process_file(file_path)
                    if processed_matrix is not None:
                        self.train_matrices.append(processed_matrix)
                        self.train_filenames.append(filename)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        # Load evaluation files
        if len(eval_files) > 0:
            print("Loading evaluation files...")
            for filename in tqdm(eval_files, desc="Loading eval data"):
                file_path = os.path.join(main_dir, filename)
                try:
                    processed_matrix = self._load_and_process_file(file_path)
                    if processed_matrix is not None:
                        self.eval_matrices.append(processed_matrix)
                        self.eval_filenames.append(filename)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        # Convert to numpy arrays for faster access
        if self.train_matrices:
            self.train_matrices = np.array(self.train_matrices, dtype=np.float32)
        if self.eval_matrices:
            self.eval_matrices = np.array(self.eval_matrices, dtype=np.float32)

        print(f"Successfully loaded:")
        print(f"  Training matrices: {len(self.train_matrices) if hasattr(self, 'train_matrices') and self.train_matrices is not None and len(self.train_matrices) > 0 else 0}")
        print(f"  Evaluation matrices: {len(self.eval_matrices) if hasattr(self, 'eval_matrices') and self.eval_matrices is not None and len(self.eval_matrices) > 0 else 0}")
        
        if hasattr(self, 'train_matrices') and self.train_matrices is not None and len(self.train_matrices) > 0:
            print(f"  Matrix shape: {self.train_matrices[0].shape}")
            
        # Calculate memory usage
        total_memory = 0
        if hasattr(self, 'train_matrices') and self.train_matrices is not None and len(self.train_matrices) > 0:
            total_memory += self.train_matrices.nbytes
        if hasattr(self, 'eval_matrices') and self.eval_matrices is not None and len(self.eval_matrices) > 0:
            total_memory += self.eval_matrices.nbytes
        print(f"  Total memory usage: {total_memory / (1024**3):.2f} GB")

    def _load_and_process_file(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load and process a single distance matrix file.
        
        Args:
            file_path: Path to .npy file
            
        Returns:
            Processed matrix of shape (max_length, max_length) or None if error
        """
        try:
            # Load the distance map
            distance_map = np.load(file_path)
            
            # Ensure it's a square matrix
            if distance_map.shape[0] != distance_map.shape[1]:
                print(f"Warning: Distance map must be square, got shape {distance_map.shape}, skipping {file_path}")
                return None
            
            # Skip if matrix is larger than max_length
            if self.max_length and (distance_map.shape[0] > self.max_length or distance_map.shape[1] > self.max_length):
                return None
            
            # Handle sequence length constraints with PDB-style padding
            if self.max_length is not None:
                distance_map = self._handle_length_constraint_pdb_style(distance_map)
            
            # Convert to contact map if requested
            if not self.return_distance:
                # Convert distance to binary contact map
                contact_map = (distance_map <= self.contact_threshold).astype(np.float32)
                # Set diagonal to 0 (residue doesn't contact itself)
                np.fill_diagonal(contact_map, 0)
                processed_map = contact_map
            else:
                processed_map = distance_map.astype(np.float32)
            
            # Apply normalization
            processed_map = self.normalize_tensor(processed_map)
            
            return processed_map
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def normalize_tensor(self, tensor):
        """Normalize tensor to [-1, 1] range using fixed min/max values"""
        return (tensor - self.minv) / (self.maxv - self.minv) * 2 - 1

    def denormalize_tensor(self, normalized_tensor):
        """Denormalize tensor from [-1, 1] back to original range"""
        return (normalized_tensor + 1) / 2 * (self.maxv - self.minv) + self.minv

    def __len__(self) -> int:
        """Return length of current split (train or eval)."""
        if self.train:
            return len(self.train_matrices) if hasattr(self, 'train_matrices') and self.train_matrices is not None else 0
        else:
            return len(self.eval_matrices) if hasattr(self, 'eval_matrices') and self.eval_matrices is not None else 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get a pre-processed protein contact/distance map from memory.

        Args:
            idx (int): Index of the sample to get

        Returns:
            tuple: (processed_map, filename) where processed_map is a torch.Tensor
        """
        # Get correct matrices and filenames based on train/eval mode
        if self.train:
            if not hasattr(self, 'train_matrices') or len(self.train_matrices) == 0:
                raise ValueError("No training matrices loaded")
            matrices = self.train_matrices
            filenames = self.train_filenames
        else:
            if not hasattr(self, 'eval_matrices') or len(self.eval_matrices) == 0:
                raise ValueError("No evaluation matrices loaded")
            matrices = self.eval_matrices
            filenames = self.eval_filenames

        # Get pre-processed matrix (already normalized)
        processed_map = matrices[idx]
        filename = filenames[idx]

        # Convert to tensor
        tensor_map = torch.from_numpy(processed_map)
        
        # Add channel dimension [1, L, L] for compatibility with training pipeline
        tensor_map = tensor_map.unsqueeze(0)

        # Apply transforms if specified
        if self.transform:
            tensor_map = self.transform(tensor_map)

        return tensor_map, filename

    def _handle_length_constraint_pdb_style(self, distance_map: np.ndarray) -> np.ndarray:
        """Handle padding/cropping with PDB-style approach: pad with pad_value, copy from top-left"""
        current_length = distance_map.shape[0]
        
        if current_length == self.max_length:
            return distance_map
        
        # Create output matrix filled with pad_value
        output_matrix = np.full((self.max_length, self.max_length), self.pad_value, dtype=np.float32)
        
        # Copy available data to top-left corner (like PDB implementation)
        copy_size = min(current_length, self.max_length)
        output_matrix[:copy_size, :copy_size] = distance_map[:copy_size, :copy_size]
        
        return output_matrix

    def _handle_length_constraint(self, distance_map: np.ndarray) -> np.ndarray:
        """Handle padding or cropping based on max_length constraint."""
        current_length = distance_map.shape[0]

        if current_length > self.max_length:
            # Crop from center
            start_idx = (current_length - self.max_length) // 2
            end_idx = start_idx + self.max_length
            distance_map = distance_map[start_idx:end_idx, start_idx:end_idx]

        elif current_length < self.max_length:
            # Pad with large distance values (or zeros for contact maps)
            pad_value = 999.0 if self.return_distance else 0.0
            pad_width = self.max_length - current_length
            pad_before = pad_width // 2
            pad_after = pad_width - pad_before

            distance_map = np.pad(
                distance_map,
                ((pad_before, pad_after), (pad_before, pad_after)),
                mode='constant',
                constant_values=pad_value
            )

        return distance_map

    def get_protein_info(self, idx: int) -> dict:
        """Get information about a specific protein."""
        filename = self.total_files[idx]
        file_path = os.path.join(self.main_dir, filename)
        distance_map = np.load(file_path)

        return {
            'filename': filename,
            'sequence_length': distance_map.shape[0],
            'file_path': file_path,
            'min_distance': np.min(distance_map[distance_map > 0]),  # Exclude diagonal
            'max_distance': np.max(distance_map),
            'mean_distance': np.mean(distance_map[distance_map > 0])
        }


# Transform functions for protein contact maps
class ProteinTransforms:
    """Collection of transforms for protein contact maps."""

    @staticmethod
    def normalize_distances(max_distance: float = 50.0):
        """Normalize distances to [0, 1] range."""
        def transform(tensor_map):
            return torch.clamp(tensor_map / max_distance, 0, 1)
        return transform

    @staticmethod
    def log_transform(epsilon: float = 1e-6):
        """Apply log transformation to distances."""
        def transform(tensor_map):
            return torch.log(tensor_map + epsilon)
        return transform

    @staticmethod
    def add_gaussian_noise(std: float = 0.1):
        """Add Gaussian noise for data augmentation."""
        def transform(tensor_map):
            noise = torch.randn_like(tensor_map) * std
            return tensor_map + noise
        return transform

    @staticmethod
    def symmetrize():
        """Ensure the matrix is symmetric (should already be for distance maps)."""
        def transform(tensor_map):
            return (tensor_map + tensor_map.transpose(-2, -1)) / 2
        return transform

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  #h = tf.round(h * ratio, tf.int32)
  #w = tf.round(w * ratio, tf.int32)
  h = int(h * ratio)
  w = int(w * ratio)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_batch(config, data_iter, data):
  try:
    batch = get_batch_(config, next(data_iter))
  except:
    logging.info('New Epoch Start')
    data_iter = iter(data)
    batch = get_batch_(config, next(data_iter))
  return batch, data_iter

def get_batch_(config, batch):
  if isinstance(batch, torch.ByteTensor):
    batch = batch#.to(config.device).float().permute(0, 3, 1, 2) / 255.
  else:
    if config.data.dataset in ['STL10', 'CIFAR100']:
      batch = batch[0]#.to(config.device)
    elif config.data.dataset in ['IMAGENET32', 'IMAGENET64']:
      batch = batch#.to(config.device)
    elif config.data.dataset == 'PROTEIN_CONTACT_MAP':
      # For protein contact maps, batch is a tuple (tensors, filenames) from DataLoader
      batch = batch[0]  # Extract just the tensor part, ignore filenames
    else:
      batch = torch.from_numpy(batch['image']._numpy()).float()#.to(config.device).float()
      batch = batch.permute(0, 3, 1, 2)
  assert batch.shape == (batch.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size)

  return batch.to(config.device)

def check_dataset(config, train_ds, eval_ds):
  if config.data.dataset in ['IMAGENET32', 'IMAGENET64']:
    num_train_data = len(train_ds.dataset)
    num_eval_data = len(eval_ds.dataset)
    assert num_train_data == config.training.num_train_data and num_eval_data == config.eval.num_test_data
  elif config.data.dataset == 'PROTEIN_CONTACT_MAP':
    # For protein dataset, update config with actual dataset sizes
    num_train_data = len(train_ds.dataset)
    num_eval_data = len(eval_ds.dataset)
    print(f"Updating config with actual dataset sizes:")
    print(f"  Training data: {num_train_data}")
    print(f"  Evaluation data: {num_eval_data}")
    config.training.num_train_data = num_train_data
    config.eval.num_test_data = num_eval_data

def get_dataset(config):
  if config.data.dataset in ['IMAGENET32', 'PROTEIN_CONTACT_MAP']:
    train_ds, eval_ds = get_dataset_from_torch(config)
  else:
    train_ds = get_dataset_from_tf(config, evaluation=False)
    eval_ds = get_dataset_from_tf(config, evaluation=True)
  check_dataset(config, train_ds, eval_ds)
  return train_ds, eval_ds

def get_dataset_from_torch(config):
  if config.data.dataset == 'IMAGENET32':
    train_data_path = os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) + \
                      '/data/IMAGENET32/small/train_32x32/'
    eval_data_path = os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) + \
                     '/data/IMAGENET32/small/valid_32x32/'
    train_transform, val_transform = _data_transforms_generic(config.data.image_size)
    train_data = ImagenetDataSet(train_data_path, train_transform)
    eval_data = ImagenetDataSet(eval_data_path, val_transform)

    train_ds = torch.utils.data.DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True,
                                           pin_memory=True, num_workers=8, drop_last=True)
    eval_ds = torch.utils.data.DataLoader(eval_data, batch_size=config.eval.batch_size, shuffle=False, pin_memory=True,
                                          num_workers=1, drop_last=False)
                                          
  elif config.data.dataset == 'PROTEIN_CONTACT_MAP':
    # Get dataset parameters from config
    pdb_dir = getattr(config.data, 'pdb_dir', 'pdb')
    max_samples = getattr(config.data, 'max_samples', None)
    
    # Calculate global min/max if provided, otherwise use defaults
    global_min = getattr(config.data, 'global_min', 0.0)
    global_max = getattr(config.data, 'global_max', 50000.0)
    
    print(f"Loading protein dataset from: {pdb_dir}")
    print(f"Using normalization range: [{global_min:.3f}, {global_max:.3f}]")
    
    # Create train dataset
    train_data = ProteinContactMapDataset(
        main_dir=pdb_dir,
        max_length=config.data.image_size,
        train=True,
        train_ratio=0.8,
        seed=42,
        return_distance=True,
        minv=global_min,
        maxv=global_max,
        pad_value=0.0,
        max_samples=max_samples
    )
    
    # Create eval dataset  
    eval_data = ProteinContactMapDataset(
        main_dir=pdb_dir,
        max_length=config.data.image_size,
        train=False,
        train_ratio=0.8,
        seed=42,
        return_distance=True,
        minv=global_min,
        maxv=global_max,
        pad_value=0.0,
        max_samples=max_samples
    )
    
    print(f"Train dataset size: {len(train_data)}")
    print(f"Eval dataset size: {len(eval_data)}")
    
    # Create data loaders
    train_ds = torch.utils.data.DataLoader(
        train_data, 
        batch_size=config.training.batch_size, 
        shuffle=True,
        pin_memory=True, 
        num_workers=2,  # Reduced since data is pre-loaded
        drop_last=True
    )
    eval_ds = torch.utils.data.DataLoader(
        eval_data, 
        batch_size=config.eval.batch_size, 
        shuffle=False, 
        pin_memory=True,
        num_workers=1, 
        drop_last=False
    )
    
  return train_ds, eval_ds

def get_dataset_from_tf(config, evaluation=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
  if batch_size % torch.cuda.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by the number of devices')

  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1

  # Create dataset builders for each dataset.
  if config.data.dataset == 'CIFAR10':
    dataset_builder = tfds.builder('cifar10', data_dir=os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) + f'/data/')
  #dataset_builder = tfds.builder('cifar10')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'SVHN':
    dataset_builder = tfds.builder('svhn_cropped')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'CELEBA':
    dataset_builder = tfds.builder('celeb_a', data_dir=os.path.dirname(
      os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) + f'/data/')
    train_split_name = 'train'
    # eval_split_name = 'validation'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = central_crop(img, 140)
      img = resize_small(img, config.data.image_size)
      return img

  elif config.data.dataset == 'LSUN':
    dataset_builder = tfds.builder(f'lsun/{config.data.category}')
    train_split_name = 'train'
    eval_split_name = 'validation'

    if config.data.image_size == 128:
      def resize_op(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = resize_small(img, config.data.image_size)
        img = central_crop(img, config.data.image_size)
        return img

    else:
      def resize_op(img):
        img = crop_resize(img, config.data.image_size)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

  elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
    dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
    train_split_name = eval_split_name = 'train'

  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  # Customize preprocess functions for each dataset.
  if config.data.dataset in ['FFHQ', 'CelebAHQ']:
    def preprocess_fn(d):
      sample = tf.io.parse_single_example(d, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
      data = tf.io.decode_raw(sample['data'], tf.uint8)
      data = tf.reshape(data, sample['shape'])
      data = tf.transpose(data, (1, 2, 0))
      img = tf.image.convert_image_dtype(data, tf.float32)
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      return dict(image=img, label=None)

  else:
    def preprocess_fn(d):
      """Basic preprocessing function scales data to [0, 1) and randomly flips."""
      img = resize_op(d['image'])
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)

      return dict(image=img, label=d.get('label', None))

  def create_dataset(dataset_builder, split, batch_size):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
        split=split, shuffle_files=True, read_config=read_config)
    else:
      ds = dataset_builder.with_options(dataset_options)
    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  if evaluation:
    data = create_dataset(dataset_builder, eval_split_name, config.eval.batch_size)
  else:
    data = create_dataset(dataset_builder, train_split_name, config.training.batch_size)
  return data

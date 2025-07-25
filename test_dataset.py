"""
Simple test of the PDB dataset without torch dependencies.
"""
import numpy as np

# Load and test the dataset
print("=== Testing PDB Dataset ===")

file_path = 'data/pdb/distance_matrices.npz'
data = np.load(file_path)
distance_matrices = data["distance_matrices"].astype(np.float32)
global_min = float(data["global_min"])
global_max = float(data["global_max"])
num_samples = int(data["num_samples"])

print(f"Loaded {num_samples} distance matrices of shape {distance_matrices.shape[1:]}")
print(f"Global min/max: {global_min:.3f}/{global_max:.3f}")

# Test train/val split
train_ratio = 0.8
seed = 42

np.random.seed(seed)
indices = np.arange(num_samples)
np.random.shuffle(indices)
num_train = int(num_samples * train_ratio)
train_indices = indices[:num_train]
eval_indices = indices[num_train:]

train_matrices = distance_matrices[train_indices]
eval_matrices = distance_matrices[eval_indices]

print(f"Train samples: {len(train_matrices)}")
print(f"Eval samples: {len(eval_matrices)}")

# Test normalization
def normalize_tensor(tensor, global_min, global_max):
    return (tensor - global_min) / (global_max - global_min) * 2 - 1

train_normalized = normalize_tensor(train_matrices, global_min, global_max)
eval_normalized = normalize_tensor(eval_matrices, global_min, global_max)

print(f"Train normalized range: [{np.min(train_normalized):.3f}, {np.max(train_normalized):.3f}]")
print(f"Eval normalized range: [{np.min(eval_normalized):.3f}, {np.max(eval_normalized):.3f}]")

# Test denormalization
def denormalize_tensor(normalized_tensor, global_min, global_max):
    return (normalized_tensor + 1) / 2 * (global_max - global_min) + global_min

train_denorm = denormalize_tensor(train_normalized, global_min, global_max)
print(f"Train denormalized range: [{np.min(train_denorm):.3f}, {np.max(train_denorm):.3f}]")

# Check if denormalization works correctly
original_sample = train_matrices[0]
normalized_sample = normalize_tensor(original_sample, global_min, global_max)
denorm_sample = denormalize_tensor(normalized_sample, global_min, global_max)

print(f"Original vs denormalized difference: {np.max(np.abs(original_sample - denorm_sample)):.6f}")

print("=== Dataset Test Complete ===")
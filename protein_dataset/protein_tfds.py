# protein_tfds.py
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import os

class ProteinDistanceMatrices(tfds.core.GeneratorBasedBuilder):
    """Protein distance matrices dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="Protein distance matrices dataset with 582,681 samples of 32x32 matrices",
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Tensor(
                    shape=(32, 32, 1),
                    dtype=tf.float32,
                ),
                'label': tfds.features.ClassLabel(names=['protein']),
            }),
            supervised_keys=('image', 'label'),
            homepage='your-project-url',
        )

    def _split_generators(self, dl_manager):
        # Path to your NPZ file
        npz_path = os.path.join('/'.join(os.path.dirname(__file__).split('/')[:-1]), 'data', 'pdb', 'distance_matrices.npz')

        # Return proper SplitGenerator objects
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    'npz_path': npz_path,
                    'split': 'train',
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    'npz_path': npz_path,
                    'split': 'test',
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    'npz_path': npz_path,
                    'split': 'validation',
                },
            ),
        ]

    def _generate_examples(self, npz_path, split):
        # Load the full NPZ file
        data = np.load(npz_path)
        matrices = data['distance_matrices']

        # Split the data (adjust ratios as needed)
        total_samples = matrices.shape[0]  # 582,681

        if split == 'train':
            start_idx = 0
            end_idx = int(0.8 * total_samples)  # 80% for training
        elif split == 'validation':
            start_idx = int(0.8 * total_samples)
            end_idx = int(0.9 * total_samples)   # 10% for validation
        else:  # test
            start_idx = int(0.9 * total_samples)
            end_idx = total_samples              # 10% for testing

        for idx in range(start_idx, end_idx):
            matrix = matrices[idx]  # Shape: (32, 32)

            # Normalize if needed (0-1 range)
            matrix_normalized = (matrix - matrix.min()) / (matrix.max() - matrix.min() + 1e-8)

            # Add channel dimension: (32, 32) -> (32, 32, 1)
            matrix_with_channel = matrix_normalized.reshape(32, 32, 1)

            yield f"protein_{idx}", {
                'image': matrix_with_channel.astype(np.float32),
                'label': 0,  # Single class
            }
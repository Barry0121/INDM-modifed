# __init__.py
import tensorflow_datasets as tfds
from .protein_tfds import ProteinDistanceMatrices

# Register the dataset
tfds.core.lazy_imports_lib.LazyImporter(
    "protein_distance_matrices", lambda: ProteinDistanceMatrices
)
#!/bin/bash

# Fix for CUDA architecture compatibility issue with RTX 4090 (compute capability 8.9)
# This script sets the TORCH_CUDA_ARCH_LIST environment variable to include
# all common CUDA architectures including 8.9 for Ada Lovelace GPUs

export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9"

echo "Set TORCH_CUDA_ARCH_LIST to: $TORCH_CUDA_ARCH_LIST"
echo "Running main.py with CUDA architecture fix..."

# Execute the main script with all passed arguments
python main.py "$@"
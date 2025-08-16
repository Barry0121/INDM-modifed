#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import glob

def load_samples(output_folder, num_samples=5):
    """
    Load sample files from the output folder.
    
    Args:
        output_folder: Path to folder containing sample .npz files
        num_samples: Number of sample files to load
        
    Returns:
        List of loaded sample arrays
    """
    sample_files = sorted(glob.glob(os.path.join(output_folder, "samples_*.npz")))
    
    if not sample_files:
        raise ValueError(f"No sample files found in {output_folder}")
    
    # Load the first num_samples files
    samples = []
    for i, sample_file in enumerate(sample_files[:num_samples]):
        try:
            data = np.load(sample_file)
            if 'samples' in data:
                samples.append(data['samples'])
                print(f"Loaded {sample_file}: shape {data['samples'].shape}")
            else:
                print(f"Warning: No 'samples' key in {sample_file}, keys: {list(data.keys())}")
        except Exception as e:
            print(f"Error loading {sample_file}: {e}")
            
    return samples

def visualize_protein_samples(samples, output_folder, num_display=8):
    """
    Create visualization for protein samples.
    
    Args:
        samples: List of sample arrays
        output_folder: Output folder to save plots
        num_display: Number of individual samples to display from each batch
    """
    num_batches = len(samples)
    
    # Create a grid layout: rows for different sample files, columns for individual samples
    fig, axes = plt.subplots(num_batches, num_display, figsize=(16, 3*num_batches))
    
    if num_batches == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Generated Protein Samples from {os.path.basename(output_folder)}', 
                 fontsize=14, fontweight='bold')
    
    for batch_idx, sample_batch in enumerate(samples):
        # Get sample info
        batch_size, height, width, channels = sample_batch.shape
        
        # Select random samples from the batch
        sample_indices = np.random.choice(batch_size, min(num_display, batch_size), replace=False)
        
        for col_idx, sample_idx in enumerate(sample_indices):
            ax = axes[batch_idx, col_idx]
            
            # Extract the sample
            sample = sample_batch[sample_idx]
            
            # Handle different channel configurations
            if channels == 1:
                # Grayscale - squeeze the channel dimension
                img = sample.squeeze()
                cmap = 'viridis'
            elif channels == 3:
                # RGB - keep as is
                img = sample
                cmap = None
            else:
                # Multi-channel - show first channel
                img = sample[:, :, 0]
                cmap = 'viridis'
            
            # Plot the sample
            im = ax.imshow(img, cmap=cmap, aspect='auto')
            ax.set_title(f'Batch {batch_idx}, Sample {sample_idx}', fontsize=10)
            ax.axis('off')
            
            # Add colorbar for the first sample in each row
            if col_idx == 0:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Remove unused subplots
    for batch_idx in range(num_batches):
        for col_idx in range(min(num_display, len(sample_indices)), num_display):
            if num_batches > 1:
                axes[batch_idx, col_idx].remove()
            else:
                axes[col_idx].remove()
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(output_folder, 'sample_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Sample visualization saved to: {save_path}")
    
    # plt.show()  # Disabled for non-interactive mode

def create_sample_statistics_plot(samples, output_folder):
    """
    Create statistical plots for the samples.
    
    Args:
        samples: List of sample arrays
        output_folder: Output folder to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Sample Statistics from {os.path.basename(output_folder)}', 
                 fontsize=14, fontweight='bold')
    
    all_values = []
    batch_stats = []
    
    for i, sample_batch in enumerate(samples):
        values = sample_batch.flatten()
        all_values.append(values)
        
        stats = {
            'batch': i,
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'shape': sample_batch.shape
        }
        batch_stats.append(stats)
        
        print(f"Batch {i} stats: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
              f"range=[{stats['min']:.3f}, {stats['max']:.3f}], shape={stats['shape']}")
    
    # Combine all values
    all_values_combined = np.concatenate(all_values)
    
    # Plot 1: Histogram of all values
    axes[0, 0].hist(all_values_combined, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Distribution of All Sample Values')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Distribution comparison across batches
    for i, values in enumerate(all_values):
        axes[0, 1].hist(values, bins=30, alpha=0.6, label=f'Batch {i}')
    axes[0, 1].set_title('Value Distribution by Batch')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Statistics by batch
    batch_indices = [s['batch'] for s in batch_stats]
    means = [s['mean'] for s in batch_stats]
    stds = [s['std'] for s in batch_stats]
    
    axes[1, 0].errorbar(batch_indices, means, yerr=stds, fmt='o-', capsize=5, markersize=8)
    axes[1, 0].set_title('Mean ± Std by Batch')
    axes[1, 0].set_xlabel('Batch Index')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Range (min/max) by batch
    mins = [s['min'] for s in batch_stats]
    maxs = [s['max'] for s in batch_stats]
    
    axes[1, 1].plot(batch_indices, mins, 'o-', label='Min', markersize=6)
    axes[1, 1].plot(batch_indices, maxs, 's-', label='Max', markersize=6)
    axes[1, 1].set_title('Value Range by Batch')
    axes[1, 1].set_xlabel('Batch Index')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(output_folder, 'sample_statistics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Sample statistics saved to: {save_path}")
    
    # plt.show()  # Disabled for non-interactive mode

def main():
    parser = argparse.ArgumentParser(description='Visualize generated samples from output folder')
    parser.add_argument('output_folder', 
                       help='Path to output folder containing sample .npz files')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of sample files to load (default: 5)')
    parser.add_argument('--num_display', type=int, default=8,
                       help='Number of individual samples to display per batch (default: 8)')
    parser.add_argument('--stats_only', action='store_true',
                       help='Only generate statistics plots, skip sample visualization')
    
    args = parser.parse_args()
    
    # Validate output folder
    if not os.path.exists(args.output_folder):
        print(f"Error: Output folder {args.output_folder} does not exist")
        sys.exit(1)
    
    print(f"Loading samples from: {args.output_folder}")
    
    try:
        # Load samples
        samples = load_samples(args.output_folder, args.num_samples)
        
        if not samples:
            print("No samples could be loaded")
            sys.exit(1)
        
        print(f"Successfully loaded {len(samples)} sample batches")
        
        # Create visualizations
        if not args.stats_only:
            print("Creating sample visualization...")
            visualize_protein_samples(samples, args.output_folder, args.num_display)
        
        print("Creating statistics plots...")
        create_sample_statistics_plot(samples, args.output_folder)
        
        print("Visualization complete!")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
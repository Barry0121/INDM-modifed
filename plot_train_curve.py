import matplotlib.pyplot as plt
import numpy as np
import re
from typing import List, Dict, Tuple

def parse_training_log(log_file_path: str) -> List[Dict]:
    """
    Parse training log file and extract loss metrics.
    Handles both flow model logs and noflow model logs.

    Args:
        log_file_path: Path to the log file

    Returns:
        List of dictionaries containing step and loss information
    """
    with open(log_file_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    step_lines = [line for line in lines if 'step:' in line and ('loss mean:' in line or 'training loss mean:' in line)]

    data = []

    for line in step_lines:
        # Extract step number
        step_match = re.search(r'step: (\d+)', line)
        if not step_match:
            continue

        step = int(step_match.group(1))

        # Check if this is a noflow log (simpler format)
        if 'training loss mean:' in line:
            # Noflow format: step: X, training loss mean: Y, training loss std: Z
            training_loss_match = re.search(r'training loss mean: ([\d\.e\+\-]+)', line)
            training_std_match = re.search(r'training loss std: ([\d\.e\+\-]+)', line)

            if training_loss_match:
                data.append({
                    'step': step,
                    'total_loss': float(training_loss_match.group(1)),
                    'score_loss': float(training_loss_match.group(1)),  # Same as total for noflow
                    'flow_loss': 0.0,  # No flow loss in noflow models
                    'logp_mean': 0.0,  # No logp in noflow models
                    'training_std': float(training_std_match.group(1)) if training_std_match else 0.0,
                    'is_noflow': True
                })
        else:
            # Flow format: step: X, loss mean: Y, score loss mean: Z, flow loss mean: W, logp mean: V
            loss_mean_match = re.search(r'loss mean: ([\d\.e\+\-]+)', line)
            score_loss_match = re.search(r'score loss mean: ([\d\.e\+\-]+)', line)
            flow_loss_match = re.search(r'flow loss mean: ([\d\.e\+\-]+)', line)
            logp_mean_match = re.search(r'logp mean: ([\d\.e\+\-]+)', line)

            if all([loss_mean_match, score_loss_match, flow_loss_match, logp_mean_match]):
                data.append({
                    'step': step,
                    'total_loss': float(loss_mean_match.group(1)),
                    'score_loss': float(score_loss_match.group(1)),
                    'flow_loss': float(flow_loss_match.group(1)),
                    'logp_mean': float(logp_mean_match.group(1)),
                    'training_std': 0.0,
                    'is_noflow': False
                })

    return data

def plot_training_losses(log_file_path: str, save_path: str = None, figsize: Tuple[int, int] = (15, 10)):
    """
    Create comprehensive training loss plots from log file.
    Handles both flow and noflow model logs.

    Args:
        log_file_path: Path to the training log file
        save_path: Optional path to save the plot
        figsize: Figure size tuple (width, height)
    """
    # Parse the data
    data = parse_training_log(log_file_path)

    if not data:
        print("No training data found in log file!")
        return

    # Convert to arrays for easier plotting
    steps = np.array([d['step'] for d in data])
    total_losses = np.array([d['total_loss'] for d in data])
    score_losses = np.array([d['score_loss'] for d in data])
    flow_losses = np.array([d['flow_loss'] for d in data])
    logp_means = np.array([d['logp_mean'] for d in data])
    training_stds = np.array([d['training_std'] for d in data])

    # Check if this is a noflow model
    is_noflow = data[0]['is_noflow'] if data else False

    # Handle training restarts (when step counter resets)
    restart_indices = []
    cumulative_steps = steps.copy()

    for i in range(1, len(steps)):
        if steps[i] < steps[i-1]:  # Step counter reset
            restart_indices.append(i)
            # Adjust all subsequent steps
            offset = cumulative_steps[i-1] + 100  # Add small gap
            cumulative_steps[i:] += offset

    print(f"Found {len(restart_indices)} training restarts")
    print(f"Total training steps: {len(data)}")
    print(f"Step range: {steps.min()} - {steps.max()}")
    print(f"Final total loss: {total_losses[-1]:.2e}")
    print(f"Model type: {'NoFlow' if is_noflow else 'Flow'}")

    # Create appropriate subplot layout based on model type
    if is_noflow:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('NoFlow Training Loss Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Total Loss (Training Loss)
        axes[0, 0].plot(cumulative_steps, total_losses, 'b-', linewidth=1.5, alpha=0.8)
        axes[0, 0].set_title('Training Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')

        # Plot 2: Training Loss with Error Bars (std)
        axes[0, 1].errorbar(cumulative_steps[::10], total_losses[::10], yerr=training_stds[::10],
                           fmt='o-', linewidth=1.5, alpha=0.8, markersize=3, capsize=3)
        axes[0, 1].set_title('Training Loss ± Std', fontweight='bold')
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')

        # Plot 3: Training Standard Deviation
        axes[1, 0].plot(cumulative_steps, training_stds, 'orange', linewidth=1.5, alpha=0.8)
        axes[1, 0].set_title('Training Loss Std', fontweight='bold')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Loss Std')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Loss smoothed with moving average
        window_size = min(50, len(total_losses) // 10)
        if window_size > 1:
            smoothed_loss = np.convolve(total_losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_steps = cumulative_steps[window_size-1:]
            axes[1, 1].plot(cumulative_steps, total_losses, 'lightblue', alpha=0.5, label='Raw')
            axes[1, 1].plot(smoothed_steps, smoothed_loss, 'blue', linewidth=2, label=f'Smoothed (n={window_size})')
            axes[1, 1].legend()
        else:
            axes[1, 1].plot(cumulative_steps, total_losses, 'b-', linewidth=1.5, alpha=0.8)
        axes[1, 1].set_title('Smoothed Training Loss', fontweight='bold')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')

    else:
        # Flow model - original 4-panel layout
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Flow Training Loss Analysis', fontsize=16, fontweight='bold')

        # Plot 1: Total Loss
        axes[0, 0].plot(cumulative_steps, total_losses, 'b-', linewidth=1.5, alpha=0.8)
        axes[0, 0].set_title('Total Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')

        # Plot 2: Score Loss
        axes[0, 1].plot(cumulative_steps, score_losses, 'g-', linewidth=1.5, alpha=0.8)
        axes[0, 1].set_title('Score Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Score Loss')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')

        # Plot 3: Flow Loss
        axes[1, 0].plot(cumulative_steps, flow_losses, 'r-', linewidth=1.5, alpha=0.8)
        axes[1, 0].set_title('Flow Loss', fontweight='bold')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Flow Loss')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: LogP Mean
        axes[1, 1].plot(cumulative_steps, logp_means, 'purple', linewidth=1.5, alpha=0.8)
        axes[1, 1].set_title('LogP Mean', fontweight='bold')
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('LogP Mean')
        axes[1, 1].grid(True, alpha=0.3)

    # Add restart markers to all subplots
    for restart_idx in restart_indices:
        for ax in axes.flat:
            ax.axvline(x=cumulative_steps[restart_idx], color='red',
                      linestyle='--', alpha=0.5)

    # Add restart markers and legend
    for restart_idx in restart_indices:
        axes[0, 0].axvline(x=cumulative_steps[restart_idx], color='red',
                          linestyle='--', alpha=0.7, label='Restart' if restart_idx == restart_indices[0] else "")

    # Add legend only to the first plot
    if restart_indices:
        axes[0, 0].legend()

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()

    # Print statistics
    print("\n" + "="*50)
    print(f"{'NOFLOW' if is_noflow else 'FLOW'} TRAINING STATISTICS")
    print("="*50)

    print(f"Initial Total Loss: {total_losses[0]:.2e}")
    print(f"Final Total Loss: {total_losses[-1]:.2e}")
    print(f"Loss Reduction: {((total_losses[0] - total_losses[-1]) / total_losses[0] * 100):.1f}%")

    if is_noflow:
        print(f"\nInitial Training Std: {training_stds[0]:.2e}")
        print(f"Final Training Std: {training_stds[-1]:.2e}")
        print(f"Mean Training Std: {training_stds.mean():.2e}")
    else:
        print(f"\nInitial Score Loss: {score_losses[0]:.2e}")
        print(f"Final Score Loss: {score_losses[-1]:.2e}")
        print(f"\nInitial Flow Loss: {flow_losses[0]:.2e}")
        print(f"Final Flow Loss: {flow_losses[-1]:.2e}")
        print(f"\nLogP Mean Range: {logp_means.min():.2e} - {logp_means.max():.2e}")
        print(f"LogP Mean Std: {logp_means.std():.2e}")

    # Find best (lowest) losses
    min_total_idx = np.argmin(total_losses)
    print(f"\nBest Total Loss: {total_losses[min_total_idx]:.2e} at step {steps[min_total_idx]}")

    if not is_noflow:
        min_score_idx = np.argmin(score_losses)
        print(f"Best Score Loss: {score_losses[min_score_idx]:.2e} at step {steps[min_score_idx]}")

def plot_smoothed_losses(log_file_path: str, window_size: int = 50, save_path: str = None):
    """
    Plot smoothed version of losses for clearer trend visualization.
    Handles both flow and noflow model logs.

    Args:
        log_file_path: Path to the training log file
        window_size: Size of moving average window
        save_path: Optional path to save the plot
    """
    data = parse_training_log(log_file_path)

    if not data:
        print("No training data found!")
        return

    steps = np.array([d['step'] for d in data])
    total_losses = np.array([d['total_loss'] for d in data])
    score_losses = np.array([d['score_loss'] for d in data])
    training_stds = np.array([d['training_std'] for d in data])

    # Check if this is a noflow model
    is_noflow = data[0]['is_noflow'] if data else False

    # Handle restarts
    cumulative_steps = steps.copy()
    for i in range(1, len(steps)):
        if steps[i] < steps[i-1]:
            offset = cumulative_steps[i-1] + 100
            cumulative_steps[i:] += offset

    # Apply moving average smoothing
    def moving_average(data, window):
        return np.convolve(data, np.ones(window)/window, mode='valid')

    if len(total_losses) >= window_size:
        smooth_steps = cumulative_steps[window_size-1:]
        smooth_total = moving_average(total_losses, window_size)

        if is_noflow:
            # NoFlow: plot training loss and training std
            smooth_std = moving_average(training_stds, window_size)

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.plot(cumulative_steps, total_losses, 'lightblue', alpha=0.5, label='Raw')
            plt.plot(smooth_steps, smooth_total, 'blue', linewidth=2, label=f'Smoothed (window={window_size})')
            plt.title('Training Loss (Smoothed)', fontweight='bold')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(cumulative_steps, training_stds, 'lightcoral', alpha=0.5, label='Raw')
            plt.plot(smooth_steps, smooth_std, 'red', linewidth=2, label=f'Smoothed (window={window_size})')
            plt.title('Training Loss Std (Smoothed)', fontweight='bold')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss Std')
            plt.grid(True, alpha=0.3)
            plt.legend()

        else:
            # Flow: plot total loss and score loss (original behavior)
            smooth_score = moving_average(score_losses, window_size)

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.plot(cumulative_steps, total_losses, 'lightblue', alpha=0.5, label='Raw')
            plt.plot(smooth_steps, smooth_total, 'blue', linewidth=2, label=f'Smoothed (window={window_size})')
            plt.title('Total Loss (Smoothed)', fontweight='bold')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(cumulative_steps, score_losses, 'lightcoral', alpha=0.5, label='Raw')
            plt.plot(smooth_steps, smooth_score, 'red', linewidth=2, label=f'Smoothed (window={window_size})')
            plt.title('Score Loss (Smoothed)', fontweight='bold')
            plt.xlabel('Training Steps')
            plt.ylabel('Score Loss')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

# Example usage:
if __name__ == "__main__":
    # Replace 'paste.txt' with your actual log file path
    log_file = '/home/ubuntu/INDM/INDM/outputs/subvp/PROTEIN/ode-noflow/stdout.txt'

    # Create the main plot
    plot_training_losses(log_file, save_path='training_losses.png')

    # Create smoothed version
    plot_smoothed_losses(log_file, window_size=50, save_path='training_losses_smoothed.png')
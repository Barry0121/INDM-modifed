#!/bin/bash

echo "Starting custom diffusion model training on GPUs 6 and 7..."
echo "Date: $(date)"
echo "Configuration 1 (GPU 6): 60,000 samples with default flow output"
echo "Configuration 2 (GPU 7): Full dataset with weighted flow output (1e-3)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Create custom configs directory if it doesn't exist
mkdir -p configs/custom/PROTEIN

echo "Custom configuration files already created:"
echo "  - configs/custom/PROTEIN/indm_fid_ode_60k.py (60k samples, default flow)"
echo "  - configs/custom/PROTEIN/indm_fid_ode_weighted_flow.py (full dataset, weighted flow output)"
echo ""

# Start custom training jobs in parallel on GPUs 6 and 7
echo "=== Starting custom training configurations in parallel ==="

# Configuration 1: 60k samples with default flow output on GPU 6
echo "Starting custom training (60k samples, default flow) with ODE solver on GPU 6..."
CUDA_VISIBLE_DEVICES=6 python main.py --mode train --config configs/custom/PROTEIN/indm_fid_ode_60k.py --workdir outputs/custom/PROTEIN/60k_default_flow/ > logs/custom_60k_default_flow_log.txt 2> logs/custom_60k_default_flow_err.txt &
PID_60K_DEFAULT=$!

# Configuration 2: Full dataset with weighted flow output on GPU 7
echo "Starting custom training (full dataset, weighted flow output) with ODE solver on GPU 7..."
CUDA_VISIBLE_DEVICES=7 python main.py --mode train --config configs/custom/PROTEIN/indm_fid_ode_weighted_flow.py --workdir outputs/custom/PROTEIN/full_weighted_flow/ > logs/custom_full_weighted_flow_log.txt 2> logs/custom_full_weighted_flow_err.txt &
PID_FULL_WEIGHTED=$!

echo ""
echo "Custom training jobs started:"
echo "GPU 6: 60k samples + default flow output - PID: $PID_60K_DEFAULT"
echo "GPU 7: Full dataset + weighted flow output (1e-3) - PID: $PID_FULL_WEIGHTED"
echo ""
echo "Waiting for jobs to complete..."

# Wait for all background jobs to complete
wait $PID_60K_DEFAULT
echo "Custom training (60k samples, default flow) completed. Log: logs/custom_60k_default_flow_log.txt, Errors: logs/custom_60k_default_flow_err.txt"

wait $PID_FULL_WEIGHTED
echo "Custom training (full dataset, weighted flow) completed. Log: logs/custom_full_weighted_flow_log.txt, Errors: logs/custom_full_weighted_flow_err.txt"

echo ""
echo "=== Custom training configurations completed ==="
echo "End time: $(date)"
echo ""
echo "Summary of custom configurations:"
echo "1. GPU 6: 60k samples + default flow output: outputs/custom/PROTEIN/60k_default_flow/"
echo "2. GPU 7: Full dataset + weighted flow output (1e-3): outputs/custom/PROTEIN/full_weighted_flow/"
echo ""
echo "Key modifications applied:"
echo "  - Configuration 1: Dataset limited to 60,000 training samples"
echo "  - Configuration 2: Flow output weighted by 1e-3 to reduce value spread"
echo ""
echo "Custom training logs are saved in the logs/ directory:"
echo "  - Standard output: custom_*_log.txt files"
echo "  - Error output: custom_*_err.txt files"
echo "GPUs 0-5 remain available for other tasks."
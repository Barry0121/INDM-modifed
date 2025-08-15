#!/bin/bash


echo "Starting PROTEIN dataset SubVP no-flow model training..."
echo "Date: $(date)"
echo "Running 2 SubVP no-flow configurations: ODE and PC on GPUs 6-7"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start 2 SubVP no-flow training jobs in parallel
echo "=== Starting SubVP no-flow configurations in parallel ==="

echo "Starting SubVP no-flow with ODE solver on GPU 6..."
CUDA_VISIBLE_DEVICES=6 python main.py --mode train --config configs/subvp/PROTEIN/indm_fid_ode_noflow.py --workdir outputs/subvp/PROTEIN/ode-noflow/ > logs/protein_subvp_noflow_ode_log.txt 2> logs/protein_subvp_noflow_ode_err.txt &
PID_SUBVP_ODE=$!

echo "Starting SubVP no-flow with PC solver on GPU 7..."
CUDA_VISIBLE_DEVICES=7 python main.py --mode train --config configs/subvp/PROTEIN/indm_fid_pc_noflow.py --workdir outputs/subvp/PROTEIN/pc-noflow/ > logs/protein_subvp_noflow_pc_log.txt 2> logs/protein_subvp_noflow_pc_err.txt &
PID_SUBVP_PC=$!

echo ""
echo "Both SubVP no-flow jobs started in parallel:"
echo "GPU 6: SubVP + ODE (PID: $PID_SUBVP_ODE)"
echo "GPU 7: SubVP + PC (PID: $PID_SUBVP_PC)"
echo ""
echo "Waiting for both jobs to complete..."

# Wait for both background jobs to complete
wait $PID_SUBVP_ODE
echo "SubVP ODE no-flow training completed. Log: logs/protein_subvp_noflow_ode_log.txt, Errors: logs/protein_subvp_noflow_ode_err.txt"

wait $PID_SUBVP_PC
echo "SubVP PC no-flow training completed. Log: logs/protein_subvp_noflow_pc_log.txt, Errors: logs/protein_subvp_noflow_pc_err.txt"

echo ""
echo "=== All SubVP no-flow configurations completed ==="
echo "End time: $(date)"
echo ""
echo "Summary of SubVP no-flow runs:"
echo "1. SubVP + ODE solver: outputs/subvp/PROTEIN/ode-noflow/ (GPU 6)"
echo "2. SubVP + PC solver: outputs/subvp/PROTEIN/pc-noflow/ (GPU 7)"
echo ""
echo "Training logs saved in logs/ directory:"
echo "  - Standard output: protein_subvp_noflow_*_log.txt files"
echo "  - Error output: protein_subvp_noflow_*_err.txt files"
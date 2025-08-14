#!/bin/bash

# No-flow diffusion model training sequence with parallel GPU execution
# This script runs 6 configurations in parallel across 6 GPUs:
# - 3 ODE solver configs (VP, VE, subVP) with BPD enabled for VP and subVP
# - 3 PC solver configs (VP, VE, subVP) 
# All configurations use 20000 training iterations and 5000 evaluation samples

echo "Starting no-flow diffusion model training sequence (parallel execution)..."
echo "Date: $(date)"
echo "Running 6 configurations in parallel across GPUs 0-5"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start all 6 training jobs in parallel, each on a different GPU
echo "=== Starting all configurations in parallel ==="

# ODE Solver Configurations (GPUs 0-2)
echo "Starting VP no-flow with ODE solver on GPU 0 (BPD enabled)..."
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --config configs/vp/PROTEIN/indm_noflow_ode.py --workdir outputs/protein_vp_noflow_ode/ > logs/vp_noflow_ode_log.txt 2>&1 &
PID_VP_ODE=$!

echo "Starting VE no-flow with ODE solver on GPU 1 (BPD enabled)..."
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --config configs/ve/PROTEIN/indm_noflow_ode.py --workdir outputs/protein_ve_noflow_ode/ > logs/ve_noflow_ode_log.txt 2>&1 &
PID_VE_ODE=$!

echo "Starting subVP no-flow with ODE solver on GPU 2 (BPD enabled)..."
CUDA_VISIBLE_DEVICES=2 python main.py --mode train --config configs/subvp/PROTEIN/indm_noflow_ode.py --workdir outputs/protein_subvp_noflow_ode/ > logs/subvp_noflow_ode_log.txt 2>&1 &
PID_SUBVP_ODE=$!

# PC Solver Configurations (GPUs 3-5)
echo "Starting VP no-flow with PC solver on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python main.py --mode train --config configs/vp/PROTEIN/indm_noflow_pc.py --workdir outputs/protein_vp_noflow_pc/ > logs/vp_noflow_pc_log.txt 2>&1 &
PID_VP_PC=$!

echo "Starting VE no-flow with PC solver on GPU 4..."
CUDA_VISIBLE_DEVICES=4 python main.py --mode train --config configs/ve/PROTEIN/indm_noflow_pc.py --workdir outputs/protein_ve_noflow_pc/ > logs/ve_noflow_pc_log.txt 2>&1 &
PID_VE_PC=$!

echo "Starting subVP no-flow with PC solver on GPU 5..."
CUDA_VISIBLE_DEVICES=5 python main.py --mode train --config configs/subvp/PROTEIN/indm_noflow_pc.py --workdir outputs/protein_subvp_noflow_pc/ > logs/subvp_noflow_pc_log.txt 2>&1 &
PID_SUBVP_PC=$!

echo ""
echo "All 6 training jobs started in parallel:"
echo "GPU 0: VP + ODE (PID: $PID_VP_ODE)"
echo "GPU 1: VE + ODE (PID: $PID_VE_ODE)"
echo "GPU 2: SubVP + ODE (PID: $PID_SUBVP_ODE)"
echo "GPU 3: VP + PC (PID: $PID_VP_PC)"
echo "GPU 4: VE + PC (PID: $PID_VE_PC)"
echo "GPU 5: SubVP + PC (PID: $PID_SUBVP_PC)"
echo ""
echo "Waiting for all jobs to complete..."

# Wait for all background jobs to complete
wait $PID_VP_ODE
echo "VP ODE training completed. Log saved to logs/vp_noflow_ode_log.txt"

wait $PID_VE_ODE
echo "VE ODE training completed. Log saved to logs/ve_noflow_ode_log.txt"

wait $PID_SUBVP_ODE
echo "SubVP ODE training completed. Log saved to logs/subvp_noflow_ode_log.txt"

wait $PID_VP_PC
echo "VP PC training completed. Log saved to logs/vp_noflow_pc_log.txt"

wait $PID_VE_PC
echo "VE PC training completed. Log saved to logs/ve_noflow_pc_log.txt"

wait $PID_SUBVP_PC
echo "SubVP PC training completed. Log saved to logs/subvp_noflow_pc_log.txt"

echo ""
echo "=== All no-flow training configurations completed ==="
echo "End time: $(date)"
echo ""
echo "Summary of configurations:"
echo "1. VP + ODE (BPD enabled): outputs/protein_vp_noflow_ode/ (GPU 0)"
echo "2. VE + ODE (BPD enabled): outputs/protein_ve_noflow_ode/ (GPU 1)"
echo "3. SubVP + ODE (BPD enabled): outputs/protein_subvp_noflow_ode/ (GPU 2)"
echo "4. VP + PC: outputs/protein_vp_noflow_pc/ (GPU 3)"
echo "5. VE + PC: outputs/protein_ve_noflow_pc/ (GPU 4)"
echo "6. SubVP + PC: outputs/protein_subvp_noflow_pc/ (GPU 5)"
echo ""
echo "All training logs are saved in the logs/ directory."
echo "GPUs 6-7 remain available for other tasks."
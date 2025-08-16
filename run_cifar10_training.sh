#!/bin/bash

echo "Starting diffusion model training sequence for CIFAR10 dataset (parallel execution)..."
echo "Date: $(date)"
echo "Running 6 configurations in parallel across GPUs 0-5"

# Create logs directory if it doesn't exist
mkdir -p logs

# Create outputs directories if they don't exist
mkdir -p outputs/vp/CIFAR10/ode/
mkdir -p outputs/vp/CIFAR10/pc/
mkdir -p outputs/ve/CIFAR10/ode/
mkdir -p outputs/ve/CIFAR10/pc/
mkdir -p outputs/subvp/CIFAR10/ode/
mkdir -p outputs/subvp/CIFAR10/pc/

# Start all 6 training jobs in parallel, each on a different GPU
echo "=== Starting all CIFAR10 configurations in parallel ==="

echo "Starting VP with ODE solver on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --config configs/vp/CIFAR10/indm_fid_ode.py --workdir outputs/vp/CIFAR10/ode/ > logs/cifar10_vp_ode_log.txt 2> logs/cifar10_vp_ode_err.txt &
PID_VP_ODE=$!

echo "Starting VE with ODE solver on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --config configs/ve/CIFAR10/indm_fid_ode.py --workdir outputs/ve/CIFAR10/ode/ > logs/cifar10_ve_ode_log.txt 2> logs/cifar10_ve_ode_err.txt &
PID_VE_ODE=$!

echo "Starting SubVP with ODE solver on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python main.py --mode train --config configs/subvp/CIFAR10/indm_fid_ode.py --workdir outputs/subvp/CIFAR10/ode/ > logs/cifar10_subvp_ode_log.txt 2> logs/cifar10_subvp_ode_err.txt &
PID_SUBVP_ODE=$!

echo "Starting VP with PC solver on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python main.py --mode train --config configs/vp/CIFAR10/indm_fid_pc.py --workdir outputs/vp/CIFAR10/pc/ > logs/cifar10_vp_pc_log.txt 2> logs/cifar10_vp_pc_err.txt &
PID_VP_PC=$!

echo "Starting VE with PC solver on GPU 4..."
CUDA_VISIBLE_DEVICES=4 python main.py --mode train --config configs/ve/CIFAR10/indm_fid_pc.py --workdir outputs/ve/CIFAR10/pc/ > logs/cifar10_ve_pc_log.txt 2> logs/cifar10_ve_pc_err.txt &
PID_VE_PC=$!

echo "Starting SubVP with PC solver on GPU 5..."
CUDA_VISIBLE_DEVICES=5 python main.py --mode train --config configs/subvp/CIFAR10/indm_fid_pc.py --workdir outputs/subvp/CIFAR10/pc/ > logs/cifar10_subvp_pc_log.txt 2> logs/cifar10_subvp_pc_err.txt &
PID_SUBVP_PC=$!

echo ""
echo "All 6 CIFAR10 training jobs started in parallel:"
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
echo "VP ODE training completed. Log: logs/cifar10_vp_ode_log.txt, Errors: logs/cifar10_vp_ode_err.txt"

wait $PID_VE_ODE
echo "VE ODE training completed. Log: logs/cifar10_ve_ode_log.txt, Errors: logs/cifar10_ve_ode_err.txt"

wait $PID_SUBVP_ODE
echo "SubVP ODE training completed. Log: logs/cifar10_subvp_ode_log.txt, Errors: logs/cifar10_subvp_ode_err.txt"

wait $PID_VP_PC
echo "VP PC training completed. Log: logs/cifar10_vp_pc_log.txt, Errors: logs/cifar10_vp_pc_err.txt"

wait $PID_VE_PC
echo "VE PC training completed. Log: logs/cifar10_ve_pc_log.txt, Errors: logs/cifar10_ve_pc_err.txt"

wait $PID_SUBVP_PC
echo "SubVP PC training completed. Log: logs/cifar10_subvp_pc_log.txt, Errors: logs/cifar10_subvp_pc_err.txt"

echo ""
echo "=== All CIFAR10 diffusion model training configurations completed ==="
echo "End time: $(date)"
echo ""
echo "Summary of configurations:"
echo "1. GPU 0: VP + ODE solver: outputs/vp/CIFAR10/ode/"
echo "2. GPU 1: VE + ODE solver: outputs/ve/CIFAR10/ode/"
echo "3. GPU 2: SubVP + ODE solver: outputs/subvp/CIFAR10/ode/"
echo "4. GPU 3: VP + PC solver: outputs/vp/CIFAR10/pc/"
echo "5. GPU 4: VE + PC solver: outputs/ve/CIFAR10/pc/"
echo "6. GPU 5: SubVP + PC solver: outputs/subvp/CIFAR10/pc/"
echo ""
echo "All training logs and error files are saved in the logs/ directory:"
echo "  - Standard output: cifar10_*_log.txt files"
echo "  - Error output: cifar10_*_err.txt files"
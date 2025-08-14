#!/bin/bash


echo "Starting diffusion model training sequence for CELEBA dataset (parallel execution)..."
echo "Date: $(date)"
echo "Running 6 configurations in parallel across GPUs 0-5 with ODE and PC solvers"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start all 6 training jobs in parallel, each on a different GPU
echo "=== Starting all configurations in parallel ==="

# ODE Solver Configurations (GPUs 0-2)
echo "Starting VP wflow with ODE solver on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --config configs/vp/CELEBA/indm_fid_ode.py --workdir outputs/vp/CELEBA/ode/ > logs/vp_ode_log.txt 2> logs/vp_ode_err.txt &
PID_VP_ODE=$!

echo "Starting VE wflow with ODE solver on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --config configs/ve/CELEBA/indm_fid_ode.py --workdir outputs/ve/CELEBA/ode/ > logs/ve_ode_log.txt 2> logs/ve_ode_err.txt &
PID_VE_ODE=$!

echo "Starting subVP wflow with ODE solver on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python main.py --mode train --config configs/subvp/CELEBA/indm_fid_ode.py --workdir outputs/subvp/CELEBA/ode/ > logs/subvp_ode_log.txt 2> logs/subvp_ode_err.txt &
PID_SUBVP_ODE=$!

# PC Solver Configurations (GPUs 3-5)
echo "Starting VP wflow with PC solver on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python main.py --mode train --config configs/vp/CELEBA/indm_fid_pc.py --workdir outputs/vp/CELEBA/pc/ > logs/vp_pc_log.txt 2> logs/vp_pc_err.txt &
PID_VP_PC=$!

echo "Starting VE wflow with PC solver on GPU 4..."
CUDA_VISIBLE_DEVICES=4 python main.py --mode train --config configs/ve/CELEBA/indm_fid_pc.py --workdir outputs/ve/CELEBA/pc/ > logs/ve_pc_log.txt 2> logs/ve_pc_err.txt &
PID_VE_PC=$!

echo "Starting subVP wflow with PC solver on GPU 5..."
CUDA_VISIBLE_DEVICES=5 python main.py --mode train --config configs/subvp/CELEBA/indm_fid_pc.py --workdir outputs/subvp/CELEBA/pc/ > logs/subvp_pc_log.txt 2> logs/subvp_pc_err.txt &
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
echo "VP ODE training completed. Log: logs/vp_ode_log.txt, Errors: logs/vp_ode_err.txt"

wait $PID_VE_ODE
echo "VE ODE training completed. Log: logs/ve_ode_log.txt, Errors: logs/ve_ode_err.txt"

wait $PID_SUBVP_ODE
echo "SubVP ODE training completed. Log: logs/subvp_ode_log.txt, Errors: logs/subvp_ode_err.txt"

wait $PID_VP_PC
echo "VP PC training completed. Log: logs/vp_pc_log.txt, Errors: logs/vp_pc_err.txt"

wait $PID_VE_PC
echo "VE PC training completed. Log: logs/ve_pc_log.txt, Errors: logs/ve_pc_err.txt"

wait $PID_SUBVP_PC
echo "SubVP PC training completed. Log: logs/subvp_pc_log.txt, Errors: logs/subvp_pc_err.txt"

echo ""
echo "=== All diffusion model training configurations completed ==="
echo "End time: $(date)"
echo ""
echo "Summary of configurations:"
echo "1. VP + ODE solver: outputs/vp/CELEBA/ode/ (GPU 0)"
echo "2. VE + ODE solver: outputs/ve/CELEBA/ode/ (GPU 1)"
echo "3. SubVP + ODE solver: outputs/subvp/CELEBA/ode/ (GPU 2)"
echo "4. VP + PC solver: outputs/vp/CELEBA/pc/ (GPU 3)"
echo "5. VE + PC solver: outputs/ve/CELEBA/pc/ (GPU 4)"
echo "6. SubVP + PC solver: outputs/subvp/CELEBA/pc/ (GPU 5)"
echo ""
echo "All training logs and error files are saved in the logs/ directory:"
echo "  - Standard output: *_log.txt files"
echo "  - Error output: *_err.txt files"
echo "GPUs 6-7 remain available for other tasks."
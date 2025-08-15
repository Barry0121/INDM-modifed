#!/bin/bash


echo "Starting PROTEIN dataset VP and VE no-flow model training..."
echo "Date: $(date)"
echo "Running 4 VP/VE no-flow configurations: VP-ODE, VP-PC, VE-ODE, VE-PC on GPUs 0-3"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start 4 VP/VE no-flow training jobs in parallel
echo "=== Starting VP and VE no-flow configurations in parallel ==="

echo "Starting VP no-flow with ODE solver on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python main.py --mode train --config configs/vp/PROTEIN/indm_fid_ode_noflow.py --workdir outputs/vp/PROTEIN/ode-noflow/ > logs/protein_vp_noflow_ode_log.txt 2> logs/protein_vp_noflow_ode_err.txt &
PID_VP_ODE=$!

echo "Starting VP no-flow with PC solver on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python main.py --mode train --config configs/vp/PROTEIN/indm_fid_pc_noflow.py --workdir outputs/vp/PROTEIN/pc-noflow/ > logs/protein_vp_noflow_pc_log.txt 2> logs/protein_vp_noflow_pc_err.txt &
PID_VP_PC=$!

echo "Starting VE no-flow with ODE solver on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python main.py --mode train --config configs/ve/PROTEIN/indm_fid_ode_noflow.py --workdir outputs/ve/PROTEIN/ode-noflow/ > logs/protein_ve_noflow_ode_log.txt 2> logs/protein_ve_noflow_ode_err.txt &
PID_VE_ODE=$!

echo "Starting VE no-flow with PC solver on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python main.py --mode train --config configs/ve/PROTEIN/indm_fid_pc_noflow.py --workdir outputs/ve/PROTEIN/pc-noflow/ > logs/protein_ve_noflow_pc_log.txt 2> logs/protein_ve_noflow_pc_err.txt &
PID_VE_PC=$!

echo ""
echo "All VP/VE no-flow jobs started in parallel:"
echo "GPU 0: VP + ODE (PID: $PID_VP_ODE)"
echo "GPU 1: VP + PC (PID: $PID_VP_PC)"
echo "GPU 2: VE + ODE (PID: $PID_VE_ODE)"
echo "GPU 3: VE + PC (PID: $PID_VE_PC)"
echo ""
echo "Waiting for all jobs to complete..."

# Wait for all background jobs to complete
wait $PID_VP_ODE
echo "VP ODE no-flow training completed. Log: logs/protein_vp_noflow_ode_log.txt, Errors: logs/protein_vp_noflow_ode_err.txt"

wait $PID_VP_PC
echo "VP PC no-flow training completed. Log: logs/protein_vp_noflow_pc_log.txt, Errors: logs/protein_vp_noflow_pc_err.txt"

wait $PID_VE_ODE
echo "VE ODE no-flow training completed. Log: logs/protein_ve_noflow_ode_log.txt, Errors: logs/protein_ve_noflow_ode_err.txt"

wait $PID_VE_PC
echo "VE PC no-flow training completed. Log: logs/protein_ve_noflow_pc_log.txt, Errors: logs/protein_ve_noflow_pc_err.txt"

echo ""
echo "=== All VP and VE no-flow configurations completed ==="
echo "End time: $(date)"
echo ""
echo "Summary of VP/VE no-flow runs:"
echo "1. VP + ODE solver: outputs/vp/PROTEIN/ode-noflow/ (GPU 0)"
echo "2. VP + PC solver: outputs/vp/PROTEIN/pc-noflow/ (GPU 1)"
echo "3. VE + ODE solver: outputs/ve/PROTEIN/ode-noflow/ (GPU 2)"
echo "4. VE + PC solver: outputs/ve/PROTEIN/pc-noflow/ (GPU 3)"
echo ""
echo "Training logs saved in logs/ directory:"
echo "  - Standard output: protein_*_noflow_*_log.txt files"
echo "  - Error output: protein_*_noflow_*_err.txt files"
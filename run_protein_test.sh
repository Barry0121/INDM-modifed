#!/bin/bash


echo "Starting PROTEIN dataset test runs for troubleshooting..."
echo "Date: $(date)"
echo "Running 2 test configurations: VP+ODE and VP+PC on GPUs 6-7"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start 2 test training jobs in parallel
echo "=== Starting test configurations in parallel ==="

echo "Starting VP wflow with ODE solver on GPU 6..."
CUDA_VISIBLE_DEVICES=6 python main.py --mode train --config configs/vp/PROTEIN/indm_fid_ode.py --workdir outputs/vp/PROTEIN/ode_test/ > logs/protein_test_vp_ode_log.txt 2> logs/protein_test_vp_ode_err.txt &
PID_VP_ODE=$!

echo "Starting VP wflow with PC solver on GPU 7..."
CUDA_VISIBLE_DEVICES=7 python main.py --mode train --config configs/vp/PROTEIN/indm_fid_pc.py --workdir outputs/vp/PROTEIN/pc_test/ > logs/protein_test_vp_pc_log.txt 2> logs/protein_test_vp_pc_err.txt &
PID_VP_PC=$!

echo ""
echo "Both test jobs started in parallel:"
echo "GPU 6: VP + ODE (PID: $PID_VP_ODE)"
echo "GPU 7: VP + PC (PID: $PID_VP_PC)"
echo ""
echo "Waiting for both jobs to complete..."

# Wait for both background jobs to complete
wait $PID_VP_ODE
echo "VP ODE test training completed. Log: logs/protein_test_vp_ode_log.txt, Errors: logs/protein_test_vp_ode_err.txt"

wait $PID_VP_PC
echo "VP PC test training completed. Log: logs/protein_test_vp_pc_log.txt, Errors: logs/protein_test_vp_pc_err.txt"

echo ""
echo "=== All test configurations completed ==="
echo "End time: $(date)"
echo ""
echo "Summary of test runs:"
echo "1. VP + ODE solver: outputs/vp/PROTEIN/ode_test/ (GPU 6)"
echo "2. VP + PC solver: outputs/vp/PROTEIN/pc_test/ (GPU 7)"
echo ""
echo "Test logs saved in logs/ directory:"
echo "  - Standard output: protein_test_*_log.txt files"
echo "  - Error output: protein_test_*_err.txt files"
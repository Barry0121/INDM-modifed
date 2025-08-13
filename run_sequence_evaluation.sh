#!/bin/bash

# Script to evaluate protein models with both ODE and PC samplers
# This script evaluates existing trained models with different sampling methods

echo "Starting evaluation of trained protein models with dual samplers..."
echo "================================================================"

# Evaluate VP (Variance Preserving) model with ODE sampler
echo "Evaluating VP model with ODE sampler..."
python main.py --mode eval --config configs/vp/PROTEIN/indm_fid_ode.py --workdir outputs/protein_vp_fid_ode_run/ --assetdir /mnt/data/gpu-server/learning/assets/stats/ > vp_ode_eval_log.txt 2>&1
VP_ODE_STATUS=$?
echo "VP ODE evaluation completed with status: $VP_ODE_STATUS"

# Evaluate VP (Variance Preserving) model with PC sampler
echo "Evaluating VP model with PC sampler..."
python main.py --mode eval --config configs/vp/PROTEIN/indm_fid_pc.py --workdir outputs/protein_vp_fid_pc_run/ --assetdir /mnt/data/gpu-server/learning/assets/stats/ > vp_pc_eval_log.txt 2>&1
VP_PC_STATUS=$?
echo "VP PC evaluation completed with status: $VP_PC_STATUS"

# Evaluate VE (Variance Exploding) model with ODE sampler
echo "Evaluating VE model with ODE sampler..."
python main.py --mode eval --config configs/ve/PROTEIN/indm_ode.py --workdir outputs/protein_ve_ode_run/ --assetdir /mnt/data/gpu-server/learning/assets/stats/ > ve_ode_eval_log.txt 2>&1
VE_ODE_STATUS=$?
echo "VE ODE evaluation completed with status: $VE_ODE_STATUS"

# Evaluate VE (Variance Exploding) model with PC sampler
echo "Evaluating VE model with PC sampler..."
python main.py --mode eval --config configs/ve/PROTEIN/indm_pc.py --workdir outputs/protein_ve_pc_run/ --assetdir /mnt/data/gpu-server/learning/assets/stats/ > ve_pc_eval_log.txt 2>&1
VE_PC_STATUS=$?
echo "VE PC evaluation completed with status: $VE_PC_STATUS"

# Evaluate subVP (sub-Variance Preserving) model with ODE sampler
echo "Evaluating subVP model with ODE sampler..."
python main.py --mode eval --config configs/subvp/PROTEIN/indm_ode.py --workdir outputs/protein_subvp_ode_run/ --assetdir /mnt/data/gpu-server/learning/assets/stats/ > subvp_ode_eval_log.txt 2>&1
SUBVP_ODE_STATUS=$?
echo "subVP ODE evaluation completed with status: $SUBVP_ODE_STATUS"

# Evaluate subVP (sub-Variance Preserving) model with PC sampler
echo "Evaluating subVP model with PC sampler..."
python main.py --mode eval --config configs/subvp/PROTEIN/indm_pc.py --workdir outputs/protein_subvp_pc_run/ --assetdir /mnt/data/gpu-server/learning/assets/stats/ > subvp_pc_eval_log.txt 2>&1
SUBVP_PC_STATUS=$?
echo "subVP PC evaluation completed with status: $SUBVP_PC_STATUS"

echo "================================================================"
echo "Evaluation Summary:"
echo "VP ODE sampler status: $VP_ODE_STATUS"
echo "VP PC sampler status: $VP_PC_STATUS"
echo "VE ODE sampler status: $VE_ODE_STATUS"
echo "VE PC sampler status: $VE_PC_STATUS"
echo "subVP ODE sampler status: $SUBVP_ODE_STATUS"
echo "subVP PC sampler status: $SUBVP_PC_STATUS"

if [ $VP_ODE_STATUS -eq 0 ] && [ $VP_PC_STATUS -eq 0 ] && [ $VE_ODE_STATUS -eq 0 ] && [ $VE_PC_STATUS -eq 0 ] && [ $SUBVP_ODE_STATUS -eq 0 ] && [ $SUBVP_PC_STATUS -eq 0 ]; then
    echo "All evaluations completed successfully!"
else
    echo "Some evaluations failed. Check the log files:"
    echo "  - vp_ode_eval_log.txt"
    echo "  - vp_pc_eval_log.txt"
    echo "  - ve_ode_eval_log.txt"
    echo "  - ve_pc_eval_log.txt"
    echo "  - subvp_ode_eval_log.txt"
    echo "  - subvp_pc_eval_log.txt"
fi

echo "================================================================"
echo "Evaluation results can be found in:"
echo "  - outputs/protein_vp_fid_ode_run/eval/"
echo "  - outputs/protein_vp_fid_pc_run/eval/"
echo "  - outputs/protein_ve_ode_run/eval/"
echo "  - outputs/protein_ve_pc_run/eval/"
echo "  - outputs/protein_subvp_ode_run/eval/"
echo "  - outputs/protein_subvp_pc_run/eval/"
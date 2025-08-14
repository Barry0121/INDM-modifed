#!/bin/bash

# Training script for ODE-configured VP, SubVP, and VE runs
# Short training runs (2005 steps) with frequent evaluation (every 1000 steps)
# BPD evaluation disabled, logging every 100 steps

echo "Starting ODE flow training sequence..."

echo "Training VP with ODE sampling..."
CUDA_VISIBLE_DEVICES=6 python main.py --mode train --config configs/vp/PROTEIN/indm_fid_ode.py --workdir outputs/protein_vp_ode_flow_run/ > vp_ode_flow_log.txt

# echo "Training SubVP with ODE sampling..."
# python main.py --mode train --config configs/subvp/PROTEIN/indm_ode.py --workdir outputs/protein_subvp_ode_flow_run/ > subvp_ode_flow_log.txt

# echo "Training VE with ODE sampling..."
# python main.py --mode train --config configs/ve/PROTEIN/indm_ode.py --workdir outputs/protein_ve_ode_flow_run/ > ve_ode_flow_log.txt

echo "ODE flow training sequence completed!"
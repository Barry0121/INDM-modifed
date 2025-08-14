#!/bin/bash

# Training script for PC (Predictor-Corrector) configured VP, SubVP, and VE runs
# Short training runs (2005 steps) with frequent evaluation (every 1000 steps)
# BPD evaluation disabled, logging every 100 steps

echo "Starting PC flow training sequence..."

echo "Training VP with PC sampling..."
python main.py --mode train --config configs/vp/PROTEIN/indm_fid_pc.py --workdir outputs/protein_vp_pc_flow_run/ > vp_pc_flow_log.txt

echo "Training SubVP with PC sampling..."
python main.py --mode train --config configs/subvp/PROTEIN/indm_pc.py --workdir outputs/protein_subvp_pc_flow_run/ > subvp_pc_flow_log.txt

echo "Training VE with PC sampling..."
python main.py --mode train --config configs/ve/PROTEIN/indm_pc.py --workdir outputs/protein_ve_pc_flow_run/ > ve_pc_flow_log.txt

echo "PC flow training sequence completed!"
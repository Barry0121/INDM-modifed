#!/bin/bash

# python main.py --mode train --config configs/vp/PROTEIN/indm_noflow_ode.py --workdir outputs/protein_vp_noflow_ode/ > vp_noflow_log.txt
python main.py --mode train --config configs/ve/PROTEIN/indm_noflow_ode.py --workdir outputs/protein_ve_noflow_ode/ > ve_noflow_log.txt
# python main.py --mode train --config configs/subvp/PROTEIN/indm_noflow_ode.py --workdir outputs/protein_subvp_noflow_ode/ > subvp_noflow_log.txt
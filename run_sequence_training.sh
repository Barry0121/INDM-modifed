#!/bin/bash

python main.py --mode train --config configs/vp/PROTEIN/indm_fid.py --workdir outputs/protein_vp_fid_run/ > vp_fid_full_log.txt
python main.py --mode train --config configs/ve/PROTEIN/indm.py --workdir outputs/protein_ve_fid_run/ > ve_fid_full_log.txt
python main.py --mode train --config configs/subvp/PROTEIN/indm.py --workdir outputs/protein_subvp_fid_run/ > subvp_fid_full_log.txt
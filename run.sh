#!/bin/bash

# Example: run active learning with different selection methods
# Usage: bash run.sh

# random
python main.py --device 0 --selection_method random --cycle 0 --expt 1 --run_name 'random_c0'

# coreset
# python main.py --device 0 --selection_method coreset --cycle 0 --expt 1 --run_name 'coreset_c0'

# lloss
# python main.py --device 0 --selection_method lloss --cycle 0 --expt 1 --run_name 'lloss_c0'

# mcdrop
# python main.py --device 0 --selection_method mcdrop --cycle 0 --expt 1 --run_name 'mcdrop_c0'

# unc_div (uncertainty + diversity)
# python main.py --device 0 --selection_method unc_div --cycle 0 --expt 1 --run_name 'unc_div_c0'

# rep
# python main.py --device 0 --selection_method rep --cycle 0 --expt 1 --run_name 'rep_c0'

# dis (discriminator)
# python main.py --device 0 --selection_method dis --cycle 0 --expt 1 --run_name 'dis_c0'

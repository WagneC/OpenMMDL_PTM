#!/bin/bash

module load conda
conda activate openmmdl_ptmv1
cd ~/openmmdl_ptm/OpenMMDL
pip uninstall -y openmmdl && pip install -e .
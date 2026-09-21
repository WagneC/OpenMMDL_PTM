#!/bin/bash

module load conda
conda env remove -n openmmdl_ptmv1
cd OpenMMDL
conda env create -f environment.yml -n openmmdl_ptmv1
conda activate openmmdl_ptmv1
pip install .
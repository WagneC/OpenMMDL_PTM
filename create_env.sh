#!/bin/bash

module load conda
conda deactivate
echo y | conda env remove -n openmmdl_ptmv1
echo y | conda env create -f environment.yml -n openmmdl_ptmv1
conda activate openmmdl_ptmv1
pip install .
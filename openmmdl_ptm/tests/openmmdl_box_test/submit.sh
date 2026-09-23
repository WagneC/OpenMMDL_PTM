#!/bin/bash
#SBATCH --job-name=Test_Openmmdl_PTM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu-ultrashort 
#SBATCH --gres=gpu:1
#SBATCH --array=1		# adjust acording to the amount of repilicas needed 1-5


# simulation variables
workdir=/home/cwagner/openmmdl_ptm/OpenMMDL/openmmdl_ptm/tests/openmmdl_box_test
topology=3ip9_dye-processed_openMMDL.pdb
ligand=maleimide.sdf
script=Box_script.py

# environment variables
module load conda
conda activate openmmdl_ptm1   					

# execute simulation
cd ${workdir}
mkdir ./${SLURM_ARRAY_TASK_ID} 

openmmdl simulation -f ./${SLURM_ARRAY_TASK_ID}/ -t ${topology} -s ${script} -l ${ligand}  

# openmmdl simulation -f sim -t 3ip9_dye-processed_openMMDL.pdb -s OpenMMDL.py -l maleimide.sdf 
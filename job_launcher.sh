#!/bin/bash
#SBATCH --job-name=R18
#SBATCH --partition=prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --array=3-4
#SBATCH --output=slurm_output/Resnet18/a_r5018_slurm-%A_%a.out

source activate py_env2
module load cuda/10.0


export PYTHONPATH="${PYTHONPATH}:/homes/svincenzi/.conda/envs/py_env2/bin/python"

srun python -u main.py --id_optim=${SLURM_ARRAY_TASK_ID}
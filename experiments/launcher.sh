#!/bin/bash -l
#SBATCH -N 1
#SBATCH --time=0-20:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL,INVALID_DEPEND,REQUEUE
#SBATCH --mail-user=<your-email@institution.edu>
conda activate xul_env
python train.py

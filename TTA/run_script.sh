#!/bin/bash

#SBATCH --partition=gpu_a100       # Partition name
#SBATCH --gres=gpu:1                  # Number of GPUs to allocate
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --gpus=1                      # This line is sometimes optional/redundant
#SBATCH --job-name=build_pybullet    # Job name
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --time=00:30:00               # Time limit hh:mm:ss
#SBATCH --mem=16000M                  # Memory pool for all cores (16GB)
#SBATCH --output=snellius_outputs/build_pybullet_%A.out  # Standard output

### --- MODULE SETUP / ENVIRONMENT ---
module purge
module load 2022
module load Anaconda3/2022.05

source activate tta

srun python run_peract_pyb.py

#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --job-name=peract_pyb
#SBATCH --output=work/peract_pyb_%A.out
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

## --- MODULES & ENVIRONMENT ---
module purge
module load 2022
# module spider CUDA/11.7.0 # 'module spider' is for searching, 'module load' for loading
# module spider cuDNN/8.4.1.50-CUDA-11.7.0
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0


# Paths
WORKDIR=$(pwd)                      # e.g. ~/dream-team
RLBENCH_DIR_HOST=/home/scur2683/dream-team/RLBench/urdfs/panda # Path on the host
SIF=$WORKDIR/peract_pybullet.sif

# where our smoke‐test script lives
PYB_SCRIPT=$WORKDIR/TTA/run_peract_pyb.py
# PYB_SCRIPT=$WORKDIR/TTA/minimal_pybullet_test.py # CHANGED


# make sure logs folder exists
mkdir -p $WORKDIR/work

# PERACT_SRC=$WORKDIR/peract # Not used in the singularity command

singularity exec --nv \
  --bind "$WORKDIR":"/root/dream-team" \
  --bind "$RLBENCH_DIR_HOST":"/urdfs/panda" \
  --env COPPELIASIM_ROOT="/opt/coppeliaSim" \
  --env LD_LIBRARY_PATH="/opt/coppeliaSim:${LD_LIBRARY_PATH}" \
  --env QT_QPA_PLATFORM_PLUGIN_PATH="/opt/coppeliaSim" \
  --env QT_QPA_PLATFORM="offscreen" \
  --env PYOPENGL_PLATFORM="egl" \
  --env PYTHONPATH="/root/install/peract:/root/install/RLBench:/root/install/YARR:/root/install/PyRep:${PYTHONPATH}" \
  $SIF \
  /opt/pyenv/bin/python $PYB_SCRIPT
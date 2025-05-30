#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --output=work/simulation_%A.out
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9

# Print node information
echo "Job running on $(hostname)"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"



# Load necessary modules (modify as needed for your system)
#module purge
#module load 2023
#module load Anaconda3/2023.07-2


# Remove any pre-loaded Anaconda module
module purge
#module unload Anaconda3

# Initialize and activate your Conda environment
# Set the PATH to include your environment's bin directory
export PATH="/gpfs/home4/scur2683/.conda/envs/peract_leonardo/bin:$PATH"

# Activate your conda environment
conda activate peract_leonardo

# Check if the environment is activated (optional)
conda info --envs
echo "Active Conda environment: $(conda env list | grep '*')"




# Load necessary modules (adjust based on your environment)
#conda activate peract_leonardo
# Run the Python script using hydra
#python create_simulation.py
python simulate.py
echo "DREMA scene building script finished."

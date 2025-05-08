#!/bin/bash
# This script is executed INSIDE the Apptainer container.
# It runs the training for 3D Diffuser Actor.

set -e # Exit on error

echo '--- Inside Container: exec_training_inside_container.sh ---'
echo "Current PWD: $(pwd)" # Should be /dream-team

# 1) Load common environment variables
echo "Sourcing common environment setup: /dream-team/snellius_env/env_setup.sh"
source /dream-team/snellius_env/env_setup.sh

# 2) Activate Python venv
echo "Activating Python venv: /opt/pyenv/bin/activate"
source /opt/pyenv/bin/activate
export PYTHONIOENCODING=utf-8

# 3) Setup PYTHONPATH
export PYTHONPATH=/dream-team/3d_diffuser_actor:${PYTHONPATH}
echo "PYTHONPATH set to: $PYTHONPATH"

# 4) GPU Check
echo '--- GPU check ---'
nvidia-smi || echo "nvidia-smi not found or failed"
python3 -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available for Torch: {torch.cuda.is_available()}')"

# 5) Install/Verify Diffuser Actor dependencies (user-space)
echo 'Installing/Checking 3d_diffuser_actor specific dependencies...'
pip list | grep diffusers || pip install --user diffusers[torch]
pip list | grep dgl || pip install --user dgl==1.1.3+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html
pip list | grep packaging || pip install --user packaging
pip list | grep ninja || pip install --user ninja
pip list | grep flash-attn || pip install --user flash-attn==2.5.9.post1 --no-build-isolation
pip list | grep typed-argument-parser || pip install --user typed-argument-parser
pip list | grep einops || pip install --user einops
pip list | grep transformers || pip install --user transformers
pip list | grep open3d || pip install --user open3d==0.17.0
echo 'Installing 3d_diffuser_actor editable...'
pip install --user -e /dream-team/3d_diffuser_actor # From the bound project directory

# 6) Navigate to the 3D Diffuser Actor code directory
cd /dream-team/3d_diffuser_actor
echo "Changed PWD to: $(pwd)"

# --- Configuration (expected to be set as ENV VARS by the calling Slurm script) ---
# PACKAGED_DATA_TRAIN_ABS: /path/to/packaged/train
# PACKAGED_DATA_VAL_ABS: /path/to/packaged/val
# INSTRUCTIONS_ABS: /path/to/instructions.pkl
# GRIPPER_BOUNDS_ABS: /path/to/gripper_bounds.json
# LOG_BASE_DIR_ABS: /path/to/log_output_base
# TASKS_TO_TRAIN_ON: Space-separated list
# Other training params like LR, BATCH_SIZE, EMBEDDING_DIM, NGPUS etc.

echo "--- Starting Training ---"
echo "Packaged Train Data: ${PACKAGED_DATA_TRAIN_ABS}"
echo "Packaged Val Data: ${PACKAGED_DATA_VAL_ABS}"
echo "Instructions: ${INSTRUCTIONS_ABS}"
# ... echo other params ...

# DDP / torchrun setup
export MASTER_PORT=$(expr 10000 + $SLURM_JOB_ID % 50000)
echo "MASTER_ADDR: $MASTER_ADDR (Expected from Slurm)"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $SLURM_NTASKS"; echo "RANK: $SLURM_PROCID"; echo "LOCAL_RANK: $SLURM_LOCALID"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# Log directory names
main_log_dir_name="Actor_${MAIN_DIR_SUFFIX}" # e.g., Actor_My3TasksCustomData
run_log_dir_name="diffusion_multitask-${RUN_LOG_SUFFIX}-C${EMBEDDING_DIM}-B${BATCH_SIZE_PER_GPU}-lr${LR}-DI${DENSE_INTERPOLATION}-${INTERPOLATION_LENGTH}-H${NUM_HISTORY}-DT${DIFFUSION_TIMESTEPS}"

# Run training using xvfb-run and torchrun
# Note: main_trajectory.py itself creates matplotlib figures. Xvfb handles this.
xvfb-run -a torchrun --nproc_per_node ${NGPUS} --master_port ${MASTER_PORT} \
    main_trajectory.py \
    --tasks ${TASKS_TO_TRAIN_ON} \
    --dataset "${PACKAGED_DATA_TRAIN_ABS}" \
    --valset "${PACKAGED_DATA_VAL_ABS}" \
    --instructions "${INSTRUCTIONS_ABS}" \
    --gripper_loc_bounds "${GRIPPER_BOUNDS_ABS}" \
    --num_workers ${NUM_WORKERS_DATALOADER:-4} \
    --train_iters ${TRAIN_ITERS:-600000} \
    --embedding_dim ${EMBEDDING_DIM:-120} \
    --use_instruction ${USE_INSTRUCTION:-1} \
    --rotation_parametrization ${ROTATION_PARAMETRIZATION:-6D} \
    --diffusion_timesteps ${DIFFUSION_TIMESTEPS:-100} \
    --val_freq ${VAL_FREQ:-4000} \
    --dense_interpolation ${DENSE_INTERPOLATION:-1} \
    --interpolation_length ${INTERPOLATION_LENGTH:-2} \
    --base_log_dir "${LOG_BASE_DIR_ABS}" \
    --exp_log_dir "${main_log_dir_name}" \
    --batch_size ${BATCH_SIZE_PER_GPU:-8} \
    --batch_size_val ${BATCH_SIZE_VAL:-14} \
    --cache_size ${CACHE_SIZE:-600} \
    --cache_size_val ${CACHE_SIZE_VAL:-0} \
    --keypose_only ${KEYPOSE_ONLY:-1} \
    --variations {0..199} \
    --lr ${LR:-1e-4} \
    --num_history ${NUM_HISTORY:-3} \
    --cameras left_shoulder right_shoulder wrist front \
    --max_episodes_per_task ${MAX_EPISODES_PER_TASK:--1} \
    --quaternion_format ${QUATERNION_FORMAT:-xyzw} \
    --run_log_dir "${run_log_dir_name}"

echo "--- Training script finished ---"

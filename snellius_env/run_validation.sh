#!/bin/bash

echo '--- container /dev/nvidia* ---'
ls -l /dev/nvidia* || echo "(none)"

# 1) load your env vars
source /dream-team/snellius_env/env_setup.sh


# 2) final GPU check
echo '--- final GPU check ---'
nvidia-smi
python3 -c "import torch; print('CUDA visible?', torch.cuda.is_available())"


# Define the log directory where the training config should be
# TRAIN_LOG_DIR="/gpfs/home4/scur2683/dream-team/tmp_all/$PORT"
TASK_NAME="slide_block_to_color_target"
METHOD_NAME="PERACT_BC"
SEED_DIR="$TRAIN_LOG_DIR/$TASK_NAME/$METHOD_NAME/seed0"


export QT_QPA_PLATFORM=xcb # <--- ADD THIS LINE
export QT_DEBUG_PLUGINS=1

# 3) run training under Xvfb
xvfb-run -a python3 /root/install/peract/eval.py \
  rlbench.tasks=${HYDRA_OVERRIDE_RLBENCH_TASKS} \
  rlbench.task_name='slide_block_to_color_target' \
  rlbench.demo_path=$DEMO \
  framework.logdir=$LOG_DIR \
  framework.csv_logging=True \
  framework.tensorboard_logging=True \
  framework.eval_envs=4 \
  framework.start_seed=0 \
  framework.eval_from_eps_number=0 \
  framework.eval_episodes=25 \
  framework.eval_type='missing' \
  rlbench.headless=True

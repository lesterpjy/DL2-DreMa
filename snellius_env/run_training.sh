#!/bin/bash

echo '--- container /dev/nvidia* ---'
ls -l /dev/nvidia* || echo "(none)"

# 1) load your env vars
source /dream-team/snellius_env/env_setup.sh


# 2) final GPU check
echo '--- final GPU check ---'
nvidia-smi
python3 -c "import torch; print('CUDA visible?', torch.cuda.is_available())"

# 3) run training under Xvfb
xvfb-run -a python3 /root/install/peract/train.py \
  method=PERACT_BC \
  rlbench.tasks=${HYDRA_OVERRIDE_RLBENCH_TASKS} \
  rlbench.task_name='slide_block_to_color_target' \
  rlbench.cameras=[front,right_shoulder,left_shoulder] \
  rlbench.demos=10 \
  rlbench.demo_path=$DEMO \
  replay.batch_size=4 \
  replay.path=$REPLAY \
  replay.max_parallel_processes=10 \
  method.voxel_sizes=[100] \
  method.voxel_patch_size=5 \
  method.voxel_patch_stride=5 \
  method.num_latents=2048 \
  method.transform_augmentation.apply_se3=True \
  method.transform_augmentation.aug_rpy=[0.0,0.0,45.0] \
  method.pos_encoding_with_lang=True \
  framework.training_iterations=100100 \
  framework.num_weights_to_keep=60 \
  framework.start_seed=0 \
  framework.log_freq=1000 \
  framework.save_freq=5000 \
  framework.logdir=$LOG_DIR \
  framework.csv_logging=True \
  framework.tensorboard_logging=True \
  ddp.num_devices=1 \
  ddp.master_port="'$PORT'" \
  +framework.weights_dir=/home/scur2683/dream-team/outputs/logs/slide_block_to_color_target/PERACT_BC/seed0/weights/55000 \
  framework.load_existing_weights=true
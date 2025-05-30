#!/bin/bash
# This script is executed INSIDE the Apptainer container

echo '--- Inside Container ---'
echo "User: $(whoami)"
echo "Hostname: $(hostname)"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "COPPELIASIM_ROOT (expected /opt/coppeliaSim): $COPPELIASIM_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo "QT_QPA_PLATFORM: $QT_QPA_PLATFORM"
echo "PYOPENGL_PLATFORM: $PYOPENGL_PLATFORM"
echo "LIBGL_ALWAYS_SOFTWARE: $LIBGL_ALWAYS_SOFTWARE"


echo '--- Inside Container ---'
# ...
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "COPPELIASIM_ROOT (expected /opt/coppeliaSim): $COPPELIASIM_ROOT"
echo "Checking /opt/coppeliaSim..."


echo "Checking /opt/coppeliaSim..."
ls -l /opt/coppeliaSim/

echo "Running nvidia-smi..."
nvidia-smi

echo "Checking python versions and paths AT RUNTIME:"
echo "which python: $(which python)"
echo "which python3: $(which python3)"
echo "python is: $(readlink -f $(which python))"
echo "python3 is: $(readlink -f $(which python3))"
python3 -V
python3 -c "import sys; print(f'Python3 sys.version (RUNTIME): {sys.version}')"
echo "Python3 sys.path (RUNTIME):"
python3 -c "import sys; print(sys.path)"
echo "Attempting to import hydra AT RUNTIME before eval.py:"
python3 -c 'import hydra; print(f"Hydra imported AT RUNTIME. Version: {hydra.__version__}")' || echo "ERROR: Hydra import FAILED AT RUNTIME"


echo "Setting current directory to /root/install/RVT..."
cd /root/install

echo "Executing PerAct evaluation command with xvfb-run..."

CONTAINER_DEMO_PATH_SCRIPT="/mnt/project_dir/evaluation/demos"
CONTAINER_LOG_DIR_SCRIPT="/mnt/project_dir/evaluation/eval_logs"
CONTAINER_CHECKPOINT_PATH_SCRIPT="/mnt/project_dir/evaluation/checkpoint"
PERACT_SCRIPT_PATH_SCRIPT="/root/install/peract/eval.py"
HYDRA_OUTPUT_SUBDIR_SCRIPT="hydra_run_output"

TASK_VARIANTS_FOR_EVAL_SCRIPT="${1:-slide_block_to_color_target}"
EVAL_EPISODES_SCRIPT="${2:-1}"
EVAL_ENVS_SCRIPT="${3:-4}"

EVAL_CMD="python3 ${PERACT_SCRIPT_PATH_SCRIPT} \
  rlbench.tasks='${TASK_VARIANTS_FOR_EVAL_SCRIPT}' \
  rlbench.task_name='${TASK_VARIANTS_FOR_EVAL_SCRIPT}' \
  rlbench.demo_path='${CONTAINER_DEMO_PATH_SCRIPT}' \
  framework.logdir='${CONTAINER_LOG_DIR_SCRIPT}' \
  framework.csv_logging=True \
  framework.tensorboard_logging=True \
  framework.eval_envs=${EVAL_ENVS_SCRIPT} \
  framework.start_seed=0 \
  framework.eval_from_eps_number=0 \
  framework.eval_episodes=${EVAL_EPISODES_SCRIPT} \
  framework.eval_type='missing' \
  rlbench.headless=True \
  hydra.run.dir='${CONTAINER_LOG_DIR_SCRIPT}/${HYDRA_OUTPUT_SUBDIR_SCRIPT}'"

echo "XVFB_RUN_ARGS: -a"
echo "PYTHON_COMMAND_WITH_ARGS: ${EVAL_CMD}"

xvfb-run -a ${EVAL_CMD}
XVFB_EXIT_CODE=$? # Capture exit code of xvfb-run

echo "--- Evaluation Finished (xvfb-run exit code: $XVFB_EXIT_CODE) ---"
exit $XVFB_EXIT_CODE
#!/bin/bash

set -e # Exit on error

echo '--- Inside Container: exec_dataprep_inside_container.sh ---'
echo "Current PWD: $(pwd)"

# --- Environment Setup ---
export XDG_RUNTIME_DIR="/tmp/xdg_runtime_$(whoami)_$$" 
mkdir -p "$XDG_RUNTIME_DIR"
chmod 0700 "$XDG_RUNTIME_DIR"
echo "XDG_RUNTIME_DIR: ${XDG_RUNTIME_DIR}"

export COPPELIASIM_ROOT=/opt/coppeliaSim
echo "COPPELIASIM_ROOT: ${COPPELIASIM_ROOT}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}:${COPPELIASIM_ROOT}/platforms:${LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}/platforms"
echo "QT_QPA_PLATFORM_PLUGIN_PATH: ${QT_QPA_PLATFORM_PLUGIN_PATH}"
export QT_QPA_PLATFORM="offscreen"
echo "QT_QPA_PLATFORM: ${QT_QPA_PLATFORM}"
export PYOPENGL_PLATFORM="egl"
echo "PYOPENGL_PLATFORM: ${PYOPENGL_PLATFORM}"
# export LIBGL_ALWAYS_SOFTWARE="1"

echo "Activating Python venv: /opt/diffuser_venv/bin/activate"
source /opt/diffuser_venv/bin/activate
export PYTHONIOENCODING=utf-8

export PYTHONPATH=/dream-team/3d_diffuser_actor:/root/install/PyRep:/root/install/RLBench:${PYTHONPATH}
echo "PYTHONPATH updated to: $PYTHONPATH"

# --- Clear CoppeliaSim user settings ---
echo "Clearing CoppeliaSim user settings..."
rm -rf /root/.config/CoppeliaSim || true
rm -rf /root/.CoppeliaSim     || true
echo "User settings cleared."

cd /dream-team/3d_diffuser_actor # Scripts will be called from here
echo "Changed PWD to: $(pwd)"

# --- Configuration ---
echo "--- Starting Data Processing for SPLIT: ${MY_SPLIT} ---"

DEMO_ROOT_FOR_RERENDER="${MY_RAW_RLBENCH_DATA_ROOT_ABS}/${MY_SPLIT}"
RAW_HIGHRES_SAVE_PATH="${MY_OUTPUT_RAW_HIGHRES_ROOT_ABS}/${MY_SPLIT}"
PACKAGED_SAVE_PATH="${MY_OUTPUT_PACKAGED_ROOT_ABS}/${MY_SPLIT}"

PYTHON_EXEC_RERENDER="./data_preprocessing/rerender_highres_rlbench.py" # No debug prints here anymore
PYTHON_EXEC_REARRANGE="./data_preprocessing/rearrange_rlbench_demos.py"
PYTHON_EXEC_PACKAGE="./data_preprocessing/package_rlbench.py"

# === Step 1: Re-render ===
echo "--- Step 1: Re-rendering ---"
for task_name in ${TASKS_TO_PROCESS}
do
    echo "Re-rendering task: $task_name for split: ${MY_SPLIT}"
    python3 ${PYTHON_EXEC_RERENDER} \
        --tasks="$task_name" \
        --save_path="${RAW_HIGHRES_SAVE_PATH}" \
        --demo_path="${DEMO_ROOT_FOR_RERENDER}" \
        --image_size=256,256 \
        --renderer=opengl \
        --processes=1 \
        --all_variations=True
done

# === Step 2 & 3 (unchanged) ===
echo "--- Step 2: Rearranging directories ---"
xvfb-run -a python3 ${PYTHON_EXEC_REARRANGE} --root_dir "${RAW_HIGHRES_SAVE_PATH}"

echo "--- Step 3: Packaging episodes ---"
for task_name in ${TASKS_TO_PROCESS}
do
    echo "Packaging task: $task_name for split: ${MY_SPLIT}"
    xvfb-run -a python3 ${PYTHON_EXEC_PACKAGE} \
        --data_dir="${RAW_HIGHRES_SAVE_PATH}" \
        --tasks="$task_name" \
        --output="${PACKAGED_SAVE_PATH}" \
        --store_intermediate_actions=1 \
        --cameras left_shoulder right_shoulder wrist front
done

echo "--- Data Processing for SPLIT: ${MY_SPLIT} finished successfully ---"

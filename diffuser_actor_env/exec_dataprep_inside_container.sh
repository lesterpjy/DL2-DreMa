#!/bin/bash
# Runs inside diffuser_actor_jammy.sif

set -e # Exit on error

echo '--- Inside Container: exec_dataprep_inside_container.sh ---'
echo "Current PWD: $(pwd)" # Should be /dream-team initially

# 1) Environment is mostly set by the SIF's %environment block now
#    (includes venv activation, COPPELIASIM_ROOT, QT vars, PYTHONPATH for PyRep/RLBench)
echo "--- Environment Check ---"
echo "PATH=$PATH"
echo "PYTHONPATH=$PYTHONPATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "QT_QPA_PLATFORM=${QT_QPA_PLATFORM}"
echo "QT_QPA_PLATFORM_PLUGIN_PATH=${QT_QPA_PLATFORM_PLUGIN_PATH}"
python3 -c "import sys; print(f'Python executable: {sys.executable}')"

# Add bind-mounted diffuser actor code dir to PYTHONPATH if needed by preprocessing scripts
export PYTHONPATH=/dream-team/3d_diffuser_actor:${PYTHONPATH}
echo "PYTHONPATH updated to: $PYTHONPATH"

# Optional: Debug QT Plugins loading if needed
# export QT_DEBUG_PLUGINS="1" 

# 2) GPU Check (optional)
echo '--- GPU check ---'
# nvidia-smi || echo "nvidia-smi not found or failed"
# python3 -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available for Torch: {torch.cuda.is_available()}')"

# 3) Navigate to the 3D Diffuser Actor code directory
cd /dream-team/3d_diffuser_actor
echo "Changed PWD to: $(pwd)"

# --- Configuration (from ENV VARS) ---
echo "--- Starting Data Processing for SPLIT: ${MY_SPLIT} ---"
# ... (echo other config vars) ...

DEMO_ROOT_FOR_RERENDER="${MY_RAW_RLBENCH_DATA_ROOT_ABS}/${MY_SPLIT}"
RAW_HIGHRES_SAVE_PATH="${MY_OUTPUT_RAW_HIGHRES_ROOT_ABS}/${MY_SPLIT}"
PACKAGED_SAVE_PATH="${MY_OUTPUT_PACKAGED_ROOT_ABS}/${MY_SPLIT}"

# Output dirs created by Slurm script, no need to mkdir here

PYTHON_EXEC_RERENDER="./data_preprocessing/rerender_highres_rlbench.py"
PYTHON_EXEC_REARRANGE="./data_preprocessing/rearrange_rlbench_demos.py"
PYTHON_EXEC_PACKAGE="./data_preprocessing/package_rlbench.py"

# === Step 1: Re-render high-resolution camera views ===
echo "--- Step 1: Re-rendering (using xvfb-run) ---"
for task_name in ${TASKS_TO_PROCESS}
do
    echo "Re-rendering task: $task_name for split: ${MY_SPLIT}"
    xvfb-run -a \
    python3 ${PYTHON_EXEC_RERENDER} \
        --tasks="$task_name" \
        --save_path="${RAW_HIGHRES_SAVE_PATH}" \
        --demo_path="${DEMO_ROOT_FOR_RERENDER}" \
        --image_size=256,256 \
        --renderer=opengl \
        --processes=1 \
        --all_variations=True
done

# === Step 2: Re-arrange directory ===
echo "--- Step 2: Rearranging directories ---"
xvfb-run -a python3 ${PYTHON_EXEC_REARRANGE} --root_dir "${RAW_HIGHRES_SAVE_PATH}"

# === Step 3: Package episodes into .dat files ===
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

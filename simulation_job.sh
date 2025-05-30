#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --output=work/drema_simulation_%A.out
#SBATCH --time=01:00:00 # Adjust time limit as needed
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

echo "--- Starting DREMA Simulation using SIF ---"

# --- Paths ---
PROJECT_ROOT_WSL2="$HOME/dream-team"
SIF_FILE_PATH="$PROJECT_ROOT_WSL2/drema_datagen.sif" # Assuming the same SIF contains necessary simulation tools
LOG_FILE="$PROJECT_ROOT_WSL2/drema_simulation_output.log"

echo "SIF File: $SIF_FILE_PATH"
echo "Log File: $LOG_FILE"
echo "Host DISPLAY is: $DISPLAY"
echo "Running… output in $LOG_FILE"


APPTAINER_COMMAND="apptainer exec --nv --bind \"$PROJECT_ROOT_WSL2:/workspace:rw\" \"$HOME/dream-team/drema_datagen.sif\" bash -lc 'set -eux; echo \"--- Starting commands inside bash -lc for simulation ---\"; echo \"Attempting to start Xvfb...\"; Xvfb :99 -screen 0 1280x1024x24 +extension GLX +render -noreset & XFB_PID=\$!; echo \"Xvfb PID: \$XFB_PID\"; sleep 2; export DISPLAY=:99; echo \"DISPLAY is now \$DISPLAY\"; echo \"Running glxinfo...\"; glxinfo | grep \"OpenGL renderer\"; echo \"Running Python simulation script...\"; cd /workspace; python3 simulate.py configs/config.yaml; PY_EXIT_CODE=\$?; echo \"Python simulation script finished with exit code: \$PY_EXIT_CODE\"; kill \$XFB_PID || echo \"Xvfb already exited or kill failed\"; exit \$PY_EXIT_CODE;'"

eval "$APPTAINER_COMMAND" > "$LOG_FILE" 2>&1

EXIT_STATUS=$?
if [ $EXIT_STATUS -eq 0 ]; then
  echo "✅ Simulation finished successfully."
  echo "  Check output in: $LOG_FILE" # Simulation output will be in the log file
else
  echo "❌ Simulation failed (exit $EXIT_STATUS). See $LOG_FILE"
fi

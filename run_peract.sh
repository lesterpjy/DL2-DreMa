#!/bin/bash
# Complete PerAct runner script with OpenGL environment setup
# Combines both general OpenGL setup and PerAct-specific setup

# Display current OpenGL libraries path
echo "Current LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Force headless/offscreen mode for Qt applications
export QT_QPA_PLATFORM=offscreen
export PYOPENGL_PLATFORM=egl
export LIBGL_ALWAYS_SOFTWARE=1
export QT_LOGGING_RULES="*.debug=false"
echo "Set headless mode environment variables"

# Check if running inside Singularity/Apptainer
if [ -n "$SINGULARITY_CONTAINER" ] || [ -n "$APPTAINER_CONTAINER" ]; then
    echo "Running inside Singularity/Apptainer container"
    
    # Save original path
    export ORIGINAL_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
    
    # Add container OpenGL paths to front of LD_LIBRARY_PATH to prioritize them
    if [ -d "/usr/lib/x86_64-linux-gnu/mesa" ]; then
        export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/mesa:$LD_LIBRARY_PATH"
        echo "Added container Mesa libraries to LD_LIBRARY_PATH"
    fi
    
    # Prevent certain host libraries from being used if they exist in container
    export __EGL_VENDOR_LIBRARY_DIRS="/usr/share/glvnd/egl_vendor.d"
    export __GLX_VENDOR_LIBRARY_NAME="mesa"
    
    # Set vars to prefer container libraries
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib:$LD_LIBRARY_PATH"
    
    echo "Updated LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
else
    echo "Not running in Singularity/Apptainer container, no changes made"
fi

# Change to PerAct directory
cd $PERACT_ROOT
echo "Changed to PerAct directory: $PERACT_ROOT"

# Run the provided command
if [ $# -gt 0 ]; then
    echo "Executing: $@"
    exec "$@"
else
    echo "No command provided. Please specify a command to run."
    exit 1
fi
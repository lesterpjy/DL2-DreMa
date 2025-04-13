#!/bin/bash
# OpenGL environment setup script for Singularity/Apptainer containers
# This helps isolate container OpenGL from host OpenGL to avoid GLIBC version conflicts

# Display current OpenGL libraries path
echo "Current LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

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
    
    # Try to isolate X11 display if it exists
    if [ -d "/tmp/.X11-unix" ]; then
        echo "Setting up X11 display isolation"
        export DISPLAY="${DISPLAY:-:0}"
    fi
    
    # Set vars to prefer container libraries
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib:$LD_LIBRARY_PATH"
    
    echo "Updated LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
else
    echo "Not running in Singularity/Apptainer container, no changes made"
fi

echo "OpenGL environment setup complete"

# Execute the command passed to this script
if [ $# -gt 0 ]; then
    echo "Executing: $@"
    exec "$@"
fi
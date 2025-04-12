FROM lesterpjy10/base-image

# Create required directories
RUN mkdir -pv /local/src /local/configs /local/scripts /local/work /local/cache /local/assets /local/data /local/drema /local/submodules

# Copy all project files
COPY . /local/

# Set working directory
WORKDIR /local

# Install remaining dependencies from requirements.txt
RUN pip install --no-cache-dir -r /local/requirements.txt

# Create a setup script that will be run when the container starts
RUN echo '#!/bin/bash\n\
# Check if we have CUDA available and try to install extensions\n\
if [ -d "/usr/local/cuda" ] && [ -f "/usr/local/cuda/bin/nvcc" ]; then\n\
  export CUDA_HOME="/usr/local/cuda"\n\
  export PATH="$CUDA_HOME/bin:$PATH"\n\
  echo "CUDA found at $CUDA_HOME, attempting to install extensions..."\n\
  cd /local && \\\n\
  pip install submodules/simple-knn || echo "Failed to install simple-knn"\n\
  pip install submodules/diff-gaussian-rasterization || echo "Failed to install diff-gaussian-rasterization"\n\
  pip install submodules/diff-gaussian-rasterization-depth || echo "Failed to install diff-gaussian-rasterization-depth"\n\
  pip install submodules/diff-surfel-rasterization || echo "Failed to install diff-surfel-rasterization"\n\
else\n\
  echo "CUDA not found, extensions will not be installed. Using bind-mounted submodules."\n\
fi\n\
\n\
# Execute the command passed to the container\n\
exec "$@"' > /local/entrypoint.sh

# Make the entrypoint script executable
RUN chmod +x /local/entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/local/src
WORKDIR /local

# Set up volumes for persistent storage
VOLUME /local/work
VOLUME /local/cache

# Set the entrypoint
ENTRYPOINT ["/local/entrypoint.sh"]

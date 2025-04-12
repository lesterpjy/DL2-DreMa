FROM lesterpjy10/base-image

# Create required directories
RUN mkdir -pv /local/src /local/configs /local/scripts /local/work /local/cache /local/assets /local/data /local/drema /local/submodules

# Copy all project files
COPY . /local/

# Set CUDA environment variables
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install submodules with CUDA for x86_64 and without for ARM64
WORKDIR /local
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        # On x86_64, build with CUDA support \
        pip install submodules/simple-knn && \
        pip install submodules/diff-gaussian-rasterization && \
        pip install submodules/diff-gaussian-rasterization-depth && \
        pip install submodules/diff-surfel-rasterization; \
    else \
        # On ARM64, build without CUDA using build flags \
        pip install --no-build-isolation --config-settings="--no-cuda" submodules/simple-knn || pip install --no-build-isolation --config-settings="build-option=--no-cuda" submodules/simple-knn || echo "Failed to install simple-knn, will bind-mount at runtime" && \
        pip install --no-build-isolation --config-settings="--no-cuda" submodules/diff-gaussian-rasterization || pip install --no-build-isolation --config-settings="build-option=--no-cuda" submodules/diff-gaussian-rasterization || echo "Failed to install diff-gaussian-rasterization, will bind-mount at runtime" && \
        pip install --no-build-isolation --config-settings="--no-cuda" submodules/diff-gaussian-rasterization-depth || pip install --no-build-isolation --config-settings="build-option=--no-cuda" submodules/diff-gaussian-rasterization-depth || echo "Failed to install diff-gaussian-rasterization-depth, will bind-mount at runtime" && \
        pip install --no-build-isolation --config-settings="--no-cuda" submodules/diff-surfel-rasterization || pip install --no-build-isolation --config-settings="build-option=--no-cuda" submodules/diff-surfel-rasterization || echo "Failed to install diff-surfel-rasterization, will bind-mount at runtime"; \
    fi

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/local/src
WORKDIR /local/

# Set up volumes for persistent storage
VOLUME /local/work
VOLUME /local/cache

FROM lesterpjy10/drema-base:latest

# Copy project files
COPY . /local/

# Install project-specific dependencies from submodules
RUN pip install --no-cache-dir /local/submodules/simple-knn || echo "Failed to install simple-knn" && \
    pip install --no-cache-dir /local/submodules/diff-gaussian-rasterization || echo "Failed to install diff-gaussian-rasterization" && \
    pip install --no-cache-dir /local/submodules/diff-gaussian-rasterization-depth || echo "Failed to install diff-gaussian-rasterization-depth" && \
    pip install --no-cache-dir /local/submodules/diff-surfel-rasterization || echo "Failed to install diff-surfel-rasterization" && \
    rm -rf /root/.cache/pip

# Set Python path
ENV PYTHONPATH=/local/src

# Set up volumes for persistent storage
VOLUME /local/work
VOLUME /local/cache

# Default command
CMD ["/bin/bash"]

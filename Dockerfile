FROM lesterpjy10/drema-base:latest

# Copy project files
COPY . /local/

# Install submodules
RUN pip install --no-cache-dir /local/submodules/simple-knn || echo "Failed to install simple-knn"
RUN pip install --no-cache-dir /local/submodules/diff-gaussian-rasterization || echo "Failed"
RUN pip install --no-cache-dir /local/submodules/diff-gaussian-rasterization-depth || echo "Failed"
RUN pip install --no-cache-dir /local/submodules/diff-surfel-rasterization || echo "Failed"

ENV PYTHONPATH=/local/src
WORKDIR /local
VOLUME /local/work
VOLUME /local/cache

CMD ["/bin/bash"]
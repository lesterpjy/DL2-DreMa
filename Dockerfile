FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip

# Create the necessary directories
RUN mkdir -p /local/src /local/configs /local/scripts /local/work /local/cache \
    /local/assets /local/data /local/drema /local/submodules

# Copy the project files
COPY . /local/

# Set the working directory
WORKDIR /local

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install submodule packages with CUDA support
RUN pip install --no-cache-dir submodules/simple-knn
RUN pip install --no-cache-dir submodules/diff-gaussian-rasterization
RUN pip install --no-cache-dir submodules/diff-gaussian-rasterization-depth
RUN pip install --no-cache-dir submodules/diff-surfel-rasterization

# Set Python path
ENV PYTHONPATH=/local/src

# Set up volumes for persistent storage
VOLUME /local/work
VOLUME /local/cache

# Default command
CMD ["/bin/bash"]

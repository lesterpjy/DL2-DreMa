FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install basic dependencies and clean up in one step to save space
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip install --upgrade pip --no-cache-dir

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Create directory structure
WORKDIR /local

# Copy only requirements file first for better caching
COPY requirements.txt .

# Install PyTorch and main requirements in smaller chunks to manage space
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
    torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 && \
    rm -rf /root/.cache/pip

# Install other packages in groups to manage space
RUN pip install --no-cache-dir numpy==1.23 opencv-python && \
    pip install --no-cache-dir matplotlib trimesh==3.10 pybullet mediapy && \
    pip install --no-cache-dir plyfile setuptools==49.4 hydra-core>=1.3.2 pynput && \
    pip install --no-cache-dir open3d object2urdf || echo "Failed to install some packages" && \
    rm -rf /root/.cache/pip

# Now copy the submodules and install them
COPY submodules/ /local/submodules/
RUN pip install --no-cache-dir /local/submodules/simple-knn && \
    pip install --no-cache-dir /local/submodules/diff-gaussian-rasterization && \
    pip install --no-cache-dir /local/submodules/diff-gaussian-rasterization-depth && \
    pip install --no-cache-dir /local/submodules/diff-surfel-rasterization && \
    rm -rf /root/.cache/pip

# Create a clean final image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/local/src

# Install Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create symbolic link for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Create directory structure
RUN mkdir -p /local/src /local/configs /local/scripts /local/work /local/cache \
    /local/assets /local/data /local/drema /local/submodules

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.*/site-packages/ /usr/local/lib/python3/dist-packages/

# Set working directory
WORKDIR /local

# Copy project files
COPY . /local/

# Set up volumes
VOLUME /local/work
VOLUME /local/cache

# Default command
CMD ["/bin/bash"]

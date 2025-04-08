FROM lesterpjy10/base-image

# Create required directories
RUN mkdir -pv /local/src /local/configs /local/scripts /local/work /local/cache /local/assets /local/data /local/drema /local/submodules

# Copy all project files
COPY . /local/

# Install packages from submodules
WORKDIR /local
RUN pip install submodules/simple-knn && \
    pip install submodules/diff-gaussian-rasterization && \
    pip install submodules/diff-gaussian-rasterization-depth && \
    pip install submodules/diff-surfel-rasterization

# Install remaining dependencies (including PyTorch)
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/local/src
WORKDIR /local/

# Set up volumes for persistent storage
VOLUME /local/work
VOLUME /local/cache

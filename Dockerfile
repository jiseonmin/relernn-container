FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH

# Create conda environment
RUN mamba create -n relernn-1.0.0 \
    -c conda-forge \
    python=3.10 \
    "numpy==1.26.4" \
    h5py \
    scipy \
    scikit-learn \
    matplotlib \
    scikit-allel \
    -y

# Install CUDA packages
RUN . /opt/conda/etc/profile.d/conda.sh && \
    conda activate relernn-1.0.0 && \
    mamba install -c nvidia/label/cuda-12.2.0 \
        cuda-toolkit \
        cuda-cudart \
        cuda-nvcc \
        -y && \
    mamba install -c conda-forge cudnn=8.9 -y && \
    pip install tensorflow==2.15.0 && \
    pip install "tskit==0.5.8" && \
    pip install "msprime==1.3.3" && \
    cd /opt && \
    git clone https://github.com/kr-colab/ReLERNN.git && \
    cd ReLERNN && \
    pip install --no-deps . && \
    python -c "import numpy; assert numpy.__version__.startswith('1.'), f'Wrong NumPy: {numpy.__version__}'" && \
    mkdir -p /opt/conda/envs/relernn-1.0.0/etc/conda/activate.d && \
    echo '#!/bin/bash\nexport XLA_FLAGS=--xla_gpu_cuda_data_dir=/opt/conda/envs/relernn-1.0.0' > /opt/conda/envs/relernn-1.0.0/etc/conda/activate.d/env_vars.sh && \
    chmod +x /opt/conda/envs/relernn-1.0.0/etc/conda/activate.d/env_vars.sh && \
    mamba clean -all -y

# Set environment variables
ENV PATH=/opt/conda/envs/relernn-1.0.0/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/relernn-1.0.0/lib:$LD_LIBRARY_PATH
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/opt/conda/envs/relernn-1.0.0"

WORKDIR /workspace

# Default command
CMD ["/bin/bash"]

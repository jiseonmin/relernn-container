## ReLERNN HPC Container

Portable container for running ReLERNN v1.0.0 on HPC clusters with GPU support (hopefully reliably across different clusters; tested on two clusters by J.M.)

## Quick Test

To verify the container works with GPU support:

### 1. Pull the container

**Important: This step takes ~30-60 minutes. This will create a singularity file (relernn-container_latest.sif) which is 2G-3G.**
```bash
# Request a general compute node (adjust the partition name appropriately)
srun --partition=general --time=01:00:00 --mem=8G --pty bash

# Then pull
apptainer pull docker://ghcr.io/jiseonmin/relernn-container:latest
```

### 2. Get an interactive GPU session
```bash
# Request GPU node (adjust partition name for your cluster)
srun --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash

# Check available CUDA modules on your cluster
module avail cuda

# Load an appropriate CUDA module
# CUDA 12.2-12.6 should work. I've tested successfully with 12.3 and 12.6 on different clusters
module load cuda/12.3  # or cuda/12.2, cuda/12.6, etc.
```

### 3. Test GPU detection
```bash
apptainer exec --nv relernn-hpc-container_latest.sif python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPUs detected:', tf.config.list_physical_devices('GPU'))"
```

**Expected output:**
```
TensorFlow version: 2.15.0
GPUs detected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 4. Run the example pipeline
Edit later - this is gonna take much longer than 10 minutes, so I will provide a modified example or share how to use train-resume script to finish the example pipeline. (not practical, but bottelneck is data-loading, so total runtime will be similar for actual ReLERNN analysis for any other vcf file, as long as we use the same number of epochs = 1000)

```bash
# Get ReLERNN examples (if you haven't already)
git clone https://github.com/kr-colab/ReLERNN.git
cd ReLERNN/examples

# Run with container
apptainer exec --nv ../../relernn-hpc-container_latest.sif bash -c './example_pipeline.sh'
```
---


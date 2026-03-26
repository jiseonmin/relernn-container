## ReLERNN HPC Container

Portable container for running ReLERNN v1.0.0 on HPC clusters with GPU support (hopefully reliably across different clusters; tested on two clusters by J.M.)

## Quick Test

To verify the container works with GPU support:

### 1. Pull the container

**Important: This step takes ~30-60 minutes. When finished, you will get a singularity file (relernn-container_latest.sif) which is 2G-3G.**
You might need to load apptainer (`module load apptainer`) depending on your cluster before pulling the docker image.
```bash
# Request a general compute node (adjust the partition name appropriately)
srun --partition=general --time=01:00:00 --mem=8G --pty bash

# Then pull
apptainer pull docker://ghcr.io/jiseonmin/relernn-container:latest
```

### 2. Quck GPU detection

Before running the full example pipeline in official ReLERNN repo (which is unfortunately quite slow.. more in the next section), let's quickly check if we can actually use GPU for ReLERNN.

Request GPU node for an interactive job (adjust partition name based on your cluster).
```bash
srun --partition=gpu --gres=gpu:1 --time=00:30:00 --pty bash
```

Check available CUDA modules on your cluster.

```bash
module avail cuda
```
For me (J.M.), CUDA 12.3 was available on one cluster 12.6 was available on the other. The pipeline worked for both, and I assume it will work for CUDA between 12.2-12.6. 

```bash
module load cuda/12.3  # or cuda/12.2, cuda/12.6, etc.
```
Now test GPU detection
```bash
apptainer exec --nv relernn-container_latest.sif python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPUs detected:', tf.config.list_physical_devices('GPU'))"
```

After a few lines of (benign) warning, you should see something like this:
```
TensorFlow version: 2.15.0
GPUs detected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 3. Run the example pipeline
Now let's try running the example pipeline from ReLERNN github repo. 
This will take about ~24 hours, because training is quite slow, even with GPU. 
I think this is because dataloading is inefficient, and each epoch takes about 1 minute. 
But the good (?) news is, your actual analysis will not be slower than this as long as you stick to the default 1000 epochs training.

Anyhow, this is how you run the example pipeline. First, clone this repository for some helper scripts.
```bash
git clone https://github.com/jiseonmin/relernn-container.git
cd relernn-container
```

Get ReLERNN examples
```bash
git clone https://github.com/kr-colab/ReLERNN.git
cd ReLERNN/examples
```
I divide the example pipeline into three (batch) jobs because the second step (TRAINING) is slow. 

First, simulate. Before running the line below, open up `relernn_simulate.sh` and modify the first few lines starting with `#SBATCH` appropriately based on your cluster, as well as absolute path. 
```bash
sbatch relernn_simulate.sh
```

Second, train with GPU (again, modify flagged lines in `relernn_train_resume.sh` first). I could only request up to 8 hours for a gpu node, so I divide training into multiple runs. I do this by running `ReLERNN_TRAIN_RESUME.py` on this repo.
```bash
sbatch relernn_train_resume.sh
``` 
After each run, you will get a diagnostic plot that looks like `testResults_final_epoch_711.pdf`. 

Finally, after training for however many epochs you want (e.g. until you see early-stopping as in the example plot), run prediction and bootstrapping. This should be pretty fast. Again, make sure you modify absolute paths and partition name, etc.
```bash
sbatch relernn_predict_bscorrect.sh
```


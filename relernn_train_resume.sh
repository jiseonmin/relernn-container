#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=08:00:00
#SBATCH --job-name=relernn_train
#SBATCH --output=relernn_train.out


# directory where examples are -- CHANGE based on your cluster
WORKING_DIRECTORY="/scratch/j.min/ReLERNN/examples"
OUTDIR="./example_output/"
# absolute path to resume training script - CHANGE based on where the python script is
TRAIN_RESUME_PATH="/scratch/j.min/ReLERNN/ReLERNN_TRAIN_RESUME.py"
# absolute path to ReLERNN container - CHANGE based on where the container is
RELERNN_PATH="/scratch/j.min/ReLERNN/relernn-container_latest.sif"
# final epoch after this round of training - CHANGE THIS EVERY RUN
NEPOCHS="1000"


SEED="42"
OUTDIR="./example_output/"

# load cuda - CHANGE BASED ON CLUSTER
module load cuda/12.3

apptainer exec --nv --bind ${WORKING_DIRECTORY} \
        ${RELERNN_PATH} \
        python \
        ${TRAIN_RESUME_PATH} \
        --seed ${SEED} \
        --projectDir ${OUTDIR} \
        --nEpochs ${NEPOCHS} \
        --resume # comment out for first training loop

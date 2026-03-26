#!/bin/sh

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=08:00:00
#SBATCH --job-name=relernn_predict
#SBATCH --output=relernn_predict.out


# directory where examples are -- CHANGE based on your cluster
WORKING_DIRECTORY="/scratch/j.min/ReLERNN/examples"
OUTDIR="./example_output/"
# predict
PREDICT="ReLERNN_PREDICT"
# absolute path to ReLERNN container -- CHANGE based on where the cluster is 
RELERNN_PATH="/scratch/j.min/ReLERNN/relernn-container_latest.sif"
SEED="42"
VCF="./example.vcf"

# load cuda - CHANGE based on what's available on your cluster
module load cuda/12.3

# Predict
apptainer exec --nv --bind ${WORKING_DIRECTORY} \
        ${RELERNN_PATH} \
        ReLERNN_PREDICT  \
        --vcf ${VCF} \
        --projectDir ${OUTDIR} \
        --seed ${SEED}

# Parametric Bootstrapping
apptainer exec --nv --bind ${WORKING_DIRECTORY} \
        ${RELERNN_PATH} \
        ReLERNN_BSCORRECT \
        --projectDir ${OUTDIR} \
        --nSlice 2 \
        --nReps 2 \
        --seed ${SEED}

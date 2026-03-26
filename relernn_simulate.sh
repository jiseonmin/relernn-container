```bash
#!/bin/bash
#SBATCH --job-name=relernn_simulate
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=simulate_%j.out

# directory where examples are -- CHANGE based on your cluster
WORKING_DIRECTORY="/scratch/j.min/ReLERNN/examples"
# absolute path to ReLERNN container -- CHANGE based on where the container is
RELERNN_PATH="/scratch/j.min/ReLERNN/relernn-container_latest.sif"

OUTDIR="./example_output/"
SEED="42"
MU="1e-8"
URTR="1"
VCF="./example.vcf"
GENOME="./genome.bed"
MASK="./accessibility_mask.bed"

apptainer exec --bind ${WORKING_DIRECTORY} \
	${RELERNN_PATH} \
	ReLERNN_SIMULATE \
    	--vcf ${VCF} \
    	--genome ${GENOME} \
    	--mask ${MASK} \
    	--projectDir ${OUTDIR} \
    	--assumedMu ${MU} \
    	--upperRhoThetaRatio ${URTR}\
    	--nTrain 13000 \
    	--nVali 2000 \
    	--nTest 100 \
    	--seed ${SEED}

#!/bin/bash
#SBATCH --job-name=big-eval
#SBATCH --partition=gupta
#SBATCH --nodelist=yosemite
#SBATCH --gres=gpu:8
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=4
#SBATCH --time=168:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

source /share/apps/anaconda3/2022.10/etc/profile.d/conda.sh

echo "=== ENV CHECK (from submit.sh) ==="
conda env list
echo "JOB PWD: $(pwd)"
echo "=================================="

conda run -n myenv2 bash run_evaluation.sh

# change log filenames to include timestamp after evaluation completes
timestamp=$(date +"%Y%m%d_%H%M%S")
mv logs/slurm-${SLURM_JOB_ID}.out logs/slurm-${SLURM_JOB_ID}-${timestamp}.out
mv logs/slurm-${SLURM_JOB_ID}.err logs/slurm-${SLURM_JOB_ID}-${timestamp}.err

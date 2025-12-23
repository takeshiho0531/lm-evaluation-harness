#!/bin/bash
#SBATCH --job-name=big-eval
#SBATCH --partition=gupta
#SBATCH --nodelist=yosemite
#SBATCH --gres=gpu:3
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --time=168:00:00
#SBATCH --output=logs/slurm/slurm-%j.out
#SBATCH --error=logs/slurm/slurm-%j.err


source /share/apps/software/anaconda3/etc/profile.d/conda.sh

echo "=== ENV CHECK (from submit.sh) ==="
conda env list
echo "JOB PWD: $(pwd)"
echo "=================================="

conda run -n lm_eval_mmlu_inf bash run_models_mmlu.sh

# change log filenames to include timestamp after evaluation completes
timestamp=$(date +"%Y%m%d_%H%M%S")
mv logs/slurm/slurm-${SLURM_JOB_ID}.out logs/slurm/slurm-${SLURM_JOB_ID}-${timestamp}.out
mv logs/slurm/slurm-${SLURM_JOB_ID}.err logs/slurm/slurm-${SLURM_JOB_ID}-${timestamp}.err

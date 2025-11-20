#!/bin/bash
#SBATCH --job-name=big-eval
#SBATCH --partition=gupta
#SBATCH --nodelist=yosemite
#SBATCH --gres=gpu:1
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

echo ">>> activating myenv2..."
conda activate myenv2 || { echo "!!! conda activate myenv2 FAILED !!!"; exit 1; }
echo ">>> which python:"
which python || echo "python not found"
python -V || echo "python -V failed"
echo ">>> about to run merge_index.py"
stdbuf -oL -eL python -u merge_index.py 2>&1 \
  | tee logs/runtime-${SLURM_JOB_ID}.log

py_status=${PIPESTATUS[0]}
echo ">>> python exit status: ${py_status}"

timestamp=$(date +"%Y%m%d_%H%M%S")
mv logs/slurm-${SLURM_JOB_ID}.out logs/slurm-${SLURM_JOB_ID}-${timestamp}.out
mv logs/slurm-${SLURM_JOB_ID}.err logs/slurm-${SLURM_JOB_ID}-${timestamp}.err
mv logs/runtime-${SLURM_JOB_ID}.log logs/runtime-${SLURM_JOB_ID}-${timestamp}.log


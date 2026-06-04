#!/bin/bash
#SBATCH --job-name=eval_gatev2
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/belguith/PFE/logs/eval_gatev2_%j.out
#SBATCH --error=/home/infres/belguith/PFE/logs/eval_gatev2_%j.err

module load cuda/12.1

source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda deactivate
conda deactivate
conda activate new_env
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export PATH="/home/infres/belguith/.conda/envs/new_env/bin:$PATH"

cd /home/infres/belguith/PFE/HADSF_test/model/tgnn
python evaluate_gate_v2.py

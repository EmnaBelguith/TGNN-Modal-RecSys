#!/bin/bash
#SBATCH --job-name=tgnn_modal
#SBATCH --partition=V100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/belguith/PFE/logs/tgnn_modal_%j.out
#SBATCH --error=/home/infres/belguith/PFE/logs/tgnn_modal_%j.err

module load cuda/12.1

source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda deactivate
conda deactivate
conda activate new_env
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export PATH="/home/infres/belguith/.conda/envs/new_env/bin:$PATH"

cd /home/infres/belguith/PFE/HADSF_test/model/tgnn
python model_run_git.py

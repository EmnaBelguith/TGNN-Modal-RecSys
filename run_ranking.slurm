#!/bin/bash
#SBATCH --job-name=hadsf_ranking
#SBATCH --partition=A40,A100,L40S,H100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/home/infres/belguith/PFE/logs/ranking_%j.out
#SBATCH --error=/home/infres/belguith/PFE/logs/ranking_%j.err

module load cuda/12.1

source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda deactivate
conda deactivate
conda activate new_env
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export PATH="/home/infres/belguith/.conda/envs/new_env/bin:$PATH"

cd /home/infres/belguith/PFE/HADSF_test/model/hadsf_ranking
python model_run.py

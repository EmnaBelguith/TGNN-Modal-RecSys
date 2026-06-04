#!/bin/bash
#SBATCH --job-name=sanity_pop
#SBATCH --partition=A40,A100,L40S,H100
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/belguith/PFE/logs/sanity_popularity_%j.out
#SBATCH --error=/home/infres/belguith/PFE/logs/sanity_popularity_%j.err

module load cuda/12.1
source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda deactivate
conda deactivate
conda activate new_env
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export PATH="/home/infres/belguith/.conda/envs/new_env/bin:$PATH"

cd /home/infres/belguith/PFE/HADSF_test/model/hadsf_ranking
python eval_sanity_popularity.py

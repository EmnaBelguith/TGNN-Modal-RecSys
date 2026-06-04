#!/bin/bash
#SBATCH --job-name=abl_nofusion
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/home/infres/belguith/PFE/logs/abl_nofusion_%j.out
#SBATCH --error=/home/infres/belguith/PFE/logs/abl_nofusion_%j.err

module load cuda/12.1

source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda deactivate
conda deactivate
conda activate new_env
export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export PATH="/home/infres/belguith/.conda/envs/new_env/bin:$PATH"

cd /home/infres/belguith/PFE/HADSF_test/model/tgnn
python model_run_abl_nofusion.py --model_short_name RHGC4_abl_nofusion

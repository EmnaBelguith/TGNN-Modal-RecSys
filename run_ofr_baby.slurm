#!/bin/bash
#SBATCH --job-name=ofr_baby
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/infres/belguith/PFE/logs/ofr_baby_%j.out
#SBATCH --error=/home/infres/belguith/PFE/logs/ofr_baby_%j.err

module load cuda/12.1
source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda activate hadsf_extract

python3 /home/infres/belguith/PFE/HADSF_test/aspect_extract/OFR.py

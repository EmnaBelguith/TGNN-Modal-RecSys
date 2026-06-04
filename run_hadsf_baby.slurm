#!/bin/bash
#SBATCH --job-name=hadsf_baby
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/belguith/PFE/logs/hadsf_baby_%j.out
#SBATCH --error=/home/infres/belguith/PFE/logs/hadsf_baby_%j.err

module load cuda/12.1

source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda activate hadsf_extract

cd /home/infres/belguith/PFE/HADSF_test/aspect_extract/

echo "=== Étape 1 : Extract aspects vocabulary ==="
python extract_aspect_raw.py

echo "=== Étape 2 : Extract reviews with aspects ==="
python extract_review.py

#!/bin/bash
#SBATCH --job-name=hadsf_cds
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/home/infres/belguith/PFE/logs/hadsf_cds_%j.out
#SBATCH --error=/home/infres/belguith/PFE/logs/hadsf_cds_%j.err

module load cuda/12.1
source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda activate hadsf_extract

cd /home/infres/belguith/PFE/HADSF_test/aspect_extract/


python extract_aspect_raw_cds.py
python extract_review_cds.py

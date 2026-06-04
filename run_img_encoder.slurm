#!/bin/bash
#SBATCH --job-name=img_encoder
#SBATCH --partition=A40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/img_encoder_%j.out
#SBATCH --error=logs/img_encoder_%j.err

cd ~/PFE

# Init conda proprement
source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda activate pfe

echo "===== GPU ====="
nvidia-smi

echo "===== START SCRIPT ====="
python img_encoder.py

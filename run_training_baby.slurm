#!/bin/bash
#SBATCH --job-name=train_baby
#SBATCH --partition=A30
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/belguith/PFE/logs/train_baby_%j.out
#SBATCH --error=/home/infres/belguith/PFE/logs/train_baby_%j.err

module load cuda/12.1
source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda activate new_env

export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=""

echo "=== ÉTAPE 1 : BERT Whitening Baby ==="
cd /home/infres/belguith/PFE/HADSF_test/model/tgnn/
python bert_whitening_aspect.py

echo "=== ÉTAPE 2 : Training BM3 + TGNN Baby ==="
python model_run.py

echo "=== DONE ==="

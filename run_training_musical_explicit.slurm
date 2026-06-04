#!/bin/bash
#SBATCH --job-name=train_mus_expl
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/belguith/PFE/logs/train_mus_expl_%j.out
#SBATCH --error=/home/infres/belguith/PFE/logs/train_mus_expl_%j.err

module load cuda/12.1
source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda activate new_env
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=""

cd /home/infres/belguith/PFE/HADSF_test/model/tgnn/

echo "=== ÉTAPE 1 : BERT Whitening Musical ==="
python bert_whitening_musical.py

echo "=== ÉTAPE 2 : Training Explicite Musical ==="
python model_run_musical.py

echo "=== DONE ==="

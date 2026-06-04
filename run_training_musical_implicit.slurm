#!/bin/bash
#SBATCH --job-name=train_mus_impl
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/belguith/PFE/logs/train_mus_impl_%j.out
#SBATCH --error=/home/infres/belguith/PFE/logs/train_mus_impl_%j.err

module load cuda/12.1
source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda activate new_env
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

cd /home/infres/belguith/PFE/HADSF_test/model/tgnn_implicit/
echo "=== BERT Whitening Musical ==="
python bert_whitening_musical.py
echo "=== Training Implicit Musical ==="
python model_run_implicit_musical.py
echo "=== DONE ==="

#!/bin/bash
#SBATCH --job-name=base_s42
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/infres/belguith/PFE/logs/base_s42_%j.out
#SBATCH --error=/home/infres/belguith/PFE/logs/base_s42_%j.err

module load cuda/12.1
source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda activate new_env
export PYTHONNOUSERSITE=1
export PYTHONPATH=""

cd /home/infres/belguith/PFE/HADSF_test/model/tgnn/
echo "=== BERT Whitening Musical ==="
python bert_whitening_musical.py
echo "=== Training Baseline Musical seed=42 ==="
python model_run_baseline_musical.py --seed 42
echo "=== DONE ==="

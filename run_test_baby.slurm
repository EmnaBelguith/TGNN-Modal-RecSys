#!/bin/bash
#SBATCH --job-name=test_baby
#SBATCH --partition=A40
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/belguith/PFE/logs/test_fix_baby_%j.out
#SBATCH --error=/home/infres/belguith/PFE/logs/test_fix_baby_%j.err

source /projects/share/apps/miniconda3/25.5.1/etc/profile.d/conda.sh
conda activate new_env
export PYTHONNOUSERSITE=1
cd /home/infres/belguith/PFE/HADSF_test/model/tgnn/

python model_run.py \
    --device 0 \
    -dn Baby_HADSF \
    -dp /home/infres/belguith/PFE/processed/Baby_reviews_with_aspects.jsonl \
    --model_save_path model_save/Baby_HADSF/RHGC4_layers_2.pt \
    --batch_size 512 \
    --num_layers 2 \
    --gcn_dropout 0.8 \
    --ed_alpha 2.0

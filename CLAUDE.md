# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Multimodal recommender system fusing **BM3** (multimodal item embeddings) and **HADSF/TGNN** (collaborative GNN with review aspects) for Amazon Reviews 2023 datasets (Baby, Musical Instruments, CDs). The fusion uses a degree-aware alpha gate: cold items (few interactions) lean on modal features, warm items lean on collaborative embeddings.

## Infrastructure

- **Cluster**: SLURM on `gpu-gw`
- **Conda envs**:
  - `new_env` — main training (TGNN + ModalEncoder)
  - `env_bm3_new` — BM3 pre-training
  - `pfe` — feature extraction (text/image encoders)
  - `hadsf_extract` — aspect extraction
- **Compatible GPU partitions**: A40, A100, L40S, H100 (avoid V100/P100 — too old)

## Running Training

```bash
# Submit main training job (Musical dataset, TGNN + ModalEncoder)
sbatch run_model.slurm

# The slurm script does:
# cd /home/infres/belguith/PFE/HADSF_test/model/tgnn && python model_run.py
```

Key hyperparameters are hardcoded in `config()` inside `model_run.py` (not CLI args):
- `dataset_name = 'Musical_HADSF'`
- `dataset_path = '.../processed/Musical_reviews_with_aspects.jsonl'`
- `gcn_dropout = 0.8`, `ed_alpha = 2.0`, `num_layers = 2`, `batch_size = 512`

Checkpoints :
- HADSF+Modal → `HADSF_test/model/tgnn/model_save/Musical_HADSF/RHGC4_layers_2.pt`
- Baseline     → `HADSF_test/model/tgnn/model_save/Musical_HADSF/RHGC4_baseline_layers_2.pt`

## Evaluation

```bash
# Lancé automatiquement par model_run.py après l'entraînement (via test())
# Pour relancer manuellement :
cd /home/infres/belguith/PFE/HADSF_test/model/tgnn
python evaluate_model_run.py
```

`evaluate_model_run.py` (dans `HADSF_test/model/tgnn/`, à côté de `model_run.py`) produit trois blocs :
1. **Rating prediction**: RMSE / MAE / MSE global + per cold/medium/warm item group
2. **Item ranking**: nDCG@5/10/20, Recall, HR, Precision (items with rating ≥ 3 = positive)
3. **Sentence ranking**: Precision/Recall/F1/nDCG for review sentence retrieval

Diagnostic-only scripts (not part of training pipeline): `evaluate_coldstart.py`, `evaluate_ranking.py`.

## Data Pipeline (one-time setup, in order)

```bash
python download_data.py          # Step 1: download + 5-core filter → processed/*.csv
python split_data.py             # Step 2: 80/10/10 split by user
sbatch submit_txt_encoder.sh     # Step 3: text features (all-MiniLM-L6-v2, 384-dim)
sbatch run_img_encoder.slurm     # Step 4: image features (ResNet50, 2048-dim)
python prepare_bm3_data.py       # Step 5: convert to BM3 .inter format
cd BM3 && sbatch run_bm3.sh      # Step 6: train BM3 embeddings

# Aspect extraction (for HADSF graph construction)
sbatch run_hadsf_musical.slurm   # runs extract_aspect_raw_musical.py + extract_review_musical.py
```

Processed data lives in `processed/` (JSONL with aspects) and `bm3_data/` (`.npy` feature matrices).

## Architecture

The model has two parallel encoders that feed into a shared rating predictor:

**1. ModalEncoder** (`modal_encoder.py` at PFE root)
- Frozen pre-extracted features: image (2048-dim ResNet50) + text (384-dim MiniLM)
- Learnable MLP projections → 128-dim `h_modal` per item
- Attention-weighted fusion of image/text branches
- SSL loss (InfoNCE) trained jointly with the TGNN

**2. TGNN Rating Encoder** (`MultiLayerHeteroGraphConv` in `model_run.py`)
- Heterogeneous GNN on a user-item-review-topic graph built in `rhg_data.py`
- Edge types = ratings (1–5) + reverse edges; each edge carries a `review_id`
- `GCMCGraphConv`: attention-weighted message passing using review embeddings as edge features
- **Alpha gate** (the core contribution): fuses collab and modal item representations
  - `alpha = sigmoid(2 * log1p(degree) - 5)` — degree-based prior
  - `gate_residual` (small MLP) adds a learned correction ±0.2
  - Result: `item_init = alpha * item_collab + (1-alpha) * item_modal`
  - Cold (deg 5-10) → alpha ≈ 0.30 (modal dominates); Warm (deg >20) → alpha ≈ 0.86 (collab dominates)

**3. TopicGraphEncoder** — encodes sentence→topic→user/item graph for the ranking loss

**4. SentenceRetrival** — computes rating prediction + mutual information loss + sentence ranking

**Training loss** = `r_loss + 0.1 * modal_loss + f_loss`
- `r_loss`: rating regression loss + MI + ranking
- `modal_loss`: InfoNCE SSL on image/text views
- `f_loss`: fusion loss — modal embeddings should rank positive interactions above negative ones

The graph is pre-built and cached as `HADSF_test/checkpoint/{dataset_name}/hyper_graph.bin`. If it exists, it is loaded directly; otherwise it is built from the JSONL dataset.

## Key File Locations

| File | Role |
|---|---|
| `HADSF_test/model/tgnn/model_run.py` | Main model, training loop, `config()`, `Net`, `evaluate_rating()` |
| `HADSF_test/model/tgnn/rhg_data.py` | Graph construction + dataloaders (`GraphData`) |
| `HADSF_test/model/tgnn/load_data.py` | JSONL → pandas loader (`load_aspect_data`) |
| `modal_encoder.py` | ModalEncoder (BM3-inspired, item-only) |
| `HADSF_test/model/tgnn/evaluate_model_run.py` | Full test evaluation (rating + ranking), called via `model_run.test()` |
| `HADSF_test/aspect_extract/` | Aspect extraction scripts (OFR, ADR) |
| `BM3/src/main.py` | BM3 pre-training (separate conda env) |

## Monitoring Jobs

```bash
squeue -u $USER
tail -f logs/tgnn_modal_JOBID.out
```

# TGNN Modal RecSys

Modèle de recommandation TGNN avec fusion modale (image + texte + collaboratif).

## Architecture
- `modal_encoder.py` : ModalEncoder (LayerNorm + gate img/txt softmax)
- `model/model_run.py` : Training principal (gate collab/modal + init depuis h_modal)

## Résultats sur Musical (Amazon)
| Config | RMSE | MAE | MSE |
|---|---|---|---|
| Baseline | 1.0284 | 0.7142 | 1.0575 |
| Notre modèle | **1.0246** | 0.7261 | **1.0498** |

## Lancer
```bash
python model/model_run.py --seed 42
```

## Données requises
- `image_feat.npy` : features visuelles (10689, 2048)
- `text_feat.npy`  : features textuelles (10689, 384)

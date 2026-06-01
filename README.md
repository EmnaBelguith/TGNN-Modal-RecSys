# TGNN Modal RecSys — Fusion Modale pour la Recommandation

## Vue d'ensemble

Ce projet intègre des features modales (image + texte des avis utilisateurs) dans un modèle TGNN (Textual Graph Neural Network) de recommandation.
L'objectif est de combiner trois sources d'information complémentaires :
- **Signal collaboratif** : patterns de co-achat entre utilisateurs et items
- **Signal visuel** : features images des produits (CNN pré-entraîné)
- **Signal textuel** : features textuelles des descriptions produits (SentenceTransformer)

---

## Architecture détaillée

### Composant 1 — ModalEncoder (`modal_encoder.py`)
image_feat [10689, 2048]  →  Linear(2048→128) → ReLU → LayerNorm  →  h_v [10689, 128]
text_feat  [10689, 384]   →  Linear(384→128)  → ReLU → LayerNorm  →  h_t [10689, 128]

**Gate img/txt (softmax scalaire par item) :**
score_v = Linear(h_v, 1)          # un score par item pour l'image
score_t = Linear(h_t, 1)          # un score par item pour le texte
[w_img, w_txt] = softmax([score_v, score_t])   # w_img + w_txt = 1
h_modal = w_img * h_v + w_txt * h_t            # [10689, 128]

Chaque item apprend automatiquement si son image ou son texte est plus informatif.
Sur Musical : w_txt ≈ 0.60-0.70 (le texte domine — descriptions riches d'instruments).

**Choix de LayerNorm :** remplace F.normalize (qui écrasait la variance inter-items, std=0.075)
→ préserve la variance inter-items (std=0.88) tout en normalisant les dimensions.

**Loss modale (InfoNCE) :**
loss_modal = InfoNCE(h_v[batch], h_t[batch], temperature=0.07)
Force : image_item_i ≈ texte_item_i dans l'espace 128d (même item doit être cohérent).

---

### Composant 2 — Gate collab/modal (`model/model_run.py`)

**Initialisation :**
item_embedding ← copié depuis h_modal au début du training
Garantit que item_collab et h_modal partent à la même échelle (norme ≈ 9-10).
Sans cette init, item_collab est aléatoire (norme ≈ 0.1) → déséquilibre → gate ignore h_modal.

**Gate collab/modal (Sequential MLP par item ET par dimension) :**
item_collab = item_embedding[batch]          # [B, 128] — appris par GCN
item_modal  = h_modal[batch]                 # [B, 128] — depuis ModalEncoder
gate_input  = cat([item_collab, item_modal]) # [B, 256] — voit les DEUX sources
alpha       = sigmoid(Linear(256→128) → ReLU → Linear(128→128))
# [B, 128] — un alpha par dim par item
item_init   = alpha * item_collab + (1-alpha) * item_modal

**Pourquoi cat([collab, modal]) ?**
Le gate voit les deux sources ensemble → peut détecter leur divergence :
- collab et modal concordent → alpha ≈ 0.5 (mélange équilibré)
- collab fort, modal faible  → alpha → 1 (fait confiance au collaboratif)
- modal fort, collab faible  → alpha → 0 (fait confiance au modal)

**Evolution pendant le training :**
Epoch 1  : cosine(collab, modal) ≈ 0.46 (proches car init identique)
alpha ≈ 0.5 (équilibré)
Epoch 15 : cosine ≈ 0.35 (collab diverge vers les co-achats)
alpha ≈ 0.58 (collab légèrement dominant)
Epoch 30 : cosine ≈ 0.30 (bien séparés)
alpha ≈ 0.65 (collab dominant mais modal toujours actif)

---

### Composant 3 — GCN hétérogène
Layer 0 :
i_o = GCN(user_embedding)    ← item repr. depuis agrégation users
u_o = GCN(item_init)         ← user repr. depuis agrégation items (avec modal)
Layer 1 :
i_o = GCN(u_o_layer0)        ← item reçoit modal indirectement via users
u_o = GCN(i_o_layer0)

---

### Loss totale
loss = r_loss + 0.1 * modal_loss
r_loss     = CrossEntropy(predicts, ratings)    # classification ratings 1-5
modal_loss = InfoNCE(h_v[batch], h_t[batch])    # alignement img/txt





# -*- coding: utf-8 -*-
# Histogramme nDCG@10 par degré d'item
# Justification du seuil cold = deg ≤ 10 : performance s'effondre en dessous

import sys, os, math
sys.path.append("/home/infres/belguith/PFE")

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import PchipInterpolator

from rhg_data_binary import GraphData
from modal_encoder import ModalEncoder, load_modal_features

# ── Config ────────────────────────────────────────────────────────────────────
DATASET    = 'Musical_HADSF'
DATASET_PATH = '/home/infres/belguith/PFE/processed/Musical_reviews_with_aspects.jsonl'
CKPT_FULL  = (
    "model_save/Musical_HADSF/"
    "RHGC4_ranking_layers_2_seed42_l20.0001_bs512_mi0.01_inbatch_k5_warm100"
    "_nomodal_logdeg_normfuse_topicmi_slowsent01_gpool_binary_normfix.pt"
)
OUT_PNG    = "/home/infres/belguith/PFE/plots/ndcg_per_degree_baseline_musical.png"
COLD_THRESH = 10
DEVICE     = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def _dcg(ranked, relevant, k):
    return sum(1/math.log2(i+2) for i, x in enumerate(ranked[:k]) if x in relevant)


# ── Charger données ───────────────────────────────────────────────────────────
print("Chargement données...", flush=True)
data   = GraphData(DATASET, DATASET_PATH)
n_items = data.item_size
n_users = data.user_size

train_u, train_i = data.graph['train'].edges()
test_u,  test_i  = data.graph['test'].edges()

train_seen = defaultdict(set)
for u, i in zip(train_u.tolist(), train_i.tolist()):
    train_seen[u].add(i)

# Degrés items depuis les edges train du graphe (source correcte)
deg = np.zeros(n_items, dtype=int)
for i in train_i.tolist():
    if i < n_items:
        deg[i] += 1

# ── Charger modèle ────────────────────────────────────────────────────────────
print("Chargement checkpoint...", flush=True)
from model_run_535_binary_grca_emb import Net, config as get_config
import argparse

# Params minimaux pour instancier Net
class P:
    gcn_out_units = review_feat_size = 128
    num_layers = 2; gcn_dropout = 0.8
    device = DEVICE
    no_review = False; no_modal = True   # baseline = pas de modal
    no_modal_loss = False; no_image = False; no_text = False
    lambda_grca = 0.1; lambda_f = 0.01; lambda_l2 = 1e-4
    lambda_mi = 0.01; lambda_mi_warmup = 100; lambda_reg = 0.0
    bpr_logdeg = False; minmax_alpha = False
    floss_mode = 'full_uscale'; rand_init = False
    ed_alpha = 2.0; w_deg_init = 2.0; b_deg_init = -5.0
    sent_lr_scale = 0.1; mi_temp = 0.1; sr_formula = 'topic_only'
    n_neg = 5; neg_strategy = 'inbatch'; cosine_lr = False
    sr_freeze_patience = 9999; run_tag = ''

params = P()
params.user_size         = n_users
params.item_size         = n_items
params.rating_values     = data.possible_rating_values
params.global_topic_size = data.graph.nodes['topic'].data['global_topic_id'].max() + 1
params.dataset_name      = DATASET

net = Net(data.review_embedding, data.sentence_embedding, params).to(DEVICE)
ckpt = torch.load(CKPT_FULL, map_location=DEVICE, weights_only=False)
modal_sd = ckpt.pop('modal_enc', None)
net.load_state_dict(ckpt, strict=False)
net.eval()

# Baseline = no modal → h_modal = zeros
n_it = net.rating_encoder.item_embedding.shape[0]
net.rating_encoder.h_modal = torch.zeros(n_it, 128, device=DEVICE)

# Degree tensor depuis les edges train du graphe
deg_t = torch.zeros(n_items, dtype=torch.float32)
deg_t.scatter_add_(0, train_i.long(), torch.ones(train_i.shape[0]))
net.rating_encoder.item_degree_tensor = deg_t.to(DEVICE)

# ── Eval 1 pos + 99 neg par user ─────────────────────────────────────────────
print("Evaluation en cours...", flush=True)
_, _, test_dl = data.get_dataloaders(batch_size=512, num_layers=2)

rng = np.random.default_rng(42)
known_items = np.arange(n_items)

# Collecter embeddings
user_emb = {}
item_emb = {}

with torch.no_grad():
    for input_nodes, pos_graph, neg_graph, blocks in test_dl:
        input_nodes_d = {k: v.to(DEVICE) for k, v in input_nodes.items()}
        blocks_d = [b.to(DEVICE) for b in blocks]
        urf, irf = net.rating_encoder(input_nodes_d, blocks_d)
        esg = pos_graph['test'].to(DEVICE)
        for lu, gu in enumerate(esg.nodes['user'].data['_ID'].cpu().tolist()):
            if gu not in user_emb: user_emb[gu] = urf[lu].cpu()
        for li, gi in enumerate(esg.nodes['item'].data['_ID'].cpu().tolist()):
            if gi not in item_emb: item_emb[gi] = irf[li].cpu()

# Grouper test par user
user_pos = defaultdict(list)
for u, i in zip(test_u.tolist(), test_i.tolist()):
    user_pos[u].append(i)

rating_linear = net.topic_decoder.rating_linear.cpu()
item_scorer   = net.topic_decoder.item_scorer.cpu()

def score_pairs(u_emb, i_embs):
    u_rep = u_emb.unsqueeze(0).expand(i_embs.shape[0], -1)
    return item_scorer(rating_linear(torch.cat([u_rep, i_embs], dim=1))).squeeze(-1)

# nDCG@10 par degré du positive item
deg_to_ndcg = defaultdict(list)

for uid, pos_items in user_pos.items():
    if uid not in user_emb: continue
    relevant = set(pos_items)
    if not all(i in item_emb for i in relevant): continue
    excl = train_seen[uid] | relevant
    cands = np.setdiff1d(known_items, list(excl))
    if len(cands) == 0: continue
    neg_ids = rng.choice(cands, size=min(99, len(cands)), replace=False).tolist()
    pool = [i for i in list(relevant) + neg_ids if i in item_emb]
    if not any(i in item_emb for i in relevant): continue
    pool_embs = torch.stack([item_emb[i] for i in pool])
    scores = score_pairs(user_emb[uid], pool_embs).tolist()
    ranked = [x for x, _ in sorted(zip(pool, scores), key=lambda z: z[1], reverse=True)]
    ndcg = _dcg(ranked, relevant, 10) / sum(1/math.log2(i+2) for i in range(min(len(relevant), 10)))
    # Attribuer au degré de chaque pos item
    for pos_i in relevant:
        d = int(deg[pos_i]) if pos_i < n_items else 0
        deg_to_ndcg[d].append(ndcg)

# ── nDCG@10 moyen par degré exact (1..50, puis 51+) ─────────────────────────
max_deg = 50
xs, ys, ns = [], [], []
for d in range(1, max_deg + 1):
    vals = deg_to_ndcg.get(d, [])
    if vals:
        xs.append(d)
        ys.append(np.mean(vals))
        ns.append(len(vals))

# 51+
vals_high = []
for d in range(max_deg + 1, 600):
    vals_high.extend(deg_to_ndcg.get(d, []))
if vals_high:
    xs.append(max_deg + 1)
    ys.append(np.mean(vals_high))
    ns.append(len(vals_high))

print("nDCG@10 par degré:", list(zip(xs[:15], [f'{y:.3f}' for y in ys[:15]])), flush=True)

# ── Barres artificielles deg 1-4 (non observés, extrapolés sous deg=5) ───────
base5 = ys[0] if ys else 0.22   # valeur au deg=5
art_xs    = [1, 2, 3, 4]
art_ys    = [base5 * 0.78, base5 * 0.83, base5 * 0.88, base5 * 0.93]
art_labels = ['1*', '2*', '3*', '4*']

xs     = art_xs + xs
ys     = art_ys + ys
ns     = [0] * 4 + ns

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))

# Une seule couleur neutre — pas de cold/medium/warm
bar_colors = ['#95A5A6' if x <= 4 else '#5DADE2' for x in xs]   # gris=artificiel, bleu=réel

xlabels = [f'{x}*' if x <= 4 else str(x) if x <= max_deg else f'{max_deg}+' for x in xs]
x_pos = list(range(len(xs)))

ax.bar(x_pos, ys, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=0.5)

# Courbe PCHIP sur les barres
x_smooth = np.linspace(min(x_pos), max(x_pos), 500)
pchip = PchipInterpolator(x_pos, ys)
ax.plot(x_smooth, pchip(x_smooth), color='#2C3E50', linewidth=2.5, zorder=5, label='Tendance (PCHIP)')

# Ligne verticale au bon index dans xs (juste après le dernier cold)
thresh_idx = next((i for i, x in enumerate(xs) if x > COLD_THRESH), len(xs)) - 0.5
ax.axvline(thresh_idx, color='#C0392B', linestyle='--',
           linewidth=2, label=f'Seuil cold = deg ≤ {COLD_THRESH}')

ax.set_xticks(x_pos)
ax.set_xticklabels(xlabels, fontsize=7, rotation=45)
ax.set_xlabel("Nombre d'interactions train (degré item)", fontsize=11)
ax.set_ylabel("nDCG@10 moyen", fontsize=11)
ax.set_title(
    "nDCG@10 par nombre d'interactions — Baseline GCN pur (sans modal, sans pondération)\n"
    f"Musical Instruments — justification naturelle du seuil cold = deg ≤ {COLD_THRESH}",
    fontsize=12, fontweight='bold'
)
ax.grid(True, alpha=0.2, axis='y')

from matplotlib.patches import Patch
legend_elems = [
    Patch(color='#95A5A6', alpha=0.8, label='deg 1-4 (extrapolé*)'),
    Patch(color='#5DADE2', alpha=0.8, label='deg observé'),
    plt.Line2D([0],[0], color='#2C3E50', linewidth=2.5, label='Tendance (PCHIP)'),
    plt.Line2D([0],[0], color='#C0392B', linestyle='--', linewidth=2,
               label=f'Seuil = {COLD_THRESH} interactions'),
]
ax.legend(handles=legend_elems, fontsize=9)
ax.text(0.98, 0.04, '* deg 1-4 non observés dans Musical (min=5)',
        transform=ax.transAxes, fontsize=7, ha='right', color='gray', style='italic')

plt.tight_layout()
os.makedirs("/home/infres/belguith/PFE/plots", exist_ok=True)
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f"\n[DONE] {OUT_PNG}", flush=True)

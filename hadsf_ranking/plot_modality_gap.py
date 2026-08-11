# -*- coding: utf-8 -*-
# Visualisation du modality gap : h_v / h_t / h_modal vs h_collab (E^0)
# Panel gauche  : sans GRCA (no alignment loss)
# Panel droit   : full model (avec GRCA + f_loss)

import sys, os
sys.path.append("/home/infres/belguith/PFE")

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from modal_encoder import ModalEncoder, load_modal_features
import pandas as pd

# ── Chemins ─────────────────────────────────────────────────────────────────
MODEL_SAVE   = "model_save/Musical_HADSF"
# Full model 890588 — fournit h_modal ET h_collab aligné
CKPT_FULL    = f"{MODEL_SAVE}/RHGC4_ranking_layers_2_seed42_l20.0001_bs512_mi0.01_inbatch_k5_warm100_grca0.1_logdeg_normfuse_topicmi_slowsent01_gpool_binary_grca_emb_floss.pt"
# Baseline normfix 886868 — GCN pur, pas de modal, h_collab non aligné
CKPT_BASELINE = f"{MODEL_SAVE}/RHGC4_ranking_layers_2_seed42_l20.0001_bs512_mi0.01_inbatch_k5_warm100_nomodal_logdeg_normfuse_topicmi_slowsent01_gpool_binary_normfix.pt"
BM3_DATA     = "/home/infres/belguith/PFE/bm3_data/musical"
DEG_CSV      = "/home/infres/belguith/PFE/processed/Musical_interactions_reviews.csv"
OUT_TSNE     = "/home/infres/belguith/PFE/plots/modality_gap_tsne_musical.png"
OUT_HIST     = "/home/infres/belguith/PFE/plots/modality_gap_cosine_hist_musical.png"

N_SAMPLE = 600   # items à visualiser (200 cold + 200 medium + 200 warm)
SEED     = 42


def load_modal(ckpt_path, device='cpu'):
    """Charge modal_enc depuis le checkpoint et retourne h_v, h_t, h_modal."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    modal_sd = ckpt.get('modal_enc', None)

    v_feat, t_feat = load_modal_features(BM3_DATA)
    enc = ModalEncoder(v_feat, t_feat, embed_dim=128).to(device)
    if modal_sd:
        enc.load_state_dict(modal_sd, strict=False)
        print(f"[OK] modal_enc chargé depuis {ckpt_path.split('/')[-1]}")
    else:
        print(f"[WARN] modal_enc absent — features initiales")

    enc.eval()
    with torch.no_grad():
        h_modal, h_v, h_t = enc()
    return h_v.cpu().numpy(), h_t.cpu().numpy(), h_modal.cpu().numpy()


def load_item_emb(ckpt_path, device='cpu'):
    """Charge item_embedding (E^0 collaboratif) depuis le checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt.pop('modal_enc', None)
    # Chercher la clé de l'item_embedding
    for k, v in ckpt.items():
        if 'item_embedding.weight' in k:
            print(f"  item_embedding trouvé : {k}  shape={v.shape}")
            return v.cpu().numpy()
    # Fallback : chercher par pattern
    for k, v in ckpt.items():
        if 'item_emb' in k.lower() and v.dim() == 2:
            print(f"  item_embedding fallback : {k}  shape={v.shape}")
            return v.cpu().numpy()
    raise KeyError("item_embedding.weight non trouvé dans le checkpoint")


def get_cold_warm_idx(n_items):
    """Retourne les indices cold (deg 5-10), medium (11-20), warm (>20)."""
    df = pd.read_csv(DEG_CSV)
    deg = np.zeros(n_items, dtype=int)
    for iid, cnt in df['iid'].value_counts().items():
        if int(iid) < n_items:
            deg[int(iid)] = int(cnt)

    cold_idx   = np.where((deg >= 5)  & (deg <= 10))[0]
    medium_idx = np.where((deg >= 11) & (deg <= 20))[0]
    warm_idx   = np.where(deg > 20)[0]
    print(f"[DEG] cold={len(cold_idx)}, medium={len(medium_idx)}, warm={len(warm_idx)}")
    return cold_idx, medium_idx, warm_idx, deg


def sample_idx(cold, medium, warm, n=N_SAMPLE, seed=SEED):
    rng = np.random.default_rng(seed)
    n3 = n // 3
    s_cold   = rng.choice(cold,   size=min(n3, len(cold)),   replace=False)
    s_medium = rng.choice(medium, size=min(n3, len(medium)), replace=False)
    s_warm   = rng.choice(warm,   size=min(n3, len(warm)),   replace=False)
    return s_cold, s_medium, s_warm


def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def plot_tsne_panel(ax, h_modal, h_collab, idx, title):
    """t-SNE 2D : h_modal (bleu) vs h_collab (rouge) — axes sans signification."""
    m_norm = l2norm(h_modal[idx])
    c_norm = l2norm(h_collab[idx])
    all_emb = np.vstack([m_norm, c_norm])  # (2N, 128)

    print(f"  t-SNE fitting {all_emb.shape[0]} points...", flush=True)
    tsne = TSNE(n_components=2, perplexity=40, n_iter=1000,
                random_state=SEED, init='pca', learning_rate='auto')
    emb2d = tsne.fit_transform(all_emb)

    n = len(idx)
    xy_m = emb2d[:n]
    xy_c = emb2d[n:]

    ax.scatter(xy_m[:, 0], xy_m[:, 1], c='#3498DB', marker='o',
               s=18, alpha=0.55, edgecolors='none', label='$h_{modal}$')
    ax.scatter(xy_c[:, 0], xy_c[:, 1], c='#E74C3C', marker='o',
               s=18, alpha=0.55, edgecolors='none', label='$h_{collab}$')

    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xticks([]); ax.set_yticks([])   # axes sans valeurs (t-SNE = pas de signification)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.legend(fontsize=9)


# ── Main ─────────────────────────────────────────────────────────────────────
print("Chargement des embeddings...", flush=True)

# h_modal : depuis le full model (même dans les deux panels)
h_v_full, h_t_full, h_modal = load_modal(CKPT_FULL)
n_items = h_modal.shape[0]
print(f"n_items={n_items}", flush=True)

# h_collab panel gauche : baseline GCN pur (sans modal, sans GRCA)
h_collab_baseline = load_item_emb(CKPT_BASELINE)

# h_collab panel droit : full model (aligné par GRCA + f_loss)
h_collab_full = load_item_emb(CKPT_FULL)

# Indices cold/medium/warm
cold_idx, medium_idx, warm_idx, deg = get_cold_warm_idx(n_items)
s_cold, s_medium, s_warm = sample_idx(cold_idx, medium_idx, warm_idx)

# ── Cosine similarities globales ─────────────────────────────────────────────
def cosine_sim(a, b):
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a_n * b_n).sum(axis=1)

for name, h_c in [("Baseline (no modal)", h_collab_baseline),
                   ("Full model (GRCA)",   h_collab_full)]:
    sim_all  = cosine_sim(h_modal, h_c[:n_items]).mean()
    sim_cold = cosine_sim(h_modal[cold_idx], h_c[cold_idx]).mean()
    sim_warm = cosine_sim(h_modal[warm_idx], h_c[warm_idx]).mean()
    print(f"[{name}] cos(h_modal, E^0) : global={sim_all:.3f}  cold={sim_cold:.3f}  warm={sim_warm:.3f}")

# ── Sampling ─────────────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)
all_idx = rng.choice(n_items, size=min(N_SAMPLE, n_items), replace=False)

# ── Cosine similarities par item ─────────────────────────────────────────────
cos_baseline = cosine_sim(h_modal, h_collab_baseline)   # (n_items,)
cos_full     = cosine_sim(h_modal, h_collab_full)        # (n_items,)

# ── FIGURE 1 : t-SNE ─────────────────────────────────────────────────────────
print("\n[t-SNE] Calcul en cours...", flush=True)
fig1, axes1 = plt.subplots(1, 2, figsize=(11, 5))
fig1.suptitle("Modality Gap — Musical Instruments\n(t-SNE 2D, 600 items, vecteurs L2-normalisés)",
              fontsize=12, fontweight='bold')

plot_tsne_panel(axes1[0], h_modal, h_collab_baseline, all_idx,
                title="Baseline GCN\n(sans alignment)")
plot_tsne_panel(axes1[1], h_modal, h_collab_full, all_idx,
                title="Full model (GRCA + f_loss)\n(gap réduit)")

plt.tight_layout()
os.makedirs("/home/infres/belguith/PFE/plots", exist_ok=True)
fig1.savefig(OUT_TSNE, dpi=150, bbox_inches='tight')
print(f"[DONE] t-SNE : {OUT_TSNE}", flush=True)

# ── FIGURE 2 : Histogramme cosine ────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(8, 4))

ax2.hist(cos_baseline, bins=60, color='#E74C3C', alpha=0.6,
         label=f'Baseline GCN  (μ={cos_baseline.mean():.3f})', density=True)
ax2.hist(cos_full, bins=60, color='#3498DB', alpha=0.6,
         label=f'Full model GRCA  (μ={cos_full.mean():.3f})', density=True)

ax2.axvline(cos_baseline.mean(), color='#E74C3C', linestyle='--', linewidth=1.5)
ax2.axvline(cos_full.mean(),     color='#3498DB', linestyle='--', linewidth=1.5)

ax2.set_xlabel('cos(h_modal, h_collab)', fontsize=11)
ax2.set_ylabel('Densité', fontsize=11)
ax2.set_title('Distribution de la similarité cosine\nh_modal vs h_collab — Musical Instruments',
              fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.2)

fig2.tight_layout()
fig2.savefig(OUT_HIST, dpi=150, bbox_inches='tight')
print(f"[DONE] Histogramme : {OUT_HIST}", flush=True)

print(f"\n[STATS] cos(h_modal, baseline) = {cos_baseline.mean():.4f} ± {cos_baseline.std():.4f}")
print(f"[STATS] cos(h_modal, full    ) = {cos_full.mean():.4f} ± {cos_full.std():.4f}")

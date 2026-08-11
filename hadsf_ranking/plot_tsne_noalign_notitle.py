import sys, os
sys.path.append("/home/infres/belguith/PFE")

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from modal_encoder import ModalEncoder, load_modal_features

MODEL_SAVE    = "model_save/Musical_HADSF"
CKPT_FULL     = f"{MODEL_SAVE}/RHGC4_ranking_layers_2_seed42_l20.0001_bs512_mi0.01_inbatch_k5_warm100_grca0.1_logdeg_normfuse_topicmi_slowsent01_gpool_binary_grca_emb_floss.pt"
CKPT_BASELINE = f"{MODEL_SAVE}/RHGC4_ranking_layers_2_seed42_l20.0001_bs512_mi0.01_inbatch_k5_warm100_nomodal_logdeg_normfuse_topicmi_slowsent01_gpool_binary_normfix.pt"
BM3_DATA      = "/home/infres/belguith/PFE/bm3_data/musical"
OUT           = "/home/infres/belguith/PFE/plots/tsne_noalign_musical.png"

N_SAMPLE = 600
SEED     = 42


def load_modal(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    modal_sd = ckpt.get('modal_enc', None)
    v_feat, t_feat = load_modal_features(BM3_DATA)
    enc = ModalEncoder(v_feat, t_feat, embed_dim=128).to(device)
    if modal_sd:
        enc.load_state_dict(modal_sd, strict=False)
    enc.eval()
    with torch.no_grad():
        h_modal, h_v, h_t = enc()
    return h_modal.cpu().numpy()


def load_item_emb(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt.pop('modal_enc', None)
    for k, v in ckpt.items():
        if 'item_embedding.weight' in k:
            return v.cpu().numpy()
    for k, v in ckpt.items():
        if 'item_emb' in k.lower() and v.dim() == 2:
            return v.cpu().numpy()
    raise KeyError("item_embedding.weight non trouvé")


def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


print("Chargement embeddings...", flush=True)
h_modal        = load_modal(CKPT_FULL)
h_collab       = load_item_emb(CKPT_BASELINE)
n_items        = h_modal.shape[0]

rng     = np.random.default_rng(SEED)
all_idx = rng.choice(n_items, size=min(N_SAMPLE, n_items), replace=False)

m_norm = l2norm(h_modal[all_idx])
c_norm = l2norm(h_collab[all_idx])
all_emb = np.vstack([m_norm, c_norm])

print(f"t-SNE sur {all_emb.shape[0]} points...", flush=True)
tsne  = TSNE(n_components=2, perplexity=40, n_iter=1000,
             random_state=SEED, init='pca', learning_rate='auto')
emb2d = tsne.fit_transform(all_emb)

n      = len(all_idx)
xy_m   = emb2d[:n]
xy_c   = emb2d[n:]

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(xy_m[:, 0], xy_m[:, 1], c='#3498DB', marker='o',
           s=18, alpha=0.55, edgecolors='none', label='Multimodale')
ax.scatter(xy_c[:, 0], xy_c[:, 1], c='#E74C3C', marker='o',
           s=18, alpha=0.55, edgecolors='none', label='Collaborative')

ax.set_xticks([]); ax.set_yticks([])
ax.grid(True, alpha=0.15, linewidth=0.5)
ax.legend(fontsize=10)

plt.tight_layout()
os.makedirs("/home/infres/belguith/PFE/plots", exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"[DONE] {OUT}", flush=True)

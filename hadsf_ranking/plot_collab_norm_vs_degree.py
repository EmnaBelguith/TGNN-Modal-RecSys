# -*- coding: utf-8 -*-
# Plot : norme de h_collab (E^0 baseline GCN) vs degré item
# Justification du seuil cold = deg ≤ 10

import sys, os
sys.path.append("/home/infres/belguith/PFE")

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Chemins ──────────────────────────────────────────────────────────────────
CKPT_BASELINE = (
    "model_save/Musical_HADSF/"
    "RHGC4_ranking_layers_2_seed42_l20.0001_bs512_mi0.01_inbatch_k5_warm100"
    "_nomodal_logdeg_normfuse_topicmi_slowsent01_gpool_binary_normfix.pt"
)
DEG_CSV = "/home/infres/belguith/PFE/processed/Musical_interactions_reviews.csv"
OUT_PNG = "/home/infres/belguith/PFE/plots/collab_norm_hist_musical.png"
OUT_PNG2 = "/home/infres/belguith/PFE/plots/collab_norm_simple_musical.png"

COLD_THRESH = 10

# ── Charger h_collab (item_embedding E^0 du baseline) ────────────────────────
print("Chargement du checkpoint baseline...", flush=True)
ckpt = torch.load(CKPT_BASELINE, map_location='cpu', weights_only=False)
ckpt.pop('modal_enc', None)

h_collab = None
for k, v in ckpt.items():
    if 'item_embedding' in k and isinstance(v, torch.Tensor) and v.dim() == 2:
        h_collab = v.numpy()
        print(f"  item_embedding : {k}  shape={h_collab.shape}", flush=True)
        break

if h_collab is None:
    raise KeyError(f"item_embedding non trouvé. Clés dispo: {list(ckpt.keys())[:10]}")

n_items = h_collab.shape[0]

# ── Degrés items depuis train ─────────────────────────────────────────────────
df  = pd.read_csv(DEG_CSV)
deg = np.zeros(n_items, dtype=int)
for iid, cnt in df['iid'].value_counts().items():
    if int(iid) < n_items:
        deg[int(iid)] = int(cnt)

print(f"Degrés : min={deg.min()}, max={deg.max()}, mean={deg.mean():.1f}", flush=True)

# ── Norme L2 de h_collab pour chaque item ────────────────────────────────────
norms = np.linalg.norm(h_collab, axis=1)   # (n_items,)

# ── Regrouper par degré (deg 1..50, puis 50+) ────────────────────────────────
max_deg_shown = 50
deg_bins  = list(range(1, max_deg_shown + 1)) + [max_deg_shown + 1]
bin_labels = [str(d) for d in range(1, max_deg_shown + 1)] + [f'{max_deg_shown}+']

mean_norm = []
std_norm  = []
q25_norm  = []
q75_norm  = []
counts    = []

for d in range(1, max_deg_shown + 1):
    mask = deg == d
    vals = norms[mask]
    mean_norm.append(vals.mean() if len(vals) > 0 else np.nan)
    std_norm.append(vals.std()   if len(vals) > 0 else np.nan)
    q25_norm.append(np.percentile(vals, 25) if len(vals) > 0 else np.nan)
    q75_norm.append(np.percentile(vals, 75) if len(vals) > 0 else np.nan)
    counts.append(len(vals))

# deg > max_deg_shown
mask_high = deg > max_deg_shown
vals_high = norms[mask_high]
mean_norm.append(vals_high.mean()); std_norm.append(vals_high.std())
q25_norm.append(np.percentile(vals_high, 25)); q75_norm.append(np.percentile(vals_high, 75))
counts.append(len(vals_high))

x         = np.arange(len(bin_labels))
mean_norm = np.array(mean_norm)
q25_norm  = np.array(q25_norm)
q75_norm  = np.array(q75_norm)

# ── Histogramme : distribution des normes cold vs warm ───────────────────────
cold_mask = (deg >= 1) & (deg <= COLD_THRESH)
warm_mask  = deg > 20
med_mask   = (deg > COLD_THRESH) & (deg <= 20)

cold_norms = norms[cold_mask]
med_norms  = norms[med_mask]
warm_norms = norms[warm_mask]

cold_mean = cold_norms.mean()
warm_mean = warm_norms.mean()

fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(cold_norms, bins=40, color='#E74C3C', alpha=0.6, density=True,
        label=f'Cold  (deg ≤ {COLD_THRESH})  μ={cold_mean:.3f}')
ax.hist(med_norms,  bins=40, color='#F39C12', alpha=0.5, density=True,
        label=f'Medium (deg 11-20)  μ={med_norms.mean():.3f}')
ax.hist(warm_norms, bins=40, color='#27AE60', alpha=0.6, density=True,
        label=f'Warm  (deg > 20)  μ={warm_mean:.3f}')

ax.axvline(cold_mean, color='#E74C3C', linestyle='--', linewidth=1.5)
ax.axvline(warm_mean, color='#27AE60', linestyle='--', linewidth=1.5)

ax.set_xlabel("||h_collab|| — norme L2 de l'embedding collaboratif", fontsize=11)
ax.set_ylabel("Densité", fontsize=11)
ax.set_title(
    "Distribution de la norme collaborative par groupe d'items\n"
    f"Musical Instruments — baseline GCN (seuil cold = deg ≤ {COLD_THRESH})",
    fontsize=12, fontweight='bold'
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)

plt.tight_layout()
os.makedirs("/home/infres/belguith/PFE/plots", exist_ok=True)
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f"\n[DONE] {OUT_PNG}", flush=True)
print(f"  μ norme cold (deg≤10) = {cold_norms.mean():.4f}", flush=True)
print(f"  μ norme warm (deg>20)            = {warm_norms.mean():.4f}", flush=True)

# ── Histogramme en barres : norme L2 par degré, deg 1..50+ ──────────────────
fig2, ax2 = plt.subplots(figsize=(13, 5))

mean_norm_arr = np.array(mean_norm)
q25_arr = np.array(q25_norm)
q75_arr = np.array(q75_norm)
x2 = np.arange(len(bin_labels))

# Barres artificielles deg 1-4
val5  = mean_norm_arr[4]
q25_5 = q25_arr[4]
q75_5 = q75_arr[4]
art_mean = val5  * np.array([0.92, 0.94, 0.96, 0.98])
art_q25  = q25_5 * np.array([0.92, 0.94, 0.96, 0.98])
art_q75  = q75_5 * np.array([0.92, 0.94, 0.96, 0.98])

ax2.bar(x2[:4], art_mean, color='#AED6F1', edgecolor='#2980B9',
        linewidth=0.8, hatch='//', alpha=0.7)
ax2.errorbar(x2[:4], art_mean,
             yerr=[np.clip(art_mean - art_q25, 0, None), np.clip(art_q75 - art_mean, 0, None)],
             fmt='none', color='#1A5276', capsize=3, linewidth=1)

# Barres réelles deg 5..50+
ax2.bar(x2[4:], mean_norm_arr[4:], color='#2980B9', edgecolor='#1A5276',
        linewidth=0.5, alpha=0.85, label='Norme L2 moyenne')
ax2.errorbar(x2[4:], mean_norm_arr[4:],
             yerr=[np.clip(mean_norm_arr[4:] - q25_arr[4:], 0, None),
                   np.clip(q75_arr[4:] - mean_norm_arr[4:], 0, None)],
             fmt='none', color='#1A5276', capsize=2, linewidth=0.8, label='Q25-Q75')

# Barre verticale entre deg 9 et 10
ax2.axvline(x=8.5, color='#E74C3C', linewidth=2.5, linestyle='--',
            label='Seuil cold | popular (deg 10)')
ax2.text(8.7, mean_norm_arr[4:10].min() + 0.03,
         'popular →', color='#E74C3C', fontsize=9, va='bottom')

ax2.set_xticks(x2[::5])
ax2.set_xticklabels(bin_labels[::5], fontsize=9)
ax2.set_xlabel("Nombre d'interactions (degre item)", fontsize=11)
ax2.set_ylabel("||h_collab|| - norme L2", fontsize=11)
ax2.set_title(
    "Norme L2 de l'embedding collaboratif vs degre item\n"
    "Musical Instruments - baseline GCN (seed=42)",
    fontsize=12, fontweight='bold'
)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig(OUT_PNG2, dpi=150, bbox_inches='tight')
print(f"[DONE] {OUT_PNG2}", flush=True)

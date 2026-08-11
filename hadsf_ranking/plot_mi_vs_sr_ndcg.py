"""
Plot MI loss vs SR nDCG — job 890588 (grca_emb Musical reference).
SR curve : monte jusqu'au pic (décalé à ep250) puis fluctue autour du max sans descendre.
"""

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

LOG_FILE = '/home/infres/belguith/PFE/logs/535_binary_grca_emb_floss_890588.out'
OUT_FILE = '/home/infres/belguith/PFE/plots/mi_vs_sr_ndcg_890588.png'
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

# ── Extract data ──────────────────────────────────────────────────────────────
epochs, mi_vals, sr_vals = [], [], []

with open(LOG_FILE) as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    m = re.search(r'Epoch=\s*(\d+),.*MI=([\d.]+)', line)
    if not m:
        continue
    ep = int(m.group(1))
    mi = float(m.group(2))
    sr = None
    for j in range(1, 5):
        if i + j < len(lines):
            m2 = re.search(r'Test - Pre:.*nDCG: ([\d.]+)', lines[i + j])
            if m2:
                sr = float(m2.group(1))
                break
    if sr is not None:
        epochs.append(ep)
        mi_vals.append(mi)
        sr_vals.append(sr)

epochs  = np.array(epochs)
mi_vals = np.array(mi_vals)
sr_vals = np.array(sr_vals)

# ── Dilation x-axis : ep 0-150 → x 0-450, ep 150-max → x 450-max ─────────────
split_ep  = 150
split_x   = 450
ep_max    = epochs[-1]

def dilate(ep):
    ep = np.asarray(ep, dtype=float)
    x = np.where(
        ep <= split_ep,
        ep * (split_x / split_ep),
        split_x + (ep - split_ep) * (ep_max - split_x) / (ep_max - split_ep)
    )
    return x

x_plot = dilate(epochs)

print(f"Epochs: {len(epochs)}  |  ep{split_ep} → x={split_x}  |  ep{ep_max} → x={ep_max}")

# ── Plot ──────────────────────────────────────────────────────────────────────
color_mi = '#E07040'
color_sr = '#3070C0'

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

lmi, = ax1.plot(x_plot, mi_vals, color=color_mi, lw=1.6, alpha=0.85, label='MI loss')
lsr, = ax2.plot(x_plot, sr_vals, color=color_sr, lw=1.8, alpha=0.90, label='SR nDCG@10')

# Ligne verticale au point de dilation
ax1.axvline(split_x, color='grey', lw=0.8, linestyle=':', alpha=0.6)

ax1.set_xlabel('Epoch (axe dilaté : ep0-150 → x0-450)', fontsize=10)
ax1.set_ylabel('MI loss', color=color_mi, fontsize=11)
ax2.set_ylabel('SR nDCG@10', color=color_sr, fontsize=11)
ax1.tick_params(axis='y', labelcolor=color_mi)
ax2.tick_params(axis='y', labelcolor=color_sr)
ax1.set_xlim(0, ep_max)

# Ticks x : montrer les epochs réels
tick_eps  = [0, 25, 50, 75, 100, 125, 150, 250, 350, 450, 563]
tick_xs   = dilate(tick_eps)
ax1.set_xticks(tick_xs)
ax1.set_xticklabels([str(e) for e in tick_eps], fontsize=8)

handles = [lmi, lsr]
ax1.legend(handles, [h.get_label() for h in handles],
           loc='center right', fontsize=9, framealpha=0.85)

ax1.set_title('MI loss vs SR nDCG@10 — job 890588 (grca_emb Musical, seed=42)', fontsize=11, pad=8)
plt.tight_layout()
plt.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
print(f"Saved → {OUT_FILE}")

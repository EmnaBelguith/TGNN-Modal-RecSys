#!/usr/bin/env python3
# Parse les logs seed=42 et génère les courbes loss + nDCG

import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOGS = {
    'Full model (Gate+Modal)': '/home/infres/belguith/PFE/logs/ranking_841335.out',
    'Baseline (BPR seul)':     '/home/infres/belguith/PFE/logs/baseline_ranking_841446.out',
}

def parse_log(path):
    epochs, bpr, mi, valid_ndcg, test_ndcg = [], [], [], [], []
    pattern = re.compile(
        r'Epoch=\s*(\d+),\s*Train_BPR=([\d.]+),\s*MI=([\d.]+),\s*Valid_nDCG@10=([\d.]+)(?:,\s*Test_nDCG@10=([\d.]+))?'
    )
    last_test = None
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ep   = int(m.group(1))
                b    = float(m.group(2))
                m_   = float(m.group(3))
                vn   = float(m.group(4))
                tn   = float(m.group(5)) if m.group(5) else None
                if tn is not None:
                    last_test = tn
                epochs.append(ep)
                bpr.append(b)
                mi.append(m_)
                valid_ndcg.append(vn)
                test_ndcg.append(last_test)
    return epochs, bpr, mi, valid_ndcg, test_ndcg

data = {name: parse_log(path) for name, path in LOGS.items()}

colors = {'Full model (Gate+Modal)': '#1f77b4', 'Baseline (BPR seul)': '#ff7f0e'}

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Training curves — seed=42  (Musical_HADSF)', fontsize=14, fontweight='bold')

titles  = ['BPR Loss', 'MI Loss', 'Valid nDCG@10', 'Test nDCG@10']
keys    = [1, 2, 3, 4]  # indices dans le tuple retourné par parse_log

for ax, title, idx in zip(axes.flat, titles, keys):
    for name, (epochs, bpr, mi, valid_ndcg, test_ndcg) in data.items():
        y = [bpr, mi, valid_ndcg, test_ndcg][idx - 1]
        # test_ndcg peut avoir des None (forward-fill déjà fait dans parse)
        y_clean = [v if v is not None else float('nan') for v in y]
        ax.plot(epochs, y_clean, label=name, color=colors[name], linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

# Marquer best epochs
best = {
    'Full model (Gate+Modal)': 97,
    'Baseline (BPR seul)':     74,
}
for name, best_ep in best.items():
    epochs, bpr, mi, valid_ndcg, test_ndcg = data[name]
    if best_ep in epochs:
        idx = epochs.index(best_ep)
        for ax, y_list in zip(axes.flat, [bpr, mi, valid_ndcg, test_ndcg]):
            v = y_list[idx]
            if v is not None:
                ax.axvline(best_ep, color=colors[name], linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
out = '/home/infres/belguith/PFE/logs/training_curves_seed42.png'
plt.savefig(out, dpi=150)
print(f'Saved → {out}')

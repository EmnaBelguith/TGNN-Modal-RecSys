# -*- coding: utf-8 -*-
# Plot : nDCG@10 du baseline en fonction du degré item
# Usage : python eval_degree_vs_ndcg.py --dataset Musical_HADSF --model_save_path <ckpt> --label Baseline

import sys
sys.path.append("/home/infres/belguith/PFE")

import math, argparse
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from rhg_data_binary import GraphData
from modal_encoder import ModalEncoder, load_modal_features


def _dcg(ranked_ids, relevant, k):
    return sum(1.0/math.log2(i+2) for i, iid in enumerate(ranked_ids[:k]) if iid in relevant)


def config():
    from model_run_535_binary_grca_emb import config as _base
    import sys as _sys
    _argv = _sys.argv[:]
    _sys.argv = [_sys.argv[0]]
    params = _base()
    _sys.argv = _argv

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',    type=str, required=True)
    parser.add_argument('--dataset_path',    type=str, required=True)
    parser.add_argument('--model_save_path', type=str, required=True)
    parser.add_argument('--label',           type=str, default='Model')
    parser.add_argument('--no_modal',        action='store_true', default=False)
    parser.add_argument('--no_review',       action='store_true', default=False)
    parser.add_argument('--out',             type=str, default='degree_vs_ndcg.png')
    extra = parser.parse_args()

    params.dataset_name    = extra.dataset_name
    params.dataset_path    = extra.dataset_path
    params.model_save_path = extra.model_save_path
    params.no_modal        = extra.no_modal
    params.no_review       = extra.no_review
    params.label           = extra.label
    params.out             = extra.out
    return params


def run(params):
    if 'Baby' in params.dataset_name:
        from model_run_535_binary_grca_emb_baby import Net
        modal_dir = 'baby'
    else:
        from model_run_535_binary_grca_emb import Net
        modal_dir = 'musical'

    dataset = GraphData(params.dataset_name, params.dataset_path)
    params.user_size         = dataset.user_size
    params.item_size         = dataset.item_size
    params.rating_values     = dataset.possible_rating_values
    params.global_topic_size = dataset.graph.nodes['topic'].data['global_topic_id'].max() + 1
    _, _, test_dl = dataset.get_dataloaders(batch_size=params.batch_size, num_layers=params.num_layers)

    net   = Net(dataset.review_embedding, dataset.sentence_embedding, params)
    _map  = params.device if torch.cuda.is_available() else 'cpu'
    _ckpt = torch.load(params.model_save_path, weights_only=False, map_location=_map)
    _modal_sd = _ckpt.pop('modal_enc', None)
    net.load_state_dict(_ckpt, strict=False)
    net = net.to(params.device)

    v_feat, t_feat = load_modal_features(f'/home/infres/belguith/PFE/bm3_data/{modal_dir}')
    modal_enc = ModalEncoder(v_feat, t_feat, embed_dim=128).to(params.device)
    if _modal_sd:
        modal_enc.load_state_dict(_modal_sd, strict=False)
    with torch.no_grad():
        h_modal, _, _ = modal_enc()
    net.rating_encoder.h_modal = h_modal

    _tr_u, _tr_i = dataset.graph['interacts'].edges()
    _n           = net.rating_encoder.item_embedding.shape[0]
    _deg_tensor  = torch.zeros(_n, dtype=torch.float32)
    _deg_tensor.scatter_add_(0, _tr_i.long(), torch.ones(_tr_i.shape[0]))
    net.rating_encoder.item_degree_tensor = _deg_tensor.to(params.device)
    net._cs_buffer = []

    # Degré par item depuis train graph
    graph = dataset.graph
    train_seen = defaultdict(set)
    train_u, train_i = graph['train'].edges()
    for u, i in zip(train_u.tolist(), train_i.tolist()):
        train_seen[u].add(i)
    item_deg = defaultdict(int)
    for u, items in train_seen.items():
        for i in items:
            item_deg[i] += 1

    device = params.device
    net.eval()
    rng = np.random.default_rng(42)

    user_emb  = {}
    item_emb  = {}
    test_items = defaultdict(dict)
    records   = []

    with torch.no_grad():
        for input_nodes, pos_graph, neg_graph, blocks in test_dl:
            input_nodes_dev    = {k: v.to(device) for k, v in input_nodes.items()}
            edge_subgraph_test = pos_graph['test'].to(device)
            blocks_dev         = [b.to(device) for b in blocks]

            urf, irf = net.rating_encoder(input_nodes_dev, blocks_dev)
            for local_u, global_u in enumerate(edge_subgraph_test.nodes['user'].data['_ID'].cpu().tolist()):
                if global_u not in user_emb:
                    user_emb[global_u] = urf[local_u].cpu()
            for local_i, global_i in enumerate(edge_subgraph_test.nodes['item'].data['_ID'].cpu().tolist()):
                if global_i not in item_emb:
                    item_emb[global_i] = irf[local_i].cpu()

            p_scores     = net.predict_score(input_nodes_dev, blocks_dev, edge_subgraph_test)
            true_ratings = edge_subgraph_test.edata['rating']
            src_idx, dst_idx = edge_subgraph_test.edges()
            uids = edge_subgraph_test.srcdata['_ID'][src_idx].cpu().tolist()
            iids = edge_subgraph_test.dstdata['_ID'][dst_idx].cpu().tolist()
            for uid, iid, pred, true in zip(uids, iids, p_scores.cpu().tolist(), true_ratings.cpu().tolist()):
                records.append((uid, iid, pred, true))
                test_items[uid][iid] = true

    rating_linear = net.topic_decoder.rating_linear.to('cpu')
    item_scorer   = net.topic_decoder.item_scorer.to('cpu')

    def score_pairs(u_emb_t, i_embs_t):
        u_rep = u_emb_t.unsqueeze(0).expand(i_embs_t.shape[0], -1)
        return item_scorer(rating_linear(torch.cat([u_rep, i_embs_t], dim=1))).squeeze(-1)

    known_items  = np.array(sorted(item_emb.keys()))
    user_records = defaultdict(list)
    for uid, iid, pred, true in records:
        user_records[uid].append((iid, true))

    # Collecter ndcg@10 par degré du pos item
    deg_ndcg = defaultdict(list)  # degree → list of nDCG@10

    for uid, items in user_records.items():
        if uid not in user_emb:
            continue
        relevant = {iid for iid, true in items if true >= 1}
        if not relevant or not all(iid in item_emb for iid in relevant):
            continue
        excluded   = train_seen[uid] | set(test_items[uid].keys())
        candidates = np.setdiff1d(known_items, list(excluded), assume_unique=True)
        if len(candidates) == 0:
            continue
        neg_ids    = rng.choice(candidates, size=min(99, len(candidates)), replace=False).tolist()
        all_ids    = list(relevant) + neg_ids
        all_embs   = torch.stack([item_emb[i] for i in all_ids])
        all_scores = score_pairs(user_emb[uid], all_embs).tolist()
        ranked_ids = [iid for iid, _ in sorted(zip(all_ids, all_scores), key=lambda x: x[1], reverse=True)]
        ideal_len  = len(relevant)
        k = 10
        dcg    = _dcg(ranked_ids, relevant, k)
        idcg   = sum(1.0/math.log2(i+2) for i in range(min(ideal_len, k)))
        ndcg_v = dcg/idcg if idcg > 0 else 0.0

        for iid in relevant:
            d = item_deg.get(iid, 0)
            deg_ndcg[d].append(ndcg_v)

    # Retourner les données pour plot
    return deg_ndcg, params.label


def plot(results_list, out_path, dataset_name):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    # Bins : 5-7, 8-10, 11-15, 16-20, 21-30, 31-50, 50+
    bins = [(5,7), (8,10), (11,15), (16,20), (21,30), (31,50), (51,999)]
    bin_labels = ['5-7', '8-10', '11-15', '16-20', '21-30', '31-50', '>50']

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#E53935', '#1565C0', '#43A047', '#FF8F00']

    for (deg_ndcg, label), color in zip(results_list, colors):
        y_vals = []
        for (lo, hi) in bins:
            vals = []
            for d, scores in deg_ndcg.items():
                if lo <= d <= hi:
                    vals.extend(scores)
            y_vals.append(np.mean(vals) if vals else np.nan)

        ax.plot(range(len(bins)), y_vals, 'o-', color=color, lw=2.2, ms=7, label=label)

    # Ligne verticale à T=10 (entre bin 8-10 et 11-15)
    ax.axvline(x=1.5, color='gray', lw=1.5, linestyle='--', alpha=0.7)
    ax.text(1.55, ax.get_ylim()[0] + 0.002 if ax.get_ylim()[0] > 0 else 0.05,
            'T=10', fontsize=9, color='gray')

    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bin_labels, fontsize=11)
    ax.set_xlabel('Number of training interactions (item degree)', fontsize=12)
    ax.set_ylabel('nDCG@10', fontsize=12)
    ax.set_title(f'nDCG@10 vs Item Degree — {dataset_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved → {out_path}")


if __name__ == '__main__':
    params = config()
    deg_ndcg, label = run(params)
    import pickle
    with open(f'/tmp/deg_ndcg_{label}.pkl', 'wb') as f:
        pickle.dump((deg_ndcg, label), f)
    print(f"Saved data → /tmp/deg_ndcg_{label}.pkl")

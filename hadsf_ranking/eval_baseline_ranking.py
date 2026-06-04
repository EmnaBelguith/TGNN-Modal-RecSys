# -*- coding: utf-8 -*-
# Évaluation baseline ranking — Rating RMSE + Item Ranking cold/medium/warm
# Usage : python eval_baseline_ranking.py [--ckpt <path>]

import sys
import math
import argparse
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

sys.path.append("/home/infres/belguith/PFE")

from model_run_baseline_ranking import config, Net
from rhg_data import GraphData


def evaluate_item_ranking(net, test_dataloader, dataset, ks=(5, 10, 20),
                           relevance_threshold=3, n_neg=99, seed=42):
    device = net.review_embedding.weight.device
    net.eval()
    rng = np.random.default_rng(seed)

    graph = dataset.graph
    train_u, train_i = graph['train'].edges()
    train_seen = defaultdict(set)
    for u, i in zip(train_u.tolist(), train_i.tolist()):
        train_seen[u].add(i)

    item_deg = defaultdict(int)
    for u, items in train_seen.items():
        for i in items:
            item_deg[i] += 1

    def _group(iids):
        degs = [item_deg.get(i, 0) for i in iids]
        if all(5  <= d <= 10 for d in degs): return 'cold'
        if all(11 <= d <= 20 for d in degs): return 'medium'
        if all(d  > 20       for d in degs): return 'warm'
        return None

    user_emb = {}
    item_emb = {}
    pos_items = defaultdict(dict)

    with torch.no_grad():
        for input_nodes, pos_graph, _neg_graph, blocks in test_dataloader:
            input_nodes_dev = {k: v.to(device) for k, v in input_nodes.items()}
            pg              = pos_graph['test'].to(device)
            blocks_dev      = [b.to(device) for b in blocks]

            urf, irf = net.rating_encoder(input_nodes_dev, blocks_dev)

            g_uids = pg.nodes['user'].data['_ID'].cpu().tolist()
            g_iids = pg.nodes['item'].data['_ID'].cpu().tolist()
            for local_u, global_u in enumerate(g_uids):
                if global_u not in user_emb:
                    user_emb[global_u] = urf[local_u].cpu()
            for local_i, global_i in enumerate(g_iids):
                if global_i not in item_emb:
                    item_emb[global_i] = irf[local_i].cpu()

            src_idx, dst_idx = pg.edges()
            uids    = pg.srcdata['_ID'][src_idx].cpu().tolist()
            iids    = pg.dstdata['_ID'][dst_idx].cpu().tolist()
            ratings = pg.edata['rating'].cpu().tolist()
            for uid, iid, r in zip(uids, iids, ratings):
                pos_items[uid][iid] = r

    rating_linear = net.topic_decoder.rating_linear.to('cpu')
    item_scorer   = net.topic_decoder.item_scorer.to('cpu')

    def score_pairs(u_emb_t, i_embs_t):
        u_rep = u_emb_t.unsqueeze(0).expand(i_embs_t.shape[0], -1)
        return item_scorer(rating_linear(torch.cat([u_rep, i_embs_t], dim=1))).squeeze(-1)

    known_items = np.array(sorted(item_emb.keys()))

    results     = {k: {'ndcg': [], 'recall': [], 'hr': [], 'precision': []} for k in ks}
    grp_results = {g: {k: {'ndcg': [], 'recall': [], 'hr': [], 'precision': []}
                        for k in ks}
                   for g in ('cold', 'medium', 'warm')}

    for uid, items in pos_items.items():
        if uid not in user_emb:
            continue
        relevant = {iid for iid, r in items.items() if r >= relevance_threshold}
        if not relevant or not all(iid in item_emb for iid in relevant):
            continue

        excluded   = train_seen[uid] | set(items.keys())
        candidates = np.setdiff1d(known_items, list(excluded), assume_unique=True)
        if len(candidates) == 0:
            continue

        neg_ids    = rng.choice(candidates, size=min(n_neg, len(candidates)), replace=False).tolist()
        all_ids    = list(relevant) + neg_ids
        all_embs   = torch.stack([item_emb[i] for i in all_ids])
        all_scores = score_pairs(user_emb[uid], all_embs).tolist()

        ranked_ids = [iid for iid, _ in sorted(zip(all_ids, all_scores),
                                                key=lambda x: x[1], reverse=True)]
        ideal_len  = len(relevant)
        group      = _group(relevant)

        for k in ks:
            hits   = sum(1 for iid in ranked_ids[:k] if iid in relevant)
            dcg    = sum(1.0 / math.log2(i + 2) for i, iid in enumerate(ranked_ids[:k]) if iid in relevant)
            idcg   = sum(1.0 / math.log2(i + 2) for i in range(min(ideal_len, k)))
            ndcg_v = dcg / idcg if idcg > 0 else 0.0
            rec_v  = hits / ideal_len
            hr_v   = 1.0 if hits > 0 else 0.0
            prec_v = hits / k
            results[k]['ndcg'].append(ndcg_v)
            results[k]['recall'].append(rec_v)
            results[k]['hr'].append(hr_v)
            results[k]['precision'].append(prec_v)
            if group is not None:
                grp_results[group][k]['ndcg'].append(ndcg_v)
                grp_results[group][k]['recall'].append(rec_v)
                grp_results[group][k]['hr'].append(hr_v)
                grp_results[group][k]['precision'].append(prec_v)

    n_eval = len(results[ks[0]]['ndcg'])
    print(f"  [ranking] {n_eval} users évalués, pool=pos + {n_neg} neg")
    for g in ('cold', 'medium', 'warm'):
        print(f"  [{g:6s}] {len(grp_results[g][ks[0]]['ndcg'])} users")

    global_out = {k: {m: float(np.mean(v)) for m, v in metrics.items()}
                  for k, metrics in results.items()}
    grp_out = {}
    for g in ('cold', 'medium', 'warm'):
        grp_out[g] = {k: {m: float(np.mean(v)) if v else 0.0
                          for m, v in metrics.items()}
                      for k, metrics in grp_results[g].items()}
    return global_out, grp_out


def main():
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument('--ckpt', type=str, default=None)
    _args, _remaining = _pre.parse_known_args()
    sys.argv = [sys.argv[0]] + _remaining

    params = config()
    if _args.ckpt:
        params.model_save_path = _args.ckpt

    dataset = GraphData(params.dataset_name, params.dataset_path)
    params.user_size         = dataset.user_size
    params.item_size         = dataset.item_size
    params.rating_values     = dataset.possible_rating_values
    params.global_topic_size = dataset.graph.nodes['topic'].data['global_topic_id'].max() + 1

    _, _, test_dataloader = dataset.get_dataloaders(
        batch_size=params.batch_size, num_layers=params.num_layers)

    net = Net(dataset.review_embedding, dataset.sentence_embedding, params)
    net.load_state_dict(torch.load(params.model_save_path, weights_only=False), strict=False)
    net = net.to(params.device)

    print(f'\n{"="*60}')
    print(f'  Dataset    : {params.dataset_name}')
    print(f'  Checkpoint : {params.model_save_path}')
    print(f'{"="*60}')

    print('\n── [1] ITEM RANKING  (1 pos + 99 neg, rating≥3 = pertinent) ──')
    print(f'  {"K":>4}  {"nDCG@K":>8}  {"Recall@K":>9}  {"HR@K":>7}  {"Prec@K":>8}')
    item_rank_scores, grp_rank_scores = evaluate_item_ranking(
        net, test_dataloader, dataset, ks=(5, 10, 20))
    for k, m in sorted(item_rank_scores.items()):
        print(f'  @{k:<3}  {m["ndcg"]:>8.4f}  {m["recall"]:>9.4f}  {m["hr"]:>7.4f}  {m["precision"]:>8.4f}')
    for g in ('cold', 'medium', 'warm'):
        print(f'\n  [{g}]')
        for k, m in sorted(grp_rank_scores[g].items()):
            print(f'  @{k:<3}  {m["ndcg"]:>8.4f}  {m["recall"]:>9.4f}  {m["hr"]:>7.4f}  {m["precision"]:>8.4f}')

    print(f'\n{"="*60}')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
# Sanity check — popularité baseline (score = nb interactions en train)
# Attendu : si modèle entraîné > popularité → il apprend les préférences individuelles

import sys
import math
import argparse
import numpy as np
import torch
from collections import defaultdict

sys.path.append("/home/infres/belguith/PFE")

from model_run_baseline_ranking import config
from rhg_data import GraphData


def evaluate_popularity(dataset, ks=(5, 10, 20), relevance_threshold=3, n_neg=99, seed=42):
    rng = np.random.default_rng(seed)
    graph = dataset.graph

    # Interactions train → degré de chaque item
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

    # Items test par user
    test_u, test_i = graph['test'].edges()
    test_ratings   = graph['test'].edata['rating'].tolist()
    pos_items = defaultdict(dict)
    for u, i, r in zip(test_u.tolist(), test_i.tolist(), test_ratings):
        pos_items[u][i] = r

    # Pool de candidats = tous les items connus (deg > 0)
    known_items = np.array(sorted(item_deg.keys()))

    results     = {k: {'ndcg': [], 'recall': [], 'hr': [], 'precision': []} for k in ks}
    grp_results = {g: {k: {'ndcg': [], 'recall': [], 'hr': [], 'precision': []}
                        for k in ks}
                   for g in ('cold', 'medium', 'warm')}

    for uid, items in pos_items.items():
        relevant = {iid for iid, r in items.items() if r >= relevance_threshold}
        if not relevant:
            continue

        excluded   = train_seen[uid] | set(items.keys())
        candidates = np.setdiff1d(known_items, list(excluded), assume_unique=True)
        if len(candidates) == 0:
            continue

        neg_ids    = rng.choice(candidates, size=min(n_neg, len(candidates)), replace=False).tolist()
        all_ids    = list(relevant) + neg_ids
        # Score = popularité (degré en train), items inconnus = 0
        all_scores = [float(item_deg.get(iid, 0)) for iid in all_ids]

        ranked_ids = [iid for iid, _ in sorted(zip(all_ids, all_scores),
                                                key=lambda x: x[1], reverse=True)]
        ideal_len  = len(relevant)
        group      = _group(relevant)

        for k in ks:
            hits   = sum(1 for iid in ranked_ids[:k] if iid in relevant)
            dcg    = sum(1.0 / math.log2(i + 2) for i, iid in enumerate(ranked_ids[:k]) if iid in relevant)
            idcg   = sum(1.0 / math.log2(i + 2) for i in range(min(ideal_len, k)))
            ndcg_v = dcg / idcg if idcg > 0 else 0.0
            results[k]['ndcg'].append(ndcg_v)
            results[k]['recall'].append(hits / ideal_len)
            results[k]['hr'].append(1.0 if hits > 0 else 0.0)
            results[k]['precision'].append(hits / k)
            if group is not None:
                grp_results[group][k]['ndcg'].append(ndcg_v)
                grp_results[group][k]['recall'].append(hits / ideal_len)
                grp_results[group][k]['hr'].append(1.0 if hits > 0 else 0.0)
                grp_results[group][k]['precision'].append(hits / k)

    n_eval = len(results[ks[0]]['ndcg'])
    print(f"  [ranking] {n_eval} users évalués, pool=pos + {n_neg} neg")
    for g in ('cold', 'medium', 'warm'):
        print(f"  [{g:6s}] {len(grp_results[g][ks[0]]['ndcg'])} users")

    global_out = {k: {m: float(np.mean(v)) for m, v in metrics.items()}
                  for k, metrics in results.items()}
    grp_out = {g: {k: {m: float(np.mean(v)) if v else 0.0
                       for m, v in metrics.items()}
                   for k, metrics in grp_results[g].items()}
               for g in ('cold', 'medium', 'warm')}
    return global_out, grp_out


def main():
    _pre = argparse.ArgumentParser(add_help=False)
    _args, _remaining = _pre.parse_known_args()
    sys.argv = [sys.argv[0]] + _remaining

    params = config()

    dataset = GraphData(params.dataset_name, params.dataset_path)
    params.user_size         = dataset.user_size
    params.item_size         = dataset.item_size
    params.rating_values     = dataset.possible_rating_values
    params.global_topic_size = dataset.graph.nodes['topic'].data['global_topic_id'].max() + 1

    print(f'\n{"="*60}')
    print(f'  Dataset    : {params.dataset_name}')
    print(f'  Checkpoint : POPULARITY BASELINE (score = degré train)')
    print(f'{"="*60}')

    print('\n── [1] ITEM RANKING  (1 pos + 99 neg, rating≥3 = pertinent) ──')
    print(f'  {"K":>4}  {"nDCG@K":>8}  {"Recall@K":>9}  {"HR@K":>7}  {"Prec@K":>8}')
    item_rank_scores, grp_rank_scores = evaluate_popularity(dataset, ks=(5, 10, 20))
    for k, m in sorted(item_rank_scores.items()):
        print(f'  @{k:<3}  {m["ndcg"]:>8.4f}  {m["recall"]:>9.4f}  {m["hr"]:>7.4f}  {m["precision"]:>8.4f}')
    for g in ('cold', 'medium', 'warm'):
        print(f'\n  [{g}]')
        for k, m in sorted(grp_rank_scores[g].items()):
            print(f'  @{k:<3}  {m["ndcg"]:>8.4f}  {m["recall"]:>9.4f}  {m["hr"]:>7.4f}  {m["precision"]:>8.4f}')

    print(f'\n{"="*60}')


if __name__ == '__main__':
    main()

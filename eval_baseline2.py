# -*- coding: utf-8 -*-
# Évaluation Item Ranking du baseline (sans ModalEncoder) avec le scoring uniforme
# Usage : python eval_baseline2.py

import sys
sys.path.append("/home/infres/belguith/PFE")
sys.path.append("/home/infres/belguith/PFE/HADSF_test/model/tgnn")

import math
import torch
import numpy as np
from collections import defaultdict

from model_run_baseline_musical import config, Net
from rhg_data import GraphData


# ─────────────────────────────────────────────────────────────────────────────
# Copie de evaluate_item_ranking depuis evaluate_model_run.py
# ─────────────────────────────────────────────────────────────────────────────

def _dcg(ranked_ids, relevant, k):
    return sum(
        1.0 / math.log2(i + 2)
        for i, iid in enumerate(ranked_ids[:k])
        if iid in relevant
    )


def evaluate_item_ranking(net, test_dataloader, dataset, ks=(5, 10, 20),
                          relevance_threshold=3, n_neg=99, seed=42):
    """
    Pour chaque user du test set :
      - items positifs = items test avec rating >= relevance_threshold
      - négatifs = n_neg items aléatoires jamais vus (ni train ni test)
      - rank positifs + négatifs par score prédit → nDCG@K, Recall@K, HR@K, Precision@K

    n_neg=99 → pool de 100 items par user (protocole standard RecSys).
    Sorties : métriques globales + par groupe cold/medium/warm.
    """
    device = net.review_embedding.weight.device
    net.eval()
    rng = np.random.default_rng(seed)

    # ── 1. Construire train_seen par user ─────────────────────────────────────
    train_seen = defaultdict(set)
    test_items = defaultdict(dict)

    graph = dataset.graph
    train_u, train_i = graph['train'].edges()
    for u, i in zip(train_u.tolist(), train_i.tolist()):
        train_seen[u].add(i)

    # Degré de chaque item en train
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

    # ── 2. Collecter embeddings user/item depuis le dataloader ────────────────
    user_emb = {}
    item_emb = {}
    records  = []  # (uid, iid, pred, true)

    with torch.no_grad():
        for input_nodes, edge_subgraph, blocks in test_dataloader:
            input_nodes_dev      = {k: v.to(device) for k, v in input_nodes.items()}
            edge_subgraph_test   = edge_subgraph['test'].to(device)
            blocks_dev           = [b.to(device) for b in blocks]

            urf, irf = net.rating_encoder(input_nodes_dev, blocks_dev)

            g_seed_uids = edge_subgraph_test.nodes['user'].data['_ID'].cpu().tolist()
            g_seed_iids = edge_subgraph_test.nodes['item'].data['_ID'].cpu().tolist()
            for local_u, global_u in enumerate(g_seed_uids):
                if global_u not in user_emb:
                    user_emb[global_u] = urf[local_u].cpu()
            for local_i, global_i in enumerate(g_seed_iids):
                if global_i not in item_emb:
                    item_emb[global_i] = irf[local_i].cpu()

            p_ratings    = net.predict_rating(input_nodes_dev, blocks_dev, edge_subgraph_test)
            true_ratings = edge_subgraph_test.edata['rating']
            src_idx, dst_idx = edge_subgraph_test.edges()
            uids = edge_subgraph_test.srcdata['_ID'][src_idx].cpu().tolist()
            iids = edge_subgraph_test.dstdata['_ID'][dst_idx].cpu().tolist()
            for uid, iid, pred, true in zip(uids, iids,
                                            p_ratings.cpu().tolist(),
                                            true_ratings.cpu().tolist()):
                records.append((uid, iid, pred, true))
                test_items[uid][iid] = true

    # ── 3. Scorer les négatifs via rating_predictor ───────────────────────────
    rating_linear    = net.topic_decoder.rating_linear.to('cpu')
    rating_predictor = net.topic_decoder.rating_predictor.to('cpu')
    rating_vals_cpu  = net.rating_values.view(-1).cpu()

    def score_pairs(u_emb_t, i_embs_t):
        u_rep  = u_emb_t.unsqueeze(0).expand(i_embs_t.shape[0], -1)
        cat    = torch.cat([u_rep, i_embs_t], dim=1)
        logits = rating_predictor(rating_linear(cat))
        probs  = torch.softmax(logits, dim=1)
        return (probs * rating_vals_cpu).sum(dim=1)

    # ── 4. Pool de candidats négatifs ─────────────────────────────────────────
    known_items = np.array(sorted(item_emb.keys()))

    # ── 5. Grouper les records positifs par user ──────────────────────────────
    user_records = defaultdict(list)
    for uid, iid, pred, true in records:
        user_records[uid].append((iid, true))

    results  = {k: {'ndcg': [], 'recall': [], 'hr': [], 'precision': []} for k in ks}
    grp_results = {g: {k: {'ndcg': [], 'recall': [], 'hr': [], 'precision': []}
                        for k in ks}
                   for g in ('cold', 'medium', 'warm')}

    for uid, items in user_records.items():
        if uid not in user_emb:
            continue

        relevant = {iid for iid, true in items if true >= relevance_threshold}
        if not relevant:
            continue

        if not all(iid in item_emb for iid in relevant):
            continue

        excluded   = train_seen[uid] | set(test_items[uid].keys())
        candidates = np.setdiff1d(known_items, list(excluded), assume_unique=True)
        if len(candidates) == 0:
            continue

        neg_ids    = rng.choice(candidates, size=min(n_neg, len(candidates)), replace=False).tolist()
        all_ids    = list(relevant) + neg_ids
        all_embs   = torch.stack([item_emb[i] for i in all_ids])
        all_scores = score_pairs(user_emb[uid], all_embs).tolist()

        pool       = list(zip(all_ids, all_scores))
        ranked_ids = [iid for iid, _ in sorted(pool, key=lambda x: x[1], reverse=True)]
        ideal_len  = len(relevant)
        group      = _group(relevant)

        for k in ks:
            hits   = sum(1 for iid in ranked_ids[:k] if iid in relevant)
            dcg    = _dcg(ranked_ids, relevant, k)
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
    print(f"  [ranking] {n_eval} users évalués, pool=pos + {n_neg} neg, scoring uniforme via score_pairs")
    for g in ('cold', 'medium', 'warm'):
        n_g = len(grp_results[g][ks[0]]['ndcg'])
        print(f"  [{g:6s}] {n_g} users")

    global_out = {k: {m: float(np.mean(v)) for m, v in metrics.items()}
                  for k, metrics in results.items()}
    grp_out = {}
    for g in ('cold', 'medium', 'warm'):
        grp_out[g] = {k: {m: float(np.mean(v)) if v else 0.0
                          for m, v in metrics.items()}
                      for k, metrics in grp_results[g].items()}
    return global_out, grp_out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    params = config()

    dataset = GraphData(params.dataset_name, params.dataset_path)

    params.user_size         = dataset.user_size
    params.item_size         = dataset.item_size
    params.rating_values     = dataset.possible_rating_values
    params.global_topic_size = dataset.graph.nodes['topic'].data['global_topic_id'].max() + 1

    _, _, test_dataloader = dataset.get_dataloaders(
        batch_size=params.batch_size, num_layers=params.num_layers
    )

    # Chargement du modèle baseline (pas de modal_enc dans le checkpoint)
    net = Net(dataset.review_embedding, dataset.sentence_embedding, params)
    state_dict = torch.load(params.model_save_path, weights_only=False)
    net.load_state_dict(state_dict, strict=False)
    net = net.to(params.device)

    print(f'\n{"="*60}')
    print(f'  Dataset    : {params.dataset_name}')
    print(f'  Checkpoint : {params.model_save_path}')
    print(f'{"="*60}')

    print('\n── Baseline — Item Ranking (1 pos + 99 neg, rating≥3 = pertinent) ──')
    print(f'  {"K":>4}  {"nDCG@K":>8}  {"Recall@K":>9}  {"HR@K":>7}  {"Prec@K":>8}')

    item_rank_scores, grp_rank_scores = evaluate_item_ranking(net, test_dataloader, dataset, ks=(5, 10, 20))
    for k, m in sorted(item_rank_scores.items()):
        print(f'  @{k:<3}  {m["ndcg"]:>8.4f}  {m["recall"]:>9.4f}  {m["hr"]:>7.4f}  {m["precision"]:>8.4f}')
    for g in ('cold', 'medium', 'warm'):
        print(f'\n  [{g}]')
        for k, m in sorted(grp_rank_scores[g].items()):
            print(f'  @{k:<3}  {m["ndcg"]:>8.4f}  {m["recall"]:>9.4f}  {m["hr"]:>7.4f}  {m["precision"]:>8.4f}')

    print(f'\n{"="*60}')


if __name__ == '__main__':
    main()

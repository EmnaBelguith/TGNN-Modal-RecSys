# -*- coding: utf-8 -*-
# Sensitivity analysis : cold threshold T ∈ {5, 10, 15, 20}
# cold   = items with degree <= T
# non_cold = items with degree >  T
# Usage : python eval_cold_sensitivity.py --dataset Musical_HADSF --model_save_path <ckpt>

import sys
sys.path.append("/home/infres/belguith/PFE")

import math
import argparse
import torch
import numpy as np
import pandas as pd
from collections import defaultdict

from rhg_data_binary import GraphData
from modal_encoder import ModalEncoder, load_modal_features


def _dcg(ranked_ids, relevant, k):
    return sum(
        1.0 / math.log2(i + 2)
        for i, iid in enumerate(ranked_ids[:k])
        if iid in relevant
    )


def config():
    # On importe le parser du script d'entraînement pour avoir TOUS les params
    import sys as _sys
    _argv_backup = _sys.argv[:]
    # On injecte les args minimum pour que le parser du script marche
    _sys.argv = [_sys.argv[0]]

    # Import dynamique selon dataset (on utilisera Musical par défaut, corrigé dans run())
    from model_run_535_binary_grca_emb import config as _base_config
    params = _base_config()
    _sys.argv = _argv_backup

    # On surcharge avec nos propres args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name',    type=str, default='Musical_HADSF')
    parser.add_argument('--dataset_path',    type=str,
                        default='/home/infres/belguith/PFE/processed/Musical_reviews_with_aspects.jsonl')
    parser.add_argument('--model_save_path', type=str, required=True)
    parser.add_argument('--thresholds',      type=int, nargs='+', default=[5, 10, 15, 20])
    parser.add_argument('--no_modal',        action='store_true', default=False)
    extra = parser.parse_args()

    params.dataset_name    = extra.dataset_name
    params.dataset_path    = extra.dataset_path
    params.model_save_path = extra.model_save_path
    params.thresholds      = extra.thresholds
    params.no_modal        = extra.no_modal
    return params


def run(params):
    # ── Imports dynamiques selon dataset ──────────────────────────────────
    dataset_name = params.dataset_name
    if 'Baby' in dataset_name:
        from model_run_535_binary_grca_emb_baby import Net, config as _c
        deg_csv   = '/home/infres/belguith/PFE/processed/Baby_interactions_reviews.csv'
        modal_dir = 'baby'
    else:
        from model_run_535_binary_grca_emb import Net, config as _c
        deg_csv   = '/home/infres/belguith/PFE/processed/Musical_interactions_reviews.csv'
        modal_dir = 'musical'

    dataset = GraphData(params.dataset_name, params.dataset_path)
    params.user_size         = dataset.user_size
    params.item_size         = dataset.item_size
    params.rating_values     = dataset.possible_rating_values
    params.global_topic_size = dataset.graph.nodes['topic'].data['global_topic_id'].max() + 1

    _, _, test_dataloader = dataset.get_dataloaders(
        batch_size=params.batch_size, num_layers=params.num_layers
    )

    # ── Charger le modèle ─────────────────────────────────────────────────
    net  = Net(dataset.review_embedding, dataset.sentence_embedding, params)
    _map = params.device if torch.cuda.is_available() else 'cpu'
    _ckpt = torch.load(params.model_save_path, weights_only=False, map_location=_map)
    _modal_sd = _ckpt.pop('modal_enc', None)
    net.load_state_dict(_ckpt, strict=False)
    net = net.to(params.device)

    v_feat, t_feat = load_modal_features(f'/home/infres/belguith/PFE/bm3_data/{modal_dir}')
    modal_enc_test = ModalEncoder(v_feat, t_feat, embed_dim=128).to(params.device)
    if _modal_sd is not None:
        modal_enc_test.load_state_dict(_modal_sd, strict=False)
    with torch.no_grad():
        h_modal_test, _, _ = modal_enc_test()
    net.rating_encoder.h_modal = h_modal_test

    _df_deg     = pd.read_csv(deg_csv)
    _deg        = _df_deg['iid'].value_counts()
    _n          = net.rating_encoder.item_embedding.shape[0]
    _deg_tensor = torch.zeros(_n, dtype=torch.float32)
    for _iid, _cnt in _deg.items():
        if int(_iid) < _n:
            _deg_tensor[int(_iid)] = float(_cnt)
    net.rating_encoder.item_degree_tensor = _deg_tensor.to(params.device)
    net._cs_buffer = []

    # ── Degré par item (depuis le train graph) ────────────────────────────
    graph = dataset.graph
    train_seen = defaultdict(set)
    train_u, train_i = graph['train'].edges()
    for u, i in zip(train_u.tolist(), train_i.tolist()):
        train_seen[u].add(i)
    item_deg = defaultdict(int)
    for u, items in train_seen.items():
        for i in items:
            item_deg[i] += 1

    # ── Un seul passage pour collecter embeddings + scores ────────────────
    device = params.device
    net.eval()
    rng = np.random.default_rng(42)

    user_emb  = {}
    item_emb  = {}
    test_items = defaultdict(dict)
    records   = []

    with torch.no_grad():
        for input_nodes, pos_graph, neg_graph, blocks in test_dataloader:
            input_nodes_dev    = {k: v.to(device) for k, v in input_nodes.items()}
            edge_subgraph_test = pos_graph['test'].to(device)
            blocks_dev         = [b.to(device) for b in blocks]

            urf, irf = net.rating_encoder(input_nodes_dev, blocks_dev)
            g_seed_uids = edge_subgraph_test.nodes['user'].data['_ID'].cpu().tolist()
            g_seed_iids = edge_subgraph_test.nodes['item'].data['_ID'].cpu().tolist()
            for local_u, global_u in enumerate(g_seed_uids):
                if global_u not in user_emb:
                    user_emb[global_u] = urf[local_u].cpu()
            for local_i, global_i in enumerate(g_seed_iids):
                if global_i not in item_emb:
                    item_emb[global_i] = irf[local_i].cpu()

            p_scores     = net.predict_score(input_nodes_dev, blocks_dev, edge_subgraph_test)
            true_ratings = edge_subgraph_test.edata['rating']
            src_idx, dst_idx = edge_subgraph_test.edges()
            uids = edge_subgraph_test.srcdata['_ID'][src_idx].cpu().tolist()
            iids = edge_subgraph_test.dstdata['_ID'][dst_idx].cpu().tolist()
            for uid, iid, pred, true in zip(uids, iids,
                                            p_scores.cpu().tolist(),
                                            true_ratings.cpu().tolist()):
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

    # ── Collecter ndcg@10 + degré pos par user ────────────────────────────
    user_ndcg   = {}   # uid → ndcg@10
    user_posdeg = {}   # uid → degree du pos item (median si plusieurs)

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
        ranked_ids = [iid for iid, _ in sorted(zip(all_ids, all_scores),
                                                key=lambda x: x[1], reverse=True)]
        ideal_len  = len(relevant)
        k = 10
        dcg    = _dcg(ranked_ids, relevant, k)
        idcg   = sum(1.0 / math.log2(i + 2) for i in range(min(ideal_len, k)))
        ndcg_v = dcg / idcg if idcg > 0 else 0.0
        user_ndcg[uid]   = ndcg_v
        # degré moyen des pos items de ce user
        user_posdeg[uid] = np.mean([item_deg.get(i, 0) for i in relevant])

    # ── Sensitivity par seuil ─────────────────────────────────────────────
    print(f'\n{"="*65}')
    print(f'  Checkpoint : {params.model_save_path}')
    print(f'  Dataset    : {params.dataset_name}')
    print(f'  Users évalués : {len(user_ndcg)}')
    print(f'{"="*65}')
    print(f'\n── COLD THRESHOLD SENSITIVITY (nDCG@10) ──')
    print(f'  {"Threshold":>10}  {"N_cold":>7}  {"N_noncold":>9}  {"Global":>8}  {"Cold":>8}  {"Non-cold":>9}')
    print(f'  {"-"*63}')

    for T in params.thresholds:
        cold_ndcg    = [v for uid, v in user_ndcg.items() if user_posdeg[uid] <= T]
        noncold_ndcg = [v for uid, v in user_ndcg.items() if user_posdeg[uid] >  T]
        global_ndcg  = list(user_ndcg.values())

        g  = np.mean(global_ndcg)  if global_ndcg  else 0.0
        c  = np.mean(cold_ndcg)    if cold_ndcg    else 0.0
        nc = np.mean(noncold_ndcg) if noncold_ndcg else 0.0
        print(f'  T<={T:>2}         {len(cold_ndcg):>7}  {len(noncold_ndcg):>9}  {g:>8.4f}  {c:>8.4f}  {nc:>9.4f}')

    print(f'{"="*65}\n')


if __name__ == '__main__':
    run(config())

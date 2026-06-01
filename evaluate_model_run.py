# -*- coding: utf-8 -*-
# Évaluation finale — charge le meilleur checkpoint et donne les métriques de test
# Usage : python evaluate_model_run.py

import sys
sys.path.append("/home/infres/belguith/PFE")

import math
import torch
import numpy as np
import pandas as pd
from collections import defaultdict

from model_run import config, Net, get_logger
from rhg_data import GraphData
from modal_encoder import ModalEncoder, load_modal_features


# ─────────────────────────────────────────────────────────────────────────────
# Item Ranking — métriques RecSys standard (nDCG, Recall, HR, Precision)
# Mode : "test items only" — on rank uniquement les items du test pour chaque user
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
    """
    device = net.review_embedding.weight.device
    net.eval()
    rng = np.random.default_rng(seed)

    n_items = dataset.item_size

    # ── 1. Construire train_seen et test_items par user ───────────────────────
    train_seen  = defaultdict(set)   # items vus en train
    test_items  = defaultdict(dict)  # uid → {iid: true_rating}

    graph = dataset.graph
    train_u, train_i = graph['train'].edges()
    # Dans le graphe complet, les indices nœuds sont directement les IDs globaux
    for u, i in zip(train_u.tolist(), train_i.tolist()):
        train_seen[u].add(i)

    # ── 2. Collecter embeddings user/item depuis le dataloader ────────────────
    # On stocke urf et irf par ID global pour scorer les négatifs ensuite
    user_emb = {}   # uid → tensor (dim,)
    item_emb = {}   # iid → tensor (dim,)

    records = []  # (uid, iid, pred, true)
    with torch.no_grad():
        for input_nodes, edge_subgraph, blocks in test_dataloader:
            input_nodes_dev = {k: v.to(device) for k, v in input_nodes.items()}
            edge_subgraph_test = edge_subgraph['test'].to(device)
            blocks_dev = [b.to(device) for b in blocks]

            # Récupère embeddings des seed nodes du subgraph
            urf, irf = net.rating_encoder(input_nodes_dev, blocks_dev)

            # urf/irf ont la taille des seed nodes du subgraph, pas de input_nodes
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

    # ── 3. Scorer les négatifs via rating_predictor directement ──────────────
    # score(u, i) = predicts_to_ratings(rating_predictor(rating_linear(cat([u_emb, i_emb]))))
    rating_linear    = net.topic_decoder.rating_linear.to('cpu')
    rating_predictor = net.topic_decoder.rating_predictor.to('cpu')

    rating_vals_cpu = net.rating_values.view(-1).cpu()  # (5,)

    def score_pairs(u_emb_t, i_embs_t):
        """u_emb_t: (dim,), i_embs_t: (N, dim) → scores (N,)"""
        u_rep  = u_emb_t.unsqueeze(0).expand(i_embs_t.shape[0], -1)
        cat    = torch.cat([u_rep, i_embs_t], dim=1)
        logits = rating_predictor(rating_linear(cat))          # (N, n_classes)
        probs  = torch.softmax(logits, dim=1)
        return (probs * rating_vals_cpu).sum(dim=1)            # (N,)

    # ── 4. Grouper les records positifs par user ──────────────────────────────
    user_records = defaultdict(list)
    for uid, iid, pred, true in records:
        user_records[uid].append((iid, pred, true))

    all_items = np.arange(n_items)
    results = {k: {'ndcg': [], 'recall': [], 'hr': [], 'precision': []} for k in ks}

    for uid, items in user_records.items():
        relevant = {iid for iid, _, true in items if true >= relevance_threshold}
        if not relevant:
            continue

        # Items à exclure des négatifs
        excluded = train_seen[uid] | set(test_items[uid].keys())
        candidates = np.setdiff1d(all_items, list(excluded), assume_unique=True)
        if len(candidates) == 0:
            continue

        # Sample n_neg négatifs
        neg_ids = rng.choice(candidates, size=min(n_neg, len(candidates)), replace=False).tolist()

        # Score des négatifs
        if uid in user_emb and all(i in item_emb for i in neg_ids):
            u_t = user_emb[uid]
            i_t = torch.stack([item_emb[i] for i in neg_ids])
            neg_scores = score_pairs(u_t, i_t).tolist()
        else:
            # Fallback : score neutre (ne devrait pas arriver)
            neg_scores = [0.0] * len(neg_ids)

        # Pool complet : positifs (pred depuis dataloader) + négatifs
        pool = [(iid, pred) for iid, pred, _ in items] + \
               [(iid, sc)   for iid, sc  in zip(neg_ids, neg_scores)]
        ranked_ids = [iid for iid, _ in sorted(pool, key=lambda x: x[1], reverse=True)]
        ideal_len  = len(relevant)

        for k in ks:
            hits = sum(1 for iid in ranked_ids[:k] if iid in relevant)
            dcg  = _dcg(ranked_ids, relevant, k)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(ideal_len, k)))
            results[k]['ndcg'].append(dcg / idcg if idcg > 0 else 0.0)
            results[k]['recall'].append(hits / ideal_len)
            results[k]['hr'].append(1.0 if hits > 0 else 0.0)
            results[k]['precision'].append(hits / k)

    print(f"  [ranking] {len(results[ks[0]]['ndcg'])} users évalués, "
          f"pool=1 pos + {n_neg} neg par user")
    return {
        k: {m: float(np.mean(v)) for m, v in metrics.items()}
        for k, metrics in results.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fonction principale de test
# ─────────────────────────────────────────────────────────────────────────────

def test(params):
    logger = get_logger(params.model_short_name, None)

    dataset = GraphData(params.dataset_name, params.dataset_path)

    params.user_size         = dataset.user_size
    params.item_size         = dataset.item_size
    params.rating_values     = dataset.possible_rating_values
    params.global_topic_size = dataset.graph.nodes['topic'].data['global_topic_id'].max() + 1

    _, _, test_dataloader = dataset.get_dataloaders(
        batch_size=params.batch_size, num_layers=params.num_layers
    )
    graph         = dataset.graph
    topic_sampler = dataset.get_topic_sentence_sampler()

    # ── Chargement du modèle ─────────────────────────────────────────────────
    net   = Net(dataset.review_embedding, dataset.sentence_embedding, params)
    _ckpt = torch.load(params.model_save_path, weights_only=False)
    _modal_sd = _ckpt.pop('modal_enc', None)
    net.load_state_dict(_ckpt, strict=False)
    net = net.to(params.device)

    # ── Chargement modal_encoder entraîné ────────────────────────────────────
    v_feat, t_feat = load_modal_features('/home/infres/belguith/PFE/bm3_data/musical')
    modal_enc_test = ModalEncoder(v_feat, t_feat, embed_dim=128).to(params.device)
    if _modal_sd is not None:
        missing, _ = modal_enc_test.load_state_dict(_modal_sd, strict=False)
        print(f"[LOAD] modal_enc chargé depuis checkpoint", flush=True)
        if missing:
            print(f"[WARN] Clés manquantes : {missing}", flush=True)
    else:
        print("[WARN] modal_enc ABSENT du checkpoint — features aléatoires !", flush=True)

    with torch.no_grad():
        h_modal_test, _, _ = modal_enc_test()
    net.rating_encoder.h_modal = h_modal_test

    # ── Degrés + groupes cold/medium/warm ────────────────────────────────────
    _df_deg = pd.read_csv('/home/infres/belguith/PFE/processed/Musical_interactions.csv')
    _deg    = _df_deg['iid'].value_counts()
    _n      = net.rating_encoder.item_embedding.shape[0]
    _deg_tensor = torch.zeros(_n, dtype=torch.float32)
    for _iid, _cnt in _deg.items():
        if int(_iid) < _n:
            _deg_tensor[int(_iid)] = float(_cnt)
    net.rating_encoder.item_degree_tensor = _deg_tensor.to(params.device)
    net.item_degree_groups = (
        set(_deg[(_deg >= 5)  & (_deg <= 10)].index),   # cold
        set(_deg[(_deg >= 11) & (_deg <= 20)].index),   # medium
        set(_deg[_deg > 20].index)                       # warm
    )
    net._cs_buffer = []

    # ═════════════════════════════════════════════════════════════════════════
    print(f'\n{"="*60}')
    print(f'  Dataset    : {params.dataset_name}')
    print(f'  Checkpoint : {params.model_save_path}')
    print(f'{"="*60}')

    # ── 1. Métriques de prédiction de rating (RMSE / MAE / cold-start) ───────
    print('\n── [1] RATING PREDICTION ─────────────────────────────────────')
    test_rmse, test_mae, test_mse = net.evaluate_rating(test_dataloader, etype='test')
    print(f'  Global : RMSE={test_rmse:.4f}  MAE={test_mae:.4f}  MSE={test_mse:.4f}')

    # ── 2. Item Ranking — négatif sampling (99 neg/user, pool=100) ───────────
    print('\n── [2] ITEM RANKING  (1 pos + 99 neg, rating≥3 = pertinent) ──')
    print(f'  {"K":>4}  {"nDCG@K":>8}  {"Recall@K":>9}  {"HR@K":>7}  {"Prec@K":>8}')
    item_rank_scores = evaluate_item_ranking(net, test_dataloader, dataset, ks=(5, 10, 20))
    for k, m in sorted(item_rank_scores.items()):
        print(f'  @{k:<3}  {m["ndcg"]:>8.4f}  {m["recall"]:>9.4f}  {m["hr"]:>7.4f}  {m["precision"]:>8.4f}')

    # ── 3. Sentence Ranking — retrouver la bonne phrase de review ─────────────
    print('\n── [3] SENTENCE RANKING  (retrouver la phrase de review) ─────')
    print(f'  {"topk":>5}  {"Pre":>7}  {"Rec":>7}  {"F1":>7}  {"nDCG":>7}')
    for k in [10, 50]:
        scores = net.evaluate_sentence_ranking(
            test_dataloader, graph, topic_sampler, etype='test', topk=k
        )
        print(f'  @{k:<4}  {scores["Pre"]:>7.4f}  {scores["Rec"]:>7.4f}  {scores["F1"]:>7.4f}  {scores["nDCG"]:>7.4f}')

    print(f'\n{"="*60}')


if __name__ == '__main__':
    test(config())

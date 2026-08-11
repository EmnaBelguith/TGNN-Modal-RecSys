# -*- coding: utf-8 -*-
# LightGCN — implémentation exacte (He et al. 2020)
# Protocole eval identique à HADSF : 1 pos + 99 neg, nDCG@10

import sys
sys.path.append("/home/infres/belguith/PFE")

import argparse, math, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from rhg_data_binary import GraphData


# ── Argparse ────────────────────────────────────────────────────────────────
def config():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',    type=str, default='Musical_HADSF')
    p.add_argument('--embed_dim',  type=int, default=64)
    p.add_argument('--n_layers',   type=int, default=3)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--lambda_reg', type=float, default=1e-4)
    p.add_argument('--epochs',     type=int, default=1000)
    p.add_argument('--patience',   type=int, default=50)
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--device',     type=int, default=0)
    p.add_argument('--eval_mode',  type=str, default='sampled',
                   choices=['sampled', 'full'],
                   help='sampled=1 pos+99 neg (HADSF), full=tous les items (LightGCN officiel)')
    args = p.parse_args()
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    return args


# ── Normalized adjacency matrix ─────────────────────────────────────────────
def build_norm_adj(train_users, train_items, n_users, n_items, device):
    """Construit D^{-1/2} A D^{-1/2} comme sparse tensor PyTorch.
    A = [[0, R], [R^T, 0]]  de taille (n_users+n_items) x (n_users+n_items).
    """
    N = n_users + n_items
    # Indices des arêtes dans A
    row = torch.cat([train_users, train_items + n_users])
    col = torch.cat([train_items + n_users, train_users])
    vals = torch.ones(row.shape[0], dtype=torch.float32)

    # Degré de chaque nœud
    deg = torch.zeros(N, dtype=torch.float32)
    deg.scatter_add_(0, row, vals)

    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[d_inv_sqrt == float('inf')] = 0.0

    # Normalisation symétrique : A_norm[i,j] = d[i]^{-1/2} * d[j]^{-1/2}
    norm_vals = d_inv_sqrt[row] * vals * d_inv_sqrt[col]

    adj = torch.sparse_coo_tensor(
        torch.stack([row, col]), norm_vals, (N, N)
    ).coalesce().to(device)
    return adj


# ── LightGCN model ──────────────────────────────────────────────────────────
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, norm_adj, embed_dim=64, n_layers=3):
        super().__init__()
        self.n_users  = n_users
        self.n_items  = n_items
        self.n_layers = n_layers
        self.norm_adj = norm_adj  # sparse (N, N) sur device

        self.embedding_user = nn.Embedding(n_users, embed_dim)
        self.embedding_item = nn.Embedding(n_items, embed_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def computer(self):
        """Propagation LightGCN : K passes, moyenne des K+1 couches."""
        all_emb = torch.cat([
            self.embedding_user.weight,
            self.embedding_item.weight
        ])  # (n_users+n_items, dim)

        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            embs.append(all_emb)

        light_out = torch.mean(torch.stack(embs, dim=1), dim=1)
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items

    def bpr_loss(self, users_idx, pos_idx, neg_idx):
        all_users, all_items = self.computer()

        u_emb   = all_users[users_idx]
        pos_emb = all_items[pos_idx]
        neg_emb = all_items[neg_idx]

        # L2 reg sur E^(0) uniquement
        u0   = self.embedding_user.weight[users_idx]
        pos0 = self.embedding_item.weight[pos_idx]
        neg0 = self.embedding_item.weight[neg_idx]
        reg = (1/2) * (u0.norm(2).pow(2) +
                       pos0.norm(2).pow(2) +
                       neg0.norm(2).pow(2)) / len(users_idx)

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)
        loss = F.softplus(neg_scores - pos_scores).mean()
        return loss, reg


# ── Negative sampler (1:1, uniforme) ────────────────────────────────────────
def sample_negatives(users, train_seen, n_items, rng):
    """Pour chaque user, tire 1 négatif uniforme hors de ses items vus."""
    negs = []
    for u in users:
        seen = train_seen[u.item()]
        while True:
            neg = rng.integers(0, n_items)
            if neg not in seen:
                negs.append(neg)
                break
    return torch.tensor(negs, dtype=torch.long)


# ── Evaluation — 1 pos + 99 neg ─────────────────────────────────────────────
def _dcg(ranked, relevant, k):
    return sum(1.0 / math.log2(i + 2)
               for i, iid in enumerate(ranked[:k]) if iid in relevant)

def evaluate(model, edges_src, edges_dst, train_seen, n_items, device,
             n_neg=99, ks=(5, 10, 20)):
    model.eval()
    rng = np.random.default_rng(42)

    with torch.no_grad():
        all_users, all_items = model.computer()
        all_users = all_users.cpu()
        all_items = all_items.cpu()

    known_items = np.arange(n_items)
    # Grouper les edges par user
    user_pos = defaultdict(list)
    for u, i in zip(edges_src.tolist(), edges_dst.tolist()):
        user_pos[u].append(i)

    results = {k: {'ndcg': [], 'recall': [], 'hr': []} for k in ks}
    item_deg = defaultdict(int)
    for items in train_seen.values():
        for i in items:
            item_deg[i] += 1

    def group(iids):
        degs = [item_deg.get(i, 0) for i in iids]
        if all(d <= 10 for d in degs): return 'cold'
        return 'non_cold'

    grp_results = {g: {k: {'ndcg': [], 'recall': []} for k in ks}
                   for g in ('cold', 'non_cold')}

    for uid, pos_items in user_pos.items():
        relevant = set(pos_items)
        excluded = train_seen[uid] | relevant
        candidates = np.setdiff1d(known_items, list(excluded))
        if len(candidates) == 0:
            continue
        neg_ids = rng.choice(candidates, size=min(n_neg, len(candidates)),
                             replace=False).tolist()
        pool = list(relevant) + neg_ids
        pool_t = torch.tensor(pool, dtype=torch.long)
        u_emb = all_users[uid]
        scores = (all_items[pool_t] @ u_emb).tolist()
        ranked = [iid for iid, _ in
                  sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)]
        g = group(list(relevant))

        for k in ks:
            hits = sum(1 for iid in ranked[:k] if iid in relevant)
            dcg  = _dcg(ranked, relevant, k)
            idcg = sum(1.0 / math.log2(i + 2)
                       for i in range(min(len(relevant), k)))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            rec  = hits / len(relevant)
            results[k]['ndcg'].append(ndcg)
            results[k]['recall'].append(rec)
            results[k]['hr'].append(1.0 if hits > 0 else 0.0)
            grp_results[g][k]['ndcg'].append(ndcg)
            grp_results[g][k]['recall'].append(rec)

    n_eval = len(results[ks[0]]['ndcg'])
    global_ndcg   = {k: float(np.mean(results[k]['ndcg']))   for k in ks}
    global_recall = {k: float(np.mean(results[k]['recall'])) for k in ks}
    cold_ndcg     = {k: float(np.mean(grp_results['cold'][k]['ndcg']))
                     if grp_results['cold'][k]['ndcg'] else 0.0 for k in ks}
    cold_recall   = {k: float(np.mean(grp_results['cold'][k]['recall']))
                     if grp_results['cold'][k]['recall'] else 0.0 for k in ks}
    noncold_ndcg  = {k: float(np.mean(grp_results['non_cold'][k]['ndcg']))
                     if grp_results['non_cold'][k]['ndcg'] else 0.0 for k in ks}
    noncold_recall= {k: float(np.mean(grp_results['non_cold'][k]['recall']))
                     if grp_results['non_cold'][k]['recall'] else 0.0 for k in ks}
    return n_eval, global_ndcg, global_recall, cold_ndcg, cold_recall, noncold_ndcg, noncold_recall


# ── Full ranking evaluation (protocole officiel LightGCN) ───────────────────
def evaluate_full(model, edges_src, edges_dst, train_seen, n_users, n_items,
                  device, ks=(5, 10, 20)):
    model.eval()
    with torch.no_grad():
        all_users, all_items = model.computer()

    item_deg = defaultdict(int)
    for items in train_seen.values():
        for i in items:
            item_deg[i] += 1

    def group(iids):
        degs = [item_deg.get(i, 0) for i in iids]
        if all(5 <= d <= 10 for d in degs): return 'cold'
        if all(d > 10 for d in degs): return 'warm'
        return None

    user_pos = defaultdict(list)
    for u, i in zip(edges_src.tolist(), edges_dst.tolist()):
        user_pos[u].append(i)

    results     = {k: {'ndcg': [], 'recall': []} for k in ks}
    grp_results = {g: {k: {'ndcg': [], 'recall': []} for k in ks}
                   for g in ('cold', 'warm', 'non_cold')}

    items_emb = all_items.to(device)  # (n_items, dim)

    for uid, pos_items in user_pos.items():
        relevant = set(pos_items)
        u_emb = all_users[uid].to(device)
        scores = (items_emb @ u_emb).cpu()

        # Masquer les items vus en train
        for seen_i in train_seen[uid]:
            if seen_i < n_items:
                scores[seen_i] = -1e9

        _, topk_idx = torch.topk(scores, max(ks))
        ranked = topk_idx.tolist()
        g = group(list(relevant))

        for k in ks:
            hits = sum(1 for iid in ranked[:k] if iid in relevant)
            dcg  = _dcg(ranked, relevant, k)
            idcg = sum(1.0/math.log2(i+2) for i in range(min(len(relevant), k)))
            ndcg = dcg/idcg if idcg > 0 else 0.0
            rec  = hits / len(relevant)
            results[k]['ndcg'].append(ndcg)
            results[k]['recall'].append(rec)
            if g is not None:
                grp_results[g][k]['ndcg'].append(ndcg)
                grp_results[g][k]['recall'].append(rec)
                if g == 'warm':
                    grp_results['non_cold'][k]['ndcg'].append(ndcg)
                    grp_results['non_cold'][k]['recall'].append(rec)
            elif g is None and all(item_deg.get(i,0) > 10 for i in relevant):
                grp_results['non_cold'][k]['ndcg'].append(ndcg)
                grp_results['non_cold'][k]['recall'].append(rec)

    n_eval = len(results[ks[0]]['ndcg'])
    global_ndcg   = {k: float(np.mean(results[k]['ndcg']))    for k in ks}
    global_recall = {k: float(np.mean(results[k]['recall']))  for k in ks}
    cold_ndcg     = {k: float(np.mean(grp_results['cold'][k]['ndcg']))
                     if grp_results['cold'][k]['ndcg'] else 0.0 for k in ks}
    cold_recall   = {k: float(np.mean(grp_results['cold'][k]['recall']))
                     if grp_results['cold'][k]['recall'] else 0.0 for k in ks}
    noncold_ndcg  = {k: float(np.mean(grp_results['non_cold'][k]['ndcg']))
                     if grp_results['non_cold'][k]['ndcg'] else 0.0 for k in ks}
    noncold_recall= {k: float(np.mean(grp_results['non_cold'][k]['recall']))
                     if grp_results['non_cold'][k]['recall'] else 0.0 for k in ks}
    return n_eval, global_ndcg, global_recall, cold_ndcg, cold_recall, noncold_ndcg, noncold_recall


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args = config()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Data ────────────────────────────────────────────────────────────────
    dataset_path = f'/home/infres/belguith/PFE/processed/{args.dataset.replace("_HADSF","")}_reviews_with_aspects.jsonl'
    data = GraphData(args.dataset, dataset_path)
    n_users = data.user_size
    n_items = data.item_size
    print(f"[DATA] {args.dataset}: {n_users} users, {n_items} items", flush=True)

    # Edges train/valid/test
    train_u, train_i = data.graph['train'].edges()
    valid_u, valid_i = data.graph['valid'].edges()
    test_u,  test_i  = data.graph['test'].edges()

    # train_seen : user → set(item_ids)
    train_seen = defaultdict(set)
    for u, i in zip(train_u.tolist(), train_i.tolist()):
        train_seen[u].add(i)

    # ── Normalized adjacency ─────────────────────────────────────────────────
    norm_adj = build_norm_adj(train_u, train_i, n_users, n_items, args.device)
    print(f"[GRAPH] norm_adj: {norm_adj.shape}, nnz={norm_adj._nnz()}", flush=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = LightGCN(n_users, n_items, norm_adj,
                     embed_dim=args.embed_dim, n_layers=args.n_layers).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[MODEL] LightGCN: embed_dim={args.embed_dim}, n_layers={args.n_layers}, "
          f"lr={args.lr}, lambda_reg={args.lambda_reg}, batch={args.batch_size}", flush=True)

    # ── Degree tensor for cold/warm logging (depuis les edges train du graphe) ─
    _deg_arr = np.zeros(n_items, dtype=np.int32)
    for iid in train_i.tolist():
        if iid < n_items:
            _deg_arr[iid] += 1
    n_cold = int(((5 <= _deg_arr) & (_deg_arr <= 10)).sum())
    n_warm = int((_deg_arr > 20).sum())
    print(f"[DIAG_SETUP] cold_items={n_cold}, warm_items={n_warm}", flush=True)

    # ── Training ─────────────────────────────────────────────────────────────
    train_users_arr = train_u.numpy()
    train_items_arr = train_i.numpy()
    n_train = len(train_users_arr)

    best_valid_ndcg = 0.0
    best_test_ndcg  = 0.0
    no_improve = 0
    rng_neg = np.random.default_rng(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Shuffle
        perm = np.random.permutation(n_train)
        total_loss = 0.0
        n_batches  = 0

        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            u_batch = torch.tensor(train_users_arr[idx], dtype=torch.long, device=args.device)
            p_batch = torch.tensor(train_items_arr[idx], dtype=torch.long, device=args.device)
            n_batch = sample_negatives(u_batch.cpu(), train_seen, n_items, rng_neg).to(args.device)

            optimizer.zero_grad()
            bpr, reg = model.bpr_loss(u_batch, p_batch, n_batch)
            loss = bpr + args.lambda_reg * reg
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / n_batches

        # ── Valid eval ───────────────────────────────────────────────────────
        eval_fn = evaluate_full if args.eval_mode == 'full' else evaluate
        eval_kwargs = dict(train_seen=train_seen, n_items=n_items, device=args.device) if args.eval_mode == 'sampled' \
                      else dict(train_seen=train_seen, n_users=n_users, n_items=n_items, device=args.device)

        if args.eval_mode == 'full':
            n_eval, g_ndcg, g_rec, cold_ndcg, cold_rec, nc_ndcg, nc_rec = eval_fn(
                model, valid_u, valid_i, **eval_kwargs)
        else:
            n_eval, g_ndcg, g_rec, cold_ndcg, cold_rec, nc_ndcg, nc_rec = eval_fn(
                model, valid_u, valid_i, **eval_kwargs)
        valid_ndcg10 = g_ndcg[10]

        if valid_ndcg10 > best_valid_ndcg:
            best_valid_ndcg = valid_ndcg10
            _, g_t, g_t_rec, cold_t, cold_t_rec, nc_t, nc_t_rec = eval_fn(
                model, test_u, test_i, **eval_kwargs)
            best_recall = g_t_rec[10]
            best_cold_recall = cold_t_rec[10]
            best_noncold_recall = nc_t_rec[10]
            best_test_ndcg = g_t[10]
            best_cold = cold_t[10]
            best_noncold = nc_t[10]
            best_ep = epoch
            no_improve = 0
        else:
            no_improve += 1

        print(f"Epoch={epoch:>4d}  loss={avg_loss:.4f}  "
              f"Valid_nDCG@10={valid_ndcg10:.4f}  Best={best_valid_ndcg:.4f}",
              flush=True)

        if no_improve >= args.patience:
            print(f"Early stopping @ep{epoch} (patience={args.patience})", flush=True)
            break

    # ── Final results ─────────────────────────────────────────────────────────
    print(f"\n── ITEM RANKING ({args.eval_mode}) ──", flush=True)
    print(f"  Best ep={best_ep}", flush=True)
    print(f"  nDCG@10   Global={best_test_ndcg:.4f}  Cold={best_cold:.4f}  Non-cold={best_noncold:.4f}", flush=True)
    print(f"  Recall@10 Global={best_recall:.4f}  Cold={best_cold_recall:.4f}  Non-cold={best_noncold_recall:.4f}", flush=True)
    print(f"\nBest Valid nDCG@10={best_valid_ndcg:.4f}", flush=True)
    print(f"Best Test  nDCG@10={best_test_ndcg:.4f}", flush=True)


if __name__ == '__main__':
    main()

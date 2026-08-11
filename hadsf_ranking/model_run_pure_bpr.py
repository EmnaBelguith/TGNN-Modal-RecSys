"""
Pure BPR baseline — même graphe que 535 (rating edges 1-5 + reverse),
aucun avis dans les messages, aucun modal, aucune MI/GRCA/f_loss.
Loss = BPR + L2 uniquement.
"""
import random, os, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from abc import ABC
from collections import defaultdict
from tqdm import tqdm

from rhg_data import GraphData          # ← même graphe que 535 (10 edge types)
from util import get_logger, args_to_str

# ── Config ────────────────────────────────────────────────────────────────────

def config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--batch_size',  type=int,   default=512)
    parser.add_argument('--lambda_l2',   type=float, default=1e-4)
    parser.add_argument('--n_neg',       type=int,   default=5)
    parser.add_argument('--gcn_dropout', type=float, default=0.8)
    parser.add_argument('--num_layers',  type=int,   default=2)
    parser.add_argument('--run_tag',     type=str,   default='')
    parser.add_argument('--device',      type=int,   default=0)
    parser.add_argument('--epoch',       type=int,   default=1000)
    parser.add_argument('--train_lr',    type=float, default=0.001)
    parser.add_argument('--train_early_stopping_patience', type=int, default=100)
    parser.add_argument('--review_feat_size', type=int, default=128)

    params = parser.parse_args()
    params.device        = torch.device(f'cuda:{params.device}')
    params.dataset_name  = 'Musical_HADSF'
    params.dataset_path  = '/home/infres/belguith/PFE/processed/Musical_reviews_with_aspects.jsonl'
    params.gcn_out_units = params.review_feat_size
    params.model_short_name = (
        f'pure_bpr_layers{params.num_layers}_seed{params.seed}'
        f'_l2{params.lambda_l2}_bs{params.batch_size}_n{params.n_neg}{params.run_tag}'
    )
    params.model_save_path = (
        f'model_save/{params.dataset_name}/{params.model_short_name}.pt'
    )
    os.makedirs(f'model_save/{params.dataset_name}', exist_ok=True)
    return params

# ── GCN Conv — attention uniforme, aucun avis ─────────────────────────────────

class GCMCGraphConv(nn.Module, ABC):
    """Identique à l'original avec no_review=True forcé — pa=1 partout."""

    def __init__(self, feature_size, add_embedding_mapping=False, dropout_rate=0.0):
        super().__init__()
        self.embedding_mapping = (nn.Linear(feature_size, feature_size)
                                  if add_embedding_mapping else None)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear  = nn.Linear(feature_size, feature_size)
        self._last_pre_ci = None

    def forward(self, graph, feat):
        with graph.local_scope():
            h = self.embedding_mapping(feat) if self.embedding_mapping else feat
            graph.srcdata['h'] = h
            # attention uniforme — aucune feature d'avis
            graph.edata['pa'] = torch.ones(graph.num_edges(), 1, device=feat.device)
            graph.update_all(
                lambda edges: {'m': edges.src['h'] * edges.data['pa']
                               * self.dropout(edges.src['cj'])},
                fn.sum(msg='m', out='h')
            )
            rst = graph.dstdata['h']
            self._last_pre_ci = rst.detach()
            rst = rst * graph.dstdata['ci']
            rst = self.linear(rst)
        return rst

# ── GCN Encoder — itère sur les 5 types de rating comme 535 ─────────────────

class GCNEncoder(nn.Module):

    def __init__(self, rating_values, user_size, item_size,
                 msg_units, num_layers, dropout_rate=0.0):
        super().__init__()
        self.num_layers   = num_layers
        self.rating_values = [str(r) for r in rating_values]

        self.user_embedding = nn.Parameter(torch.Tensor(user_size, msg_units))
        self.item_embedding = nn.Parameter(torch.Tensor(item_size, msg_units))
        nn.init.xavier_uniform_(self.user_embedding.unsqueeze(0)).squeeze(0)
        nn.init.xavier_uniform_(self.item_embedding.unsqueeze(0)).squeeze(0)

        self.conv_layers = nn.ModuleList()
        for l in range(num_layers):
            sub_conv = {}
            for r in self.rating_values:
                sub_conv[r]          = GCMCGraphConv(msg_units,
                                                      add_embedding_mapping=(l == 0),
                                                      dropout_rate=dropout_rate)
                sub_conv[f'rev-{r}'] = GCMCGraphConv(msg_units,
                                                      add_embedding_mapping=(l == 0),
                                                      dropout_rate=dropout_rate)
            self.conv_layers.append(nn.ModuleDict(sub_conv))

        self.ufc     = nn.Linear(msg_units, msg_units)
        self.ifc     = nn.Linear(msg_units, msg_units)
        self.dropout = nn.Dropout(0.5)
        self.agg_act = nn.GELU()

    def forward(self, input_nodes, blocks):
        user_outputs_per_layer = []
        item_outputs_per_layer = []

        for l, conv_layer in enumerate(self.conv_layers):
            block = blocks[l]
            u_layer, i_layer = {}, {}
            for r in self.rating_values:
                if l == 0:
                    u_src = self.user_embedding[input_nodes['user']]
                    i_src = self.item_embedding[input_nodes['item']]
                else:
                    u_src = user_outputs_per_layer[-1][r]
                    i_src = item_outputs_per_layer[-1][r]
                i_layer[r] = conv_layer[r](block['user', r, 'item'], u_src)
                u_layer[r] = conv_layer[f'rev-{r}'](block['item', f'rev-{r}', 'user'], i_src)
            user_outputs_per_layer.append(u_layer)
            item_outputs_per_layer.append(i_layer)

        # Somme sur tous les types de rating (même que 535)
        user_out = sum(user_outputs_per_layer[-1].values())
        item_out = sum(item_outputs_per_layer[-1].values())

        user_out = self.agg_act(user_out)
        user_out = self.dropout(user_out)
        user_out = self.ufc(user_out)

        item_out = self.agg_act(item_out)
        item_out = self.dropout(item_out)
        item_out = self.ifc(item_out)

        return user_out, item_out

# ── Scorer MLP — identique à rating_linear + item_scorer de 535 ──────────────

class ItemScorer(nn.Module):

    def __init__(self, in_units):
        super().__init__()
        self.rating_linear = nn.Sequential(
            nn.Linear(in_units * 2, in_units, bias=False),
            nn.ReLU(),
            nn.Linear(in_units, in_units, bias=False),
        )
        self.item_scorer = nn.Linear(in_units, 1, bias=False)

    def score(self, u_emb, i_emb):
        h = self.rating_linear(torch.cat([u_emb, i_emb], dim=1))
        return self.item_scorer(h).squeeze(-1)

# ── Net ───────────────────────────────────────────────────────────────────────

class Net(nn.Module):

    def __init__(self, rating_values, params):
        super().__init__()
        self.lambda_l2 = params.lambda_l2
        self.n_neg     = params.n_neg
        self.encoder   = GCNEncoder(
            rating_values,
            params.user_size, params.item_size,
            params.gcn_out_units, params.num_layers,
            dropout_rate=params.gcn_dropout
        )
        self.scorer = ItemScorer(params.gcn_out_units)

    def calc_loss(self, input_nodes, blocks, pos_graph):
        self.train()
        urf, irf = self.encoder(input_nodes, blocks)

        src_pos, dst_pos = pos_graph.edges()
        u_emb     = urf[src_pos]
        i_pos_emb = irf[dst_pos]
        N = i_pos_emb.shape[0]

        bpr_terms = []
        for _ in range(self.n_neg):
            perm = torch.randperm(N, device=i_pos_emb.device)
            clash = perm == torch.arange(N, device=i_pos_emb.device)
            if clash.any():
                perm[clash] = (perm[clash] + 1) % N
            i_neg_emb = i_pos_emb[perm]
            s_pos = self.scorer.score(u_emb, i_pos_emb)
            s_neg = self.scorer.score(u_emb, i_neg_emb)
            bpr_terms.append(F.logsigmoid(s_pos - s_neg))

        bpr_loss = -torch.stack(bpr_terms).mean(0).mean()
        l2_reg = self.lambda_l2 * (
            self.encoder.user_embedding.norm(2).pow(2) +
            self.encoder.item_embedding.norm(2).pow(2)
        ) / u_emb.shape[0]
        return bpr_loss + l2_reg

    @torch.no_grad()
    def evaluate_ranking_ndcg(self, dataloader, dataset, K=10,
                               relevance_threshold=1, etype='valid',
                               n_neg=99, seed=42):
        device = next(self.parameters()).device
        self.eval()
        rng = np.random.default_rng(seed)

        graph = dataset.graph
        train_seen = defaultdict(set)
        for r in self.encoder.rating_values:
            tu, ti = graph[r].edges()
            for u, i in zip(tu.tolist(), ti.tolist()):
                train_seen[u].add(i)

        user_emb, item_emb, pos_items = {}, {}, defaultdict(dict)
        for input_nodes, pos_graph, _neg_graph, blocks in dataloader:
            inp = {k: v.to(device) for k, v in input_nodes.items()}
            pg  = pos_graph[etype].to(device)
            blk = [b.to(device) for b in blocks]
            urf, irf = self.encoder(inp, blk)
            for lu, gu in enumerate(pg.nodes['user'].data['_ID'].cpu().tolist()):
                if gu not in user_emb: user_emb[gu] = urf[lu].cpu()
            for li, gi in enumerate(pg.nodes['item'].data['_ID'].cpu().tolist()):
                if gi not in item_emb: item_emb[gi] = irf[li].cpu()
            src_idx, dst_idx = pg.edges()
            for uid, iid, r in zip(pg.srcdata['_ID'][src_idx].cpu().tolist(),
                                    pg.dstdata['_ID'][dst_idx].cpu().tolist(),
                                    pg.edata['rating'].cpu().tolist()):
                pos_items[uid][iid] = r

        import pandas as pd
        _deg = pd.read_csv('/home/infres/belguith/PFE/processed/Musical_interactions_reviews.csv'
                           ).groupby('iid').size().to_dict()

        scorer = self.scorer.to('cpu')
        def score_pairs(u_t, i_t):
            u_rep = u_t.unsqueeze(0).expand(i_t.shape[0], -1)
            return scorer.score(u_rep, i_t)

        known = np.array(sorted(item_emb.keys()))
        buckets = {'global': [], 'cold': [], 'medium': [], 'warm': [], 'non_cold': []}

        for uid, items in pos_items.items():
            if uid not in user_emb: continue
            relevant = {iid for iid, r in items.items() if r >= relevance_threshold}
            if not relevant or not all(iid in item_emb for iid in relevant): continue
            excluded   = train_seen[uid] | set(items.keys())
            candidates = np.setdiff1d(known, list(excluded), assume_unique=True)
            if len(candidates) == 0: continue
            neg_ids  = rng.choice(candidates, size=min(n_neg, len(candidates)), replace=False).tolist()
            all_ids  = list(relevant) + neg_ids
            all_embs = torch.stack([item_emb[i] for i in all_ids])
            scores   = score_pairs(user_emb[uid], all_embs).tolist()
            ranked   = [iid for iid, _ in sorted(zip(all_ids, scores),
                                                   key=lambda x: x[1], reverse=True)]
            ideal_n  = len(relevant)
            dcg  = sum(1.0 / math.log2(i+2) for i, iid in enumerate(ranked[:K]) if iid in relevant)
            idcg = sum(1.0 / math.log2(i+2) for i in range(min(ideal_n, K)))
            v = dcg / idcg if idcg > 0 else 0.0
            buckets['global'].append(v)
            degs = [_deg.get(iid, 0) for iid in relevant]
            if   all(5 <= d <= 10 for d in degs):   buckets['cold'].append(v)
            elif all(11 <= d <= 20 for d in degs):   buckets['medium'].append(v)
            elif all(d > 20 for d in degs):           buckets['warm'].append(v)
            if not all(5 <= d <= 10 for d in degs):  buckets['non_cold'].append(v)

        scorer.to(device)
        return {k: float(np.mean(v)) if v else 0.0 for k, v in buckets.items()}

# ── Training ──────────────────────────────────────────────────────────────────

def train(params):
    random.seed(params.seed); np.random.seed(params.seed)
    torch.manual_seed(params.seed); torch.cuda.manual_seed_all(params.seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"[SEED] {params.seed}", flush=True)

    global logger
    logger = get_logger(params.model_short_name, None)
    logger.info(f"Parameters:\n{args_to_str(params)}")

    dataset = GraphData(params.dataset_name, params.dataset_path)
    params.user_size = dataset.user_size
    params.item_size = dataset.item_size
    rating_values    = dataset.possible_rating_values
    print(f"[DATA] users={params.user_size} items={params.item_size} "
          f"ratings={list(rating_values)}", flush=True)

    net = Net(rating_values, params).to(params.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=params.train_lr)
    logger.info("Network ready.\n")

    train_dl, valid_dl, test_dl = dataset.get_dataloaders(
        batch_size=params.batch_size, num_layers=params.num_layers
    )

    best_valid = 0.0; best_test_res = {}; no_better = 0; best_iter = -1

    for iter_idx in range(1, params.epoch + 1):
        net.train()
        pbar = tqdm(train_dl, desc=f'Ep {iter_idx}', leave=False)
        r_loss = None
        for input_nodes, pos_graph, _neg_graph, blocks in pbar:
            inp = {k: v.to(params.device) for k, v in input_nodes.items()}
            pg  = pos_graph['train'].to(params.device)
            blk = [b.to(params.device) for b in blocks]
            optimizer.zero_grad()
            r_loss = net.calc_loss(inp, blk, pg)
            r_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

        res_v = net.evaluate_ranking_ndcg(valid_dl, dataset, K=10, etype='valid')
        valid_ndcg = res_v['global']
        log_str = (f"Epoch={iter_idx:>3d}, Train_BPR={r_loss.item():.4f}, "
                   f"Valid_nDCG@10={valid_ndcg:.4f}")

        if valid_ndcg > best_valid:
            best_valid = valid_ndcg
            no_better  = 0
            best_iter  = iter_idx
            res_t = net.evaluate_ranking_ndcg(test_dl, dataset, K=10, etype='test')
            best_test_res = res_t
            log_str += (f", Test_nDCG@10={res_t['global']:.4f}"
                        f" [cold={res_t['cold']:.4f} non_cold={res_t['non_cold']:.4f}]")
            torch.save(net.state_dict(), params.model_save_path)
        else:
            no_better += 1
            if no_better >= params.train_early_stopping_patience:
                logger.info("Early stopping.")
                break

        logger.info(log_str)

    logger.info(f'Best Iter={best_iter}  Best Valid={best_valid:.4f}')

    print("\n── ITEM RANKING  (1 pos + 99 neg, rating≥1) ──", flush=True)
    print(f"  Best epoch={best_iter}", flush=True)
    for k, v in best_test_res.items():
        print(f"  [{k:8s}] nDCG@10 = {v:.4f}", flush=True)


if __name__ == '__main__':
    params = config()
    train(params)

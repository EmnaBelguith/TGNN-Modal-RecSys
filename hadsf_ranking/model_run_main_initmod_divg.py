# -*- coding: utf-8 -*-

import sys
sys.path.append("/home/infres/belguith/PFE")
from modal_encoder import ModalEncoder, load_modal_features
import torch.nn.functional as F

import argparse
import numpy as np
from abc import ABC
import os
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from rhg_data import GraphData
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from tqdm import tqdm
from util import get_logger, args_to_dict, args_to_str
from collections import defaultdict


def config():
    # ── Params modifiables via CLI ──────────────────────────────────────────
    parser = argparse.ArgumentParser(description='Full ranking model — reference config')
    parser.add_argument('--seed',              type=int,   default=42)
    parser.add_argument('--batch_size',        type=int,   default=512)
    parser.add_argument('--lambda_l2',         type=float, default=1e-4)
    parser.add_argument('--lambda_mi',         type=float, default=0.01)
    parser.add_argument('--lambda_mi_warmup',  type=int,   default=20)
    parser.add_argument('--neg_strategy',      type=str,   default='inbatch',
                        choices=['random', 'multi_random', 'inbatch'])
    parser.add_argument('--n_neg',             type=int,   default=5)
    parser.add_argument('--gcn_dropout',       type=float, default=0.8)
    parser.add_argument('--num_layers',        type=int,   default=2)
    parser.add_argument('--run_tag',           type=str,   default='',
                        help='Suffix appended to checkpoint filename (e.g. v3)')
    parser.add_argument('--test_only',         action='store_true', default=False)
    parser.add_argument('--no_modal',          action='store_true', default=False)
    parser.add_argument('--lambda_grca',       type=float, default=0.1,
                        help='Poids de la loss GRCA (InfoNCE modal↔collab pondérée par (1-alpha))')

    # ── Params fixes (rarement changés) ─────────────────────────────────────
    parser.add_argument('--device',                      type=int,   default=0)
    parser.add_argument('--epoch',                       type=int,   default=200)
    parser.add_argument('--train_grad_clip',             type=float, default=1.0)
    parser.add_argument('--train_lr',                    type=float, default=0.001)
    parser.add_argument('--train_min_lr',                type=float, default=0.0001)
    parser.add_argument('--train_lr_decay_factor',       type=float, default=0.5)
    parser.add_argument('--train_decay_patience',        type=int,   default=8)
    parser.add_argument('--train_early_stopping_patience', type=int, default=10)
    parser.add_argument('--review_feat_size',            type=int,   default=128)
    parser.add_argument('--lambda_f',                    type=float, default=1e-4)
    parser.add_argument('--ed_alpha',                    type=float, default=2.0)
    parser.add_argument('--w_deg_init',                  type=float, default=2.0)
    parser.add_argument('--b_deg_init',                  type=float, default=-5.0)

    args = parser.parse_args()

    # ── Config fixe dataset ─────────────────────────────────────────────────
    args.dataset_name = 'Musical_HADSF'
    args.dataset_path = '/home/infres/belguith/PFE/processed/Musical_reviews_with_aspects.jsonl'
    args.model_short_name = 'RHGC4_ranking'
    args.device = f"cuda:{args.device}" if args.device >= 0 else 'cpu'
    args.gcn_out_units = args.review_feat_size

    # ── Nom du checkpoint (tracabilité hyperparams) ─────────────────────────
    _l2_tag   = f'_l2{args.lambda_l2}'   if args.lambda_l2 > 0            else ''
    _bs_tag   = f'_bs{args.batch_size}'
    _mi_tag   = f'_mi{args.lambda_mi}'   if args.lambda_mi > 0            else ''
    _dp_tag   = f'_dp{args.gcn_dropout}' if args.gcn_dropout != 0.8       else ''
    _neg_tag  = f'_{args.neg_strategy}'  if args.neg_strategy != 'random' else ''
    _nneg_tag = f'_k{args.n_neg}'        if args.neg_strategy == 'multi_random' else ''
    _warm_tag = f'_warm{args.lambda_mi_warmup}' if args.lambda_mi_warmup > 0 else ''
    _nmod_tag = '_nomodal'               if args.no_modal                 else ''
    _run_tag  = f'_{args.run_tag}'       if args.run_tag                  else ''
    _grca_tag = f'_grca{args.lambda_grca}' if args.lambda_grca > 0       else ''
    args.model_save_path = (
        f'model_save/{args.dataset_name}/{args.model_short_name}'
        f'_layers_{args.num_layers}_seed{args.seed}'
        f'{_l2_tag}{_bs_tag}{_mi_tag}{_dp_tag}{_neg_tag}{_nneg_tag}{_warm_tag}{_nmod_tag}{_grca_tag}{_run_tag}.pt'
    )
    os.makedirs(f'model_save/{args.dataset_name}', exist_ok=True)

    return args


def reset_parameters(model):
    em_set = set(['review_embedding.weight', 'sentence_embedding.weight'])
    for n, p in model.named_parameters():
        if n in em_set:
            continue
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def format_dict_to_str(data_dict):
    result = []
    for k, v in data_dict.items():
        result.append(f'{k}: {v:>.4f}')
    return ', '.join(result)


class GCMCGraphConv(nn.Module, ABC):

    def __init__(self,
                 feature_size,
                 review_embedding,
                 add_embedding_mapping=False,
                 add_review=False,
                 dropout_rate=0.0):
        super(GCMCGraphConv, self).__init__()

        self.embedding_mapping = nn.Linear(feature_size, feature_size) if add_embedding_mapping else None
        self.prob_score = nn.Linear(128, 1, bias=False)
        self.review_embedding = review_embedding
        if add_review:
            self.review_w = nn.Sequential(
                nn.Linear(128, feature_size, bias=False),
                nn.GELU(),
                nn.Linear(feature_size, feature_size, bias=False),
                nn.GELU(),
                nn.Linear(feature_size, feature_size, bias=False),
            )
            self.review_score = nn.Linear(128, 1, bias=False)
        else:
            self.review_w = None
            self.review_score = None
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(feature_size, feature_size)

    def get_review_feature(self, rid):
        num_embeddings = self.review_embedding.num_embeddings
        assert torch.all(rid < num_embeddings), \
            f"review_id hors range: max={rid.max().item()}, num_embeddings={num_embeddings}"
        return self.review_embedding(rid)

    def forward(self, graph, feat):
        with graph.local_scope():
            graph.srcdata['h'] = self.embedding_mapping(feat) if self.embedding_mapping else feat

            review_feat = self.get_review_feature(graph.edata['review_id'])
            graph.edata['pa'] = torch.sigmoid(self.prob_score(review_feat))

            if self.review_w is not None:
                graph.edata['ra'] = torch.sigmoid(self.review_score(review_feat))
                graph.edata['rf'] = self.review_w(review_feat)
                graph.update_all(lambda edges: {'m': (edges.src['h'] * edges.data['pa']
                                                      + edges.data['rf'] * edges.data['ra'])
                                                     * self.dropout(edges.src['cj'])},
                                 fn.sum(msg='m', out='h'))
            else:
                graph.update_all(lambda edges: {'m': edges.src['h'] * edges.data['pa']
                                                     * self.dropout(edges.src['cj'])},
                                 fn.sum(msg='m', out='h'))

            rst = graph.dstdata['h']
            self._last_pre_ci = rst.detach()   # pre-ci signal: norm scales with sqrt(degree)
            rst = rst * graph.dstdata['ci']
            rst = self.linear(rst)
        return rst


class MultiLayerHeteroGraphConv(nn.Module):

    def __init__(self, rating_values, review_embedding, user_size, item_size, msg_units, num_layers, aggregate='sum', dropout_rate=0.0):
        super(MultiLayerHeteroGraphConv, self).__init__()

        assert num_layers > 0, "The number of conv layers must have at least one!"
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        rating_values = [str(r) for r in rating_values]
        self.rating_values = rating_values

        self.user_embedding = nn.Parameter(torch.Tensor(user_size, msg_units))
        self.item_embedding = nn.Parameter(torch.Tensor(item_size, msg_units))
        nn.init.xavier_uniform_(self.item_embedding.unsqueeze(0)).squeeze(0)
        self.h_modal = None
        self.item_degree_tensor = None
        self._epoch_alpha_acc  = {}
        self._epoch_mc_sum     = 0.0
        self._epoch_cc_sum     = 0.0
        self._epoch_contrib_n  = 0
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        self.gate_temp = nn.Parameter(torch.tensor(2.0))
        self.modal_adapter = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        nn.init.zeros_(self.modal_adapter[-1].weight)
        nn.init.zeros_(self.modal_adapter[-1].bias)

        for l in range(num_layers):
            sub_conv = {}
            for rating in rating_values:
                rating = str(rating)
                rev_rating = f'rev-{rating}'
                sub_conv[rating] = GCMCGraphConv(msg_units,
                                                 review_embedding,
                                                 add_embedding_mapping=l == 0,
                                                 add_review=l == (num_layers - 1),
                                                 dropout_rate=dropout_rate)
                sub_conv[rev_rating] = GCMCGraphConv(msg_units,
                                                     review_embedding,
                                                     add_embedding_mapping=l == 0,
                                                     add_review=l == (num_layers - 1),
                                                     dropout_rate=dropout_rate)
            self.conv_layers.append(nn.ModuleDict(sub_conv))

        self.ufc = nn.Linear(msg_units, msg_units)
        self.ifc = nn.Linear(msg_units, msg_units)
        self.dropout = nn.Dropout(0.5)
        self.agg_act = nn.GELU()

    def reset_epoch_diag(self):
        self._epoch_alpha_acc = {}
        self._epoch_mc_sum    = 0.0
        self._epoch_cc_sum    = 0.0
        self._epoch_contrib_n = 0

    def forward(self, input_nodes, encoder_blocks):
        user_outputs = []
        item_outputs = []
        _item_collab_out = None

        for l in range(len(self.conv_layers)):
            u_layer_output = dict()
            m_layer_output = dict()
            block = encoder_blocks[l]
            conv_layer = self.conv_layers[l]

            for rating in self.rating_values:
                if l == 0:
                    i_o = conv_layer[rating](block['user', rating, 'item'],
                                             self.user_embedding[input_nodes['user']])
                    item_collab = self.item_embedding[input_nodes['item']]
                    if _item_collab_out is None:
                        _item_collab_out = item_collab
                    u_o = conv_layer[f'rev-{rating}'](block['item', f'rev-{rating}', 'user'],
                                                      item_collab)
                else:
                    _u_feats = user_outputs[-1][rating]
                    _i_feats = item_outputs[-1][rating]
                    i_o = conv_layer[rating](block['user', rating, 'item'], _u_feats)
                    u_o = conv_layer[f'rev-{rating}'](block['item', f'rev-{rating}', 'user'], _i_feats)

                m_layer_output[rating] = i_o
                u_layer_output[rating] = u_o

            user_outputs.append(u_layer_output)
            item_outputs.append(m_layer_output)

        user_outputs = sum(list(user_outputs[-1].values()))
        item_outputs = sum(list(item_outputs[-1].values()))

        _seed_gids       = encoder_blocks[-1].dstdata['_ID']['item']
        _degrees_out     = self.item_degree_tensor[_seed_gids]
        _degree_norm_out = (_degrees_out / self.item_degree_tensor.max().clamp(min=1.0)).unsqueeze(-1)
        _item_modal_out  = self.h_modal[_seed_gids]

        user_outputs = self.agg_act(user_outputs)
        user_outputs = self.dropout(user_outputs)
        user_outputs = self.ufc(user_outputs)

        # divergence gate : norm(item_post_gcn) / norm_max(h_modal) — direct, no degree proxy
        _collab_norm  = item_outputs.norm(dim=-1, keepdim=True)
        _modal_norm_max = self.h_modal.norm(dim=-1).detach().max().clamp(min=1e-8)
        _div_signal   = (_collab_norm / _modal_norm_max).clamp(max=1.0)
        alpha = torch.sigmoid(F.softplus(self.gate_temp) * self.gate_mlp(_div_signal))
        if self.training:
            self._last_alpha = alpha.squeeze(-1).detach()
            self._gate_step = getattr(self, '_gate_step', 0) + 1
            # accumulate post-GCN alpha per item for epoch-level diagnostics
            for _g, _av in zip(_seed_gids.cpu().tolist(), alpha.squeeze(-1).detach().cpu().tolist()):
                self._epoch_alpha_acc[_g] = _av
            _mc = ((1 - alpha) * _item_modal_out).norm(dim=-1).detach().cpu()
            _cc = (alpha * item_outputs).norm(dim=-1).detach().cpu()
            self._epoch_mc_sum    += _mc.sum().item()
            self._epoch_cc_sum    += _cc.sum().item()
            self._epoch_contrib_n += _mc.shape[0]
            if self._gate_step % 100 == 1:
                _cold_mask = (_degrees_out >= 5) & (_degrees_out <= 10)
                _warm_mask = _degrees_out > 20
                if _cold_mask.any() and _warm_mask.any():
                    _a = alpha.squeeze(-1).detach()
                    _ac = _a[_cold_mask].mean().item()
                    _aw = _a[_warm_mask].mean().item()
                    _status = 'OK cold<warm' if _ac < _aw else 'WARN: cold>=warm'
                    print(
                        f"[GATE_BATCH] step={self._gate_step}"
                        f"  alpha cold={_ac:.3f}  warm={_aw:.3f}"
                        f"  [{_status}]"
                        f"  n_cold={_cold_mask.sum().item()} n_warm={_warm_mask.sum().item()}",
                        flush=True
                    )
        if not self.training:
            _modal_contrib  = ((1 - alpha) * _item_modal_out).norm(dim=-1).detach().cpu()
            _collab_contrib = (alpha * item_outputs).norm(dim=-1).detach().cpu()
            _ratio_modal    = _modal_contrib / (_modal_contrib + _collab_contrib + 1e-8)
            if not hasattr(self, '_alpha_buffer'): self._alpha_buffer = []
            self._alpha_buffer.extend(_ratio_modal.tolist())
            # cache divergence signal per item for _get_item_emb_global
            if not hasattr(self, '_h_collab_cache'): self._h_collab_cache = {}
            for _g, _n in zip(_seed_gids.cpu().tolist(), _div_signal.squeeze(-1).detach().cpu()):
                self._h_collab_cache[_g] = _n

        if self.training:
            # cache pré-fusion pour la GRCA loss (avec gradient)
            self._grca_item_collab = item_outputs
            self._grca_h_modal     = _item_modal_out
            self._grca_alpha       = alpha

        _adapted = self.modal_adapter(torch.cat([item_outputs, _item_modal_out], dim=-1))
        item_outputs = item_outputs + (1 - alpha) * _adapted
        item_outputs = self.agg_act(item_outputs)
        item_outputs = self.dropout(item_outputs)
        item_outputs = self.ifc(item_outputs)

        return user_outputs, item_outputs


class ContrastLoss(nn.Module, ABC):

    def __init__(self, h_size, feat_size):
        super(ContrastLoss, self).__init__()
        self.w = nn.Parameter(torch.Tensor(feat_size, h_size))
        torch.nn.init.xavier_uniform_(self.w.data)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, y, y_neg=None):
        if y_neg is None:
            idx = torch.randperm(y.shape[0])
            y_neg = y[idx, :]
        neg_scores = (y_neg @ self.w * x).sum(1)
        neg_labels = neg_scores.new_zeros(neg_scores.shape)
        neg_loss = self.bce_loss(neg_scores, neg_labels)
        scores = (y @ self.w * x).sum(1)
        labels = scores.new_ones(scores.shape)
        pos_loss = self.bce_loss(scores, labels)
        return pos_loss + neg_loss

    def measure_sim(self, x, y):
        if len(y.shape) < 3:
            scores = (y @ self.w * x).sum(1).sigmoid()
        else:
            scores = torch.einsum('bld,bd->bl', y @ self.w, x).sigmoid()
        return scores


class TopicGraphEncoder(nn.Module):

    def __init__(self, sentence_embedding, topic_size, feature_size):
        super().__init__()
        self.sentence_embedding = sentence_embedding
        self.sentence_w = nn.Sequential(
            nn.Linear(128, feature_size, bias=False),
            nn.GELU(),
            nn.Linear(feature_size, feature_size, bias=False),
            nn.GELU(),
            nn.Linear(feature_size, feature_size, bias=False),
        )
        self.gelu = nn.GELU()
        self.sentence_w1 = nn.Parameter(torch.Tensor(topic_size, feature_size))
        self.sentence_score_w = nn.Parameter(torch.Tensor(topic_size, feature_size))
        self.sentence_linear = nn.Linear(feature_size, feature_size)
        self.topic_user_linear = nn.Linear(feature_size, feature_size)
        self.topic_item_linear = nn.Linear(feature_size, feature_size)
        self.topic_user_w = nn.Parameter(torch.Tensor(topic_size, feature_size))
        self.topic_item_w = nn.Parameter(torch.Tensor(topic_size, feature_size))
        self.dropout = nn.Dropout(0.5)

    def sentence_to_topic(self, graph, sentence_id):
        sent_feat = self.sentence_embedding(sentence_id)
        stid = graph.srcdata['global_topic_id']
        graph.srcdata['h'] = self.sentence_w1[stid] * sent_feat
        with graph.local_scope():
            graph.update_all(lambda edges: {'m': edges.src['h']},
                             fn.sum(msg='m', out='sum_h'))
            calc_attn = lambda edges: {'attn_score': (edges.src['h'] * edges.dst['sum_h']).sum(1, keepdim=True)}
            graph.apply_edges(calc_attn)
            graph.edata['attn_score'] = edge_softmax(graph, graph.edata['attn_score'])
            graph.update_all(lambda edges: {'m': edges.src['h'] * self.dropout(edges.data['attn_score'])},
                             fn.sum(msg='m', out='h'))
            result = graph.dstdata['h']
        return self.sentence_linear(result)

    def topic_to_user_item(self, graphs, topic_feat):
        graph = graphs[('topic', 'topic_to_user', 'user')]
        stid = graph.srcdata['global_topic_id']
        graph.srcdata['h'] = self.gelu(topic_feat * self.topic_user_w[stid])
        with graph.local_scope():
            graph.update_all(lambda edges: {'m': edges.src['h']},
                             fn.sum(msg='m', out='sum_h'))
            calc_attn = lambda edges: {'attn_score': (edges.src['h'] * edges.dst['sum_h']).sum(1, keepdim=True)}
            graph.apply_edges(calc_attn)
            graph.edata['attn_score'] = edge_softmax(graph, graph.edata['attn_score'])
            graph.update_all(lambda edges: {'m': edges.src['h'] * self.dropout(edges.data['attn_score'])},
                             fn.sum(msg='m', out='h'))
            user_feat = graph.dstdata['h']
        user_feat = self.topic_user_linear(user_feat)

        graph = graphs[('topic', 'topic_to_item', 'item')]
        stid = graph.srcdata['global_topic_id']
        graph.srcdata['h'] = self.gelu(topic_feat * self.topic_item_w[stid])
        with graph.local_scope():
            graph.update_all(lambda edges: {'m': edges.src['h']},
                             fn.sum(msg='m', out='sum_h'))
            calc_attn = lambda edges: {'attn_score': (edges.src['h'] * edges.dst['sum_h']).sum(1, keepdim=True)}
            graph.apply_edges(calc_attn)
            graph.edata['attn_score'] = edge_softmax(graph, graph.edata['attn_score'])
            graph.update_all(lambda edges: {'m': edges.src['h'] * self.dropout(edges.data['attn_score'])},
                             fn.sum(msg='m', out='h'))
            item_feat = graph.dstdata['h']
        return user_feat, self.topic_item_linear(item_feat)

    def forward(self, input_nodes, encoder_blocks):
        topic_embedding = self.sentence_to_topic(
            encoder_blocks[0][('sentence', 'sentence_to_topic', 'topic')],
            input_nodes['sentence']
        )
        return self.topic_to_user_item(encoder_blocks[1], topic_embedding)


class SentenceRetrival(nn.Module):

    def __init__(self, in_units, num_classes, review_embedding, sentence_embedding, dropout_rate=0.0):
        super(SentenceRetrival, self).__init__()
        self.sentence_embedding = sentence_embedding
        self.review_embedding = review_embedding
        print(f"Sentence Embedding - num_embeddings: {self.sentence_embedding.num_embeddings}, embedding_dim: {self.sentence_embedding.embedding_dim}")
        self.rating_linear = nn.Sequential(
            nn.Linear(in_units * 2, in_units, bias=False),
            nn.ReLU(),
            nn.Linear(in_units, in_units, bias=False),
        )
        self.topic_linear = nn.Sequential(
            nn.Linear(in_units * 2, in_units, bias=False),
            nn.ReLU(),
            nn.Linear(in_units, in_units, bias=False),
        )
        self.item_scorer = nn.Linear(in_units, 1, bias=False)

    def get_review_feature(self, sid):
        length = (sid > 0).float().sum(dim=-1, keepdim=True) + 1e-9
        review_feat = self.sentence_embedding(sid).sum(dim=-2)
        return review_feat / length

    def calc_sentence_ranking(self, edges):
        rh = self.rating_linear(torch.cat([edges.src['rf'], edges.dst['rf']], dim=1))
        th = self.topic_linear(torch.cat([edges.src['tf'], edges.dst['tf']], dim=1))
        th = th + rh
        pos_sid = edges.data['sentence_id']
        pos_review = self.get_review_feature(pos_sid)
        pos_score = (th * pos_review).sum(1)
        n_neg = getattr(self, '_n_neg', 1)
        losses = []
        for _ in range(n_neg):
            neg_sid = torch.randint(1, self.sentence_embedding.weight.shape[0],
                                    pos_sid.shape, device=pos_sid.device)
            neg_review = self.get_review_feature(neg_sid)
            neg_score = (th * neg_review).sum(1)
            losses.append(-(pos_score - neg_score).sigmoid().log())
        return {'mi_score': torch.stack(losses, dim=0).mean(0),
                'ranking_loss': torch.stack(losses, dim=0).mean(0)}

    def predict_score(self, graph, urf, irf):
        graph.nodes['item'].data['rf'] = irf
        graph.nodes['user'].data['rf'] = urf
        def _score_func(e):
            h = self.rating_linear(torch.cat([e.src['rf'], e.dst['rf']], dim=1))
            return {'s': self.item_scorer(h)}
        with graph.local_scope():
            graph.apply_edges(_score_func)
            return graph.edata['s'].squeeze(-1)

    def forward(self, graph, urf, irf, utf, itf, neg_strategy='random', n_neg=1):
        graph.nodes['user'].data['rf'] = urf
        graph.nodes['item'].data['rf'] = irf
        graph.nodes['user'].data['tf'] = utf
        graph.nodes['item'].data['tf'] = itf
        with graph.local_scope():
            if neg_strategy == 'inbatch':
                def _store(edges):
                    rh = self.rating_linear(torch.cat([edges.src['rf'], edges.dst['rf']], dim=1))
                    th_val = self.topic_linear(torch.cat([edges.src['tf'], edges.dst['tf']], dim=1))
                    return {'_th': th_val + rh, '_pids': edges.data['sentence_id']}
                graph.apply_edges(_store)
                th = graph.edata['_th']
                pos_sid = graph.edata['_pids']
                pos_review = self.get_review_feature(pos_sid)
                pos_score = (th * pos_review).sum(1)
                N = th.shape[0]
                score_mat = th @ pos_review.T
                diag = torch.eye(N, dtype=torch.bool, device=th.device)
                score_mat = score_mat.masked_fill(diag, float('-inf'))
                pos_exp = pos_score.unsqueeze(1).expand(N, N)
                bpr = -(pos_exp - score_mat).sigmoid().log()
                bpr = bpr.masked_fill(diag, 0.0)
                loss = bpr.sum(1) / (N - 1)
                mi = loss.mean()
                return mi, mi
            else:
                self._n_neg = n_neg if neg_strategy == 'multi_random' else 1
                graph.apply_edges(self.calc_sentence_ranking)
                mi_score = graph.edata['mi_score']
                ranking_loss = graph.edata['ranking_loss']
        return mi_score.mean(), ranking_loss.mean()

    def measure_sim(self, interaction_feat, sid_list):
        min_sid = torch.min(sid_list).item()
        max_sid = torch.max(sid_list).item()
        num_embeddings = self.sentence_embedding.num_embeddings
        assert min_sid >= 0, f"sid_list negative: min_sid={min_sid}"
        assert max_sid < num_embeddings, f"sid_list >= num_embeddings: max_sid={max_sid}"
        sent_feat = self.sentence_embedding(sid_list)
        return torch.einsum('bd,bkd->bk', interaction_feat, sent_feat)

    @staticmethod
    def _rank_batch(_h, _cand, _trues, _measure_func, topk):
        _cand_mask = (_cand > 0).float()
        _ml = _cand_mask.int().sum(dim=1).max()
        _cand = _cand[:, :_ml]
        _cand_mask = _cand_mask[:, :_ml]
        _scores = _measure_func(_h, _cand)
        _, _topk_idx = torch.topk(_scores, k=topk, dim=-1)
        _topk_items = torch.gather(_cand, 1, _topk_idx)
        return calc_ranking_metrics(_topk_items.cpu().numpy(), _trues.cpu().numpy())

    @torch.no_grad()
    def get_ranking_scores(self, graph, user_feat, item_feat, topk=5):
        graph.nodes['item'].data['th'] = item_feat
        graph.nodes['user'].data['th'] = user_feat
        def _get(edges):
            h = self.topic_linear(torch.cat([edges.src['th'], edges.dst['th']], dim=1))
            return {'th': h, 'cand_sid': edges.dst['candidate_sentence_id']}
        graph.apply_edges(_get)
        h = graph.edata['th']
        true_sents = graph.edata['sentence_id']
        cand_sents = graph.edata['cand_sid']
        rank_list = []
        _bs = 2000
        for i in range(0, h.shape[0], _bs):
            rank_list.append(self._rank_batch(h[i:i+_bs], cand_sents[i:i+_bs],
                                              true_sents[i:i+_bs], self.measure_sim, topk=topk))
        result = {k: sum([list(_rl[k]) for _rl in rank_list], []) for k in rank_list[0].keys()}
        return result


def calc_ranking_metrics(topk_items, true_list):
    precision, recall = precision_recall_score(topk_items, true_list)
    f1 = [2 * p * r / (p + r) if p + r > 0. else 0. for p, r in zip(precision, recall)]
    ndcg = ndcg_score(topk_items, true_list)
    return {'Pre': precision, 'Rec': recall, 'F1': f1, 'nDCG': ndcg}


def precision_recall_score(predicts, trues):
    def pr_each(ps, ts):
        ps = ps[ps > 0]
        ts = ts[ts > 0]
        if len(ts) < 1 or len(ps) < 1:
            return 0., 0.
        inter = np.intersect1d(ps, ts)
        return len(inter) / len(ps), len(inter) / len(ts)
    prs, rcs = zip(*[pr_each(predicts[i], trues[i]) for i in range(len(predicts))])
    return prs, rcs


def ndcg_score(predicts, trues):
    def _ndcg(ps, ts):
        ps = ps[ps > 0]
        ts = ts[ts > 0]
        if len(ts) < 1 or len(ps) < 1:
            return 0.
        isin = np.isin(ps, ts)
        if isin.sum() == 0.:
            return 0.
        dcg = isin / np.log2(np.arange(2, len(isin) + 2))
        idcg = np.sort(isin)[::-1] / np.log2(np.arange(2, len(isin) + 2))
        return np.sum(dcg) / np.sum(idcg)
    return [_ndcg(predicts[i], trues[i]) for i in range(len(predicts))]


class Net(nn.Module):

    def __init__(self, review_embedding, sentence_embedding, params):
        super(Net, self).__init__()
        self.sentence_embedding = sentence_embedding
        self.review_embedding = nn.Embedding.from_pretrained(review_embedding)
        self.lambda_l2 = params.lambda_l2
        self.neg_strategy = params.neg_strategy
        self.n_neg = params.n_neg
        self.rating_encoder = MultiLayerHeteroGraphConv(
            params.rating_values,
            self.review_embedding,
            params.user_size,
            params.item_size,
            params.gcn_out_units,
            params.num_layers,
            dropout_rate=params.gcn_dropout
        )
        self.topic_encoder = TopicGraphEncoder(
            self.sentence_embedding, params.global_topic_size, params.gcn_out_units
        )
        self.topic_decoder = SentenceRetrival(
            params.gcn_out_units, 5, self.review_embedding, self.sentence_embedding
        )
        reset_parameters(self)

    def state_dict(self):
        sd = super().state_dict()
        pop_keys = [k for k in sd if 'review_embedding' in k or 'sentence_embedding' in k]
        for k in pop_keys:
            sd.pop(k)
        return sd

    def predict_score(self, input_nodes, encoder_blocks, decoder_graph):
        user_feat, item_feat = self.rating_encoder(input_nodes, encoder_blocks)
        return self.topic_decoder.predict_score(decoder_graph, user_feat, item_feat)

    def calc_loss(self,
                  rating_input_nodes,
                  rating_encoder_blocks,
                  topic_input_nodes,
                  topic_encoder_blocks,
                  pos_graph,
                  sample_weight=None):
        self.train()
        urf, irf = self.rating_encoder(rating_input_nodes, rating_encoder_blocks)
        utf, itf = self.topic_encoder(topic_input_nodes, topic_encoder_blocks)

        ed_mi, ranking_loss = self.topic_decoder(
            pos_graph, urf, irf, utf + urf, itf + irf,
            neg_strategy=self.neg_strategy, n_neg=self.n_neg
        )

        src_pos, dst_pos = pos_graph.edges()
        u_emb     = urf[src_pos]
        i_pos_emb = irf[dst_pos]
        N = i_pos_emb.shape[0]
        perm = torch.randperm(N, device=i_pos_emb.device)
        clash = perm == torch.arange(N, device=i_pos_emb.device)
        if clash.any():
            perm[clash] = (perm[clash] + 1) % N
        i_neg_emb = i_pos_emb[perm]

        h_pos = self.topic_decoder.rating_linear(torch.cat([u_emb, i_pos_emb], dim=1))
        h_neg = self.topic_decoder.rating_linear(torch.cat([u_emb, i_neg_emb], dim=1))
        score_pos = self.topic_decoder.item_scorer(h_pos).squeeze(-1)
        score_neg = self.topic_decoder.item_scorer(h_neg).squeeze(-1)

        ratings = pos_graph.edata['rating'].float()
        weight  = ratings / 5.0
        if sample_weight is not None:
            weight = weight * sample_weight
            weight = weight / (weight.mean() + 1e-9)
        bpr_loss = -(weight * F.logsigmoid(score_pos - score_neg)).mean()
        l2_reg = self.lambda_l2 * (
            self.rating_encoder.user_embedding.norm(2).pow(2) +
            self.rating_encoder.item_embedding.norm(2).pow(2)
        ) / u_emb.shape[0]
        bpr_loss = bpr_loss + l2_reg

        self._bpr_step = getattr(self, '_bpr_step', 0) + 1
        if self._bpr_step <= 10 or self._bpr_step % 100 == 0:
            with torch.no_grad():
                diff = score_pos.detach() - score_neg.detach()
                _neg_global = pos_graph.nodes['item'].data['_ID'][dst_pos[perm]]
                _neg_deg    = self.rating_encoder.item_degree_tensor[_neg_global].float()
                _rat_dist   = ratings.long().bincount(minlength=6)[1:].tolist()
                print(
                    f"[BPR_DEBUG] step={self._bpr_step}"
                    f" | diff mean={diff.mean():.4f} std={diff.std():.4f}"
                    f" | pos={score_pos.detach().mean():.4f} neg={score_neg.detach().mean():.4f}"
                    f" | weight mean={weight.mean():.3f} min={weight.min():.3f} max={weight.max():.3f}"
                    f" | neg_deg mean={_neg_deg.mean():.1f} min={_neg_deg.min():.0f} max={_neg_deg.max():.0f}"
                    f" | rating_dist(1..5)={_rat_dist}",
                    flush=True
                )

        return bpr_loss, ed_mi, ranking_loss, urf, irf

    def _get_item_emb_global(self, global_iids):
        # gate input = divergence signal (cached) — same as training
        device    = self.rating_encoder.item_embedding.device
        cache     = self.rating_encoder._h_collab_cache
        div_sig   = torch.tensor([cache[i.item()] for i in global_iids], device=device).unsqueeze(-1)
        collab    = self.rating_encoder.item_embedding[global_iids]
        modal     = self.rating_encoder.h_modal[global_iids]
        alpha     = torch.sigmoid(F.softplus(self.rating_encoder.gate_temp) * self.rating_encoder.gate_mlp(div_sig))
        _ad       = self.rating_encoder.modal_adapter(torch.cat([collab, modal], dim=-1))
        emb       = collab + (1 - alpha) * _ad
        return self.rating_encoder.ifc(self.rating_encoder.agg_act(emb))

    @torch.no_grad()
    def evaluate_sentence_ranking(self, dataloader, raw_graph, sampler, etype='valid', topk=5):
        device = self.review_embedding.weight.device
        group_scores = defaultdict(lambda: defaultdict(list))
        scores_list = []
        for rating_input_nodes, pos_graph, _neg_graph, rating_encoder_blocks in dataloader:
            decoder_graph = pos_graph[etype].to(device)
            input_nodes, _, blocks = sampler.sample(
                raw_graph,
                {'user': decoder_graph.nodes['user'].data['_ID'].cpu(),
                 'item': decoder_graph.nodes['item'].data['_ID'].cpu()}
            )
            rating_input_nodes = {k: v.to(device) for k, v in rating_input_nodes.items()}
            input_nodes        = {k: v.to(device) for k, v in input_nodes.items()}
            blocks             = [b.to(device) for b in blocks]
            rating_encoder_blocks = [b.to(device) for b in rating_encoder_blocks]
            ratings = decoder_graph.edata['rating'].cpu().tolist()
            urf, irf = self.rating_encoder(rating_input_nodes, rating_encoder_blocks)
            utf, itf = self.topic_encoder(input_nodes, blocks)
            ranking_scores = self.topic_decoder.get_ranking_scores(
                decoder_graph, utf + urf, itf + irf, topk
            )
            scores_list.append(ranking_scores)
            for idx, rating in enumerate(ratings):
                group = int(rating)
                for metric, values in ranking_scores.items():
                    group_scores[group][metric].append(values[idx])

        group_metrics = {
            group: {metric: np.mean(vals) for metric, vals in metrics.items()}
            for group, metrics in group_scores.items()
        }
        print("Rating group metrics (1-5):")
        for group in sorted(group_metrics.keys()):
            m = group_metrics[group]
            print("Group {}: Pre={:.4f} Rec={:.4f} F1={:.4f} nDCG={:.4f}".format(
                group, m.get('Pre', 0), m.get('Rec', 0), m.get('F1', 0), m.get('nDCG', 0)))

        scores_list = {k: sum([list(_rl[k]) for _rl in scores_list], []) for k in scores_list[0].keys()}
        return {k: np.mean(v) for k, v in scores_list.items()}

    @torch.no_grad()
    def evaluate_ranking_ndcg(self, dataloader, dataset, K=10,
                               relevance_threshold=1, etype='valid',
                               n_neg=99, seed=42):
        import math
        device = self.review_embedding.weight.device
        self.eval()
        rng = np.random.default_rng(seed)

        graph = dataset.graph
        train_u, train_i = graph['train'].edges()
        train_seen = defaultdict(set)
        for u, i in zip(train_u.tolist(), train_i.tolist()):
            train_seen[u].add(i)

        user_emb  = {}
        item_emb  = {}
        pos_items = defaultdict(dict)

        for input_nodes, pos_graph, _neg_graph, blocks in dataloader:
            input_nodes_dev = {k: v.to(device) for k, v in input_nodes.items()}
            pg              = pos_graph[etype].to(device)
            blocks_dev      = [b.to(device) for b in blocks]
            urf, irf = self.rating_encoder(input_nodes_dev, blocks_dev)
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

        rating_linear = self.topic_decoder.rating_linear.to('cpu')
        item_scorer   = self.topic_decoder.item_scorer.to('cpu')

        def score_pairs(u_emb_t, i_embs_t):
            u_rep = u_emb_t.unsqueeze(0).expand(i_embs_t.shape[0], -1)
            return item_scorer(rating_linear(torch.cat([u_rep, i_embs_t], dim=1))).squeeze(-1)

        known_items = np.array(sorted(item_emb.keys()))
        ndcg_list = []
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
            ranked_ids = [iid for iid, _ in sorted(zip(all_ids, all_scores), key=lambda x: x[1], reverse=True)]
            ideal_n    = len(relevant)
            dcg        = sum(1.0 / math.log2(i + 2) for i, iid in enumerate(ranked_ids[:K]) if iid in relevant)
            idcg       = sum(1.0 / math.log2(i + 2) for i in range(min(ideal_n, K)))
            ndcg_list.append(dcg / idcg if idcg > 0 else 0.0)

        self.topic_decoder.rating_linear.to(device)
        self.topic_decoder.item_scorer.to(device)
        return float(np.mean(ndcg_list)) if ndcg_list else 0.0

    def compute_fusion_loss(self, user_emb, h_modal_pos, h_modal_neg, lambda_f=1e-4):
        pos_score = (user_emb * h_modal_pos).sum(dim=-1)
        neg_score = (user_emb * h_modal_neg).sum(dim=-1)
        return lambda_f * (-F.logsigmoid(pos_score - neg_score).mean())


import random

def train(params):
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"[SEED] {params.seed}", flush=True)

    global logger
    logger = get_logger(params.model_short_name, None)
    logger.info(f"Parameters:\n{args_to_str(params)}")

    dataset = GraphData(params.dataset_name, params.dataset_path)

    train_sentence_ids = dataset.train_sentence_ids
    valid_sentence_ids = dataset.valid_sentence_ids
    test_sentence_ids  = dataset.test_sentence_ids

    params.user_size         = dataset.user_size
    params.item_size         = dataset.item_size
    params.rating_values     = dataset.possible_rating_values
    params.global_topic_size = dataset.graph.nodes['topic'].data['global_topic_id'].max() + 1

    all_sentence_ids = torch.cat([train_sentence_ids, valid_sentence_ids, test_sentence_ids])
    print(f"Sentence IDs — min: {torch.min(all_sentence_ids).item()}, max: {torch.max(all_sentence_ids).item()}")

    net = Net(dataset.review_embedding, dataset.sentence_embedding, params)
    net = net.to(params.device)

    v_feat, t_feat = load_modal_features('/home/infres/belguith/PFE/bm3_data/musical')
    modal_enc = ModalEncoder(v_feat, t_feat, embed_dim=params.gcn_out_units).to(params.device)

    optimizer = torch.optim.Adam(
        list(net.parameters()) + list(modal_enc.parameters()),
        lr=params.train_lr
    )
    logger.info("Loading network finished ...\n")

    train_dataloader, valid_dataloader, test_dataloader = dataset.get_dataloaders(
        batch_size=params.batch_size, num_layers=params.num_layers
    )
    graph = dataset.graph
    topic_sampler = dataset.get_topic_sentence_sampler()

    best_valid_ndcg = 0.0
    best_test_ndcg  = 0.0
    no_better_valid = 0
    best_iter       = -1
    learning_rate   = params.train_lr
    repr_norm_history = []

    h_modal, h_v, h_t = modal_enc()
    net.rating_encoder.h_modal = h_modal

    with torch.no_grad():
        _n_items = net.rating_encoder.item_embedding.shape[0]
        net.rating_encoder.item_embedding.data.copy_(h_modal[:_n_items].detach())
    print(f"[INIT] item_embedding ← h_modal (semantic init, norm={h_modal[:_n_items].norm(dim=-1).mean():.3f})", flush=True)

    import pandas as pd
    _df_deg_train = pd.read_csv('/home/infres/belguith/PFE/processed/Musical_interactions_reviews.csv')
    _deg_train = _df_deg_train['iid'].value_counts()
    _n_init = net.rating_encoder.item_embedding.shape[0]
    _deg_tensor_train = torch.zeros(_n_init, dtype=torch.float32)
    for _iid, _cnt in _deg_train.items():
        if int(_iid) < _n_init:
            _deg_tensor_train[int(_iid)] = float(_cnt)
    net.rating_encoder.item_degree_tensor = _deg_tensor_train.to(params.device)
    print("[INIT] item_degree_tensor injected", flush=True)

    _diag_deg = _deg_tensor_train
    _diag_cold_idx = torch.where((_diag_deg >= 5) & (_diag_deg <= 10))[0]
    _diag_warm_idx = torch.where(_diag_deg > 20)[0]
    print(f"[DIAG_SETUP] cold_items={len(_diag_cold_idx)}, warm_items={len(_diag_warm_idx)}", flush=True)

    logger.info('Test - ' + format_dict_to_str(
        net.evaluate_sentence_ranking(test_dataloader, graph, topic_sampler, etype='test')
    ))
    logger.info("Start training ...")

    with torch.no_grad():
        _u0 = net.rating_encoder.user_embedding.norm(dim=-1).mean().item()
        _i0 = net.rating_encoder.item_embedding.norm(dim=-1).mean().item()
        repr_norm_history.append((0, _u0, _i0, _u0 / (_i0 + 1e-9)))
        print(f"[REPR_NORM] epoch=  0  user={_u0:.4f}  item={_i0:.4f}  ratio={_u0/(_i0+1e-9):.4f}  (INIT)", flush=True)
        print(f"[GATE_INIT] divergence gate — gate_T=softplus(2.0)={F.softplus(net.rating_encoder.gate_temp).item():.4f}", flush=True)

    for iter_idx in range(1, params.epoch):
        net.rating_encoder.reset_epoch_diag()
        net.train()
        pbar = tqdm(train_dataloader)
        for rating_input_nodes, pos_graph, _neg_graph, rating_blocks in pbar:
            topic_input_nodes, _, topic_blocks = topic_sampler.sample(
                graph,
                {'user': pos_graph.nodes['user'].data['_ID'],
                 'item': pos_graph.nodes['item'].data['_ID']}
            )
            rating_input_nodes = {k: v.to(params.device) for k, v in rating_input_nodes.items()}
            topic_input_nodes  = {k: v.to(params.device) for k, v in topic_input_nodes.items()}
            pos_graph_train    = pos_graph['train'].to(params.device)
            rating_blocks      = [b.to(params.device) for b in rating_blocks]
            topic_blocks       = [b.to(params.device) for b in topic_blocks]

            if params.no_modal:
                h_modal = torch.zeros(net.rating_encoder.item_embedding.shape[0],
                                      params.gcn_out_units, device=params.device)
                h_v = h_t = h_modal
            else:
                h_modal, h_v, h_t = modal_enc()
            net.rating_encoder.h_modal = h_modal

            r_loss, mi_score, ranking_loss, urf, irf = net.calc_loss(
                rating_input_nodes, rating_blocks,
                topic_input_nodes, topic_blocks,
                pos_graph_train,
            )

            batch_items = pos_graph_train.nodes['item'].data['_ID']
            modal_loss  = (torch.tensor(0.0, device=params.device) if params.no_modal
                           else modal_enc.calculate_loss_infonce(h_v, h_t, batch_items))

            _src_idx_f, _dst_idx_f = pos_graph_train.edges()
            _item_gids_f      = pos_graph_train.dstdata['_ID'][_dst_idx_f]
            _u_emb_per_edge   = urf[_src_idx_f]
            _h_modal_per_edge = h_modal[_item_gids_f]
            f_loss = torch.tensor(0.0, device=params.device)
            _n_f = _h_modal_per_edge.shape[0]
            if not params.no_modal and _n_f > 1:
                _perm_f = torch.randperm(_n_f, device=params.device)
                _clash  = _perm_f == torch.arange(_n_f, device=params.device)
                if _clash.any():
                    _perm_f[_clash] = (_perm_f[_clash] + 1) % _n_f
                f_loss = net.compute_fusion_loss(
                    _u_emb_per_edge, _h_modal_per_edge, _h_modal_per_edge[_perm_f],
                    lambda_f=params.lambda_f
                )

            _eff_lambda_mi = params.lambda_mi if iter_idx > params.lambda_mi_warmup else 0.0
            mi_term = (_eff_lambda_mi * mi_score if _eff_lambda_mi > 0
                       else torch.tensor(0.0, device=params.device))

            loss = r_loss + 0.1 * modal_loss + f_loss + mi_term

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), params.train_grad_clip)
            optimizer.step()

            pbar.set_description(
                f"train_loss={r_loss:.4f}, MI={mi_score:.2f}, "
                f"mi_term={mi_term.item():.4f}, modal={modal_loss.item():.4f}, "
                f"f_loss={f_loss.item():.6f}"
            )

        with torch.no_grad():
            _n = net.rating_encoder.item_embedding.shape[0]
            _h = net.rating_encoder.h_modal
            _e = net.rating_encoder.item_embedding
            _collab_norm = _e.norm(dim=-1).mean()
            _modal_norm  = _h.norm(dim=-1).mean()
            _user_emb_norm = net.rating_encoder.user_embedding.norm(dim=-1).mean().item()
            print(f"[EMB_NORM] epoch={iter_idx:>3d}  user={_user_emb_norm:.4f}  item={_collab_norm.item():.4f}", flush=True)
            print(f"[DIAG] h_modal std_per_dim={_h.std(dim=0).mean():.4f} norm={_modal_norm:.3f} | collab norm={_collab_norm:.3f}", flush=True)
            _sim = F.cosine_similarity(_h[:_n], _e).mean()
            print(f"[DIAG] cosine sim h_modal/collab = {_sim:.4f}", flush=True)

            # diagnostics gate — alpha post-GCN accumulé pendant l'epoch (valeurs réelles)
            _acc = net.rating_encoder._epoch_alpha_acc
            if _acc:
                _gids_t = torch.tensor(sorted(_acc.keys()), dtype=torch.long)
                _alph_t = torch.tensor([_acc[g.item()] for g in _gids_t])
                _deg_t  = net.rating_encoder.item_degree_tensor[_gids_t.to(params.device)].cpu().float()
                print(f"[GATE] epoch={iter_idx} mean={_alph_t.mean():.3f} std={_alph_t.std():.3f} "
                      f"min={_alph_t.min():.3f} max={_alph_t.max():.3f}  n_items={len(_acc)}", flush=True)
                _corr = torch.corrcoef(torch.stack([_alph_t, _deg_t]))[0, 1].item()
                print(f"[GATE_CORR] alpha vs degree={_corr:.4f}  T={F.softplus(net.rating_encoder.gate_temp).item():.4f}", flush=True)
                _cold_m = (_deg_t >= 5) & (_deg_t <= 10)
                _warm_m = _deg_t > 20
                if _cold_m.any() and _warm_m.any():
                    _ac = _alph_t[_cold_m];  _aw = _alph_t[_warm_m]
                    _status = 'OK cold<warm' if _ac.mean() < _aw.mean() else 'WARN: cold>=warm'
                    print(f"[GATE_COLDWARM] epoch={iter_idx}  alpha cold={_ac.mean():.3f}±{_ac.std():.3f}"
                          f"  warm={_aw.mean():.3f}±{_aw.std():.3f}  [{_status}]", flush=True)
                _n_c = net.rating_encoder._epoch_contrib_n
                if _n_c > 0:
                    _mc = net.rating_encoder._epoch_mc_sum / _n_c
                    _cc = net.rating_encoder._epoch_cc_sum / _n_c
                    print(f"[CONTRIB] collab={_cc:.3f} modal={_mc:.3f} "
                          f"ratio_modal={_mc/(_mc+_cc)*100:.1f}%", flush=True)

            with torch.no_grad():
                _hv = (modal_enc.norm_v(F.relu(modal_enc.image_trs(modal_enc.image_embedding.weight)))
                       if hasattr(modal_enc, 'norm_v')
                       else modal_enc.image_trs(modal_enc.image_embedding.weight))
                _ht = (modal_enc.norm_t(F.relu(modal_enc.text_trs(modal_enc.text_embedding.weight)))
                       if hasattr(modal_enc, 'norm_t')
                       else modal_enc.text_trs(modal_enc.text_embedding.weight))
                _ww = torch.softmax(torch.cat([_hv.norm(dim=-1, keepdim=True), _ht.norm(dim=-1, keepdim=True)], dim=-1), dim=-1)
                _w_img = _ww[:, 0].mean().item()
                _w_txt = _ww[:, 1].mean().item()
            print(f"[EPOCH_MODAL] epoch={iter_idx} w_img={_w_img:.3f} w_txt={_w_txt:.3f} "
                  f"{'COLLAPSE' if _w_img > 0.95 or _w_txt > 0.95 else 'OK'}", flush=True)

            _urf_norm = urf.detach().norm(dim=-1).mean().item()
            _irf_norm = irf.detach().norm(dim=-1).mean().item()
            _ratio    = _urf_norm / (_irf_norm + 1e-9)
            repr_norm_history.append((iter_idx, _urf_norm, _irf_norm, _ratio))
            print(f"[REPR_NORM] epoch={iter_idx:>3d}  user={_urf_norm:.4f}  item={_irf_norm:.4f}  ratio={_ratio:.4f}", flush=True)

            if params.lambda_mi > 0:
                _topic_w_norm = sum(p.norm().item() for p in net.topic_encoder.parameters())
                _topic_g_norm = sum(p.grad.norm().item() for p in net.topic_encoder.parameters() if p.grad is not None)
                print(f"[TOPIC_ENCODER] epoch={iter_idx:>3d}  param_norm={_topic_w_norm:.4f}"
                      f"  grad_norm={_topic_g_norm:.4f}  ed_mi={mi_score:.4f}"
                      f"  eff_lambda={_eff_lambda_mi}  (warmup={params.lambda_mi_warmup})", flush=True)

        valid_ndcg  = net.evaluate_ranking_ndcg(valid_dataloader, dataset, K=10, etype='valid')
        logging_str = (f"Epoch={iter_idx:>3d}, "
                       f"Train_BPR={r_loss.item():.4f}, MI={mi_score:.2f}, Valid_nDCG@10={valid_ndcg:.4f}, ")

        if valid_ndcg > best_valid_ndcg:
            best_valid_ndcg = valid_ndcg
            no_better_valid = 0
            best_iter       = iter_idx
            test_ndcg       = net.evaluate_ranking_ndcg(test_dataloader, dataset, K=10, etype='test')
            best_test_ndcg  = test_ndcg
            logging_str    += f'Test_nDCG@10={test_ndcg:.4f}'
            checkpoint      = net.state_dict()
            checkpoint['modal_enc'] = modal_enc.state_dict()
            torch.save(checkpoint, params.model_save_path)
        else:
            no_better_valid += 1
            if no_better_valid > params.train_early_stopping_patience and learning_rate <= params.train_min_lr:
                logger.info("Early stopping threshold reached. Stop training.")
                break
            if no_better_valid > params.train_decay_patience:
                new_lr = max(learning_rate * params.train_lr_decay_factor, params.train_min_lr)
                if new_lr < learning_rate:
                    learning_rate = new_lr
                    logger.info(f"\tChange the LR to {new_lr:g}")
                    for p in optimizer.param_groups:
                        p['lr'] = learning_rate
                    no_better_valid = 0

        logger.info(logging_str)
        logger.info('Test - ' + format_dict_to_str(
            net.evaluate_sentence_ranking(test_dataloader, graph, topic_sampler, etype='test')
        ))

    logger.info(f'Best Iter Idx={best_iter}, Best Valid nDCG@10={best_valid_ndcg:.4f}, Best Test nDCG@10={best_test_ndcg:.4f}')
    logger.info(params.model_save_path)
    logger.info("=== Repr norm history ===")
    logger.info(f"{'Epoch':>6} | {'user_norm':>9} | {'item_norm':>9} | {'ratio_u/i':>9}")
    for ep, un, it, rt in repr_norm_history:
        flag = "  <-- DESEQUILIBRE" if rt > 2.0 or rt < 0.5 else ""
        logger.info(f"{ep:>6d} | {un:>9.4f} | {it:>9.4f} | {rt:>9.4f}{flag}")


def test(params):
    import evaluate_model_run as _em
    _em.test(params, net_class=Net)


if __name__ == '__main__':
    config_args = config()
    if config_args.test_only:
        test(config_args)
    else:
        train(config_args)
        test(config_args)

# -*- coding: utf-8 -*-

import copy
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
from rhg_data_binary import GraphData
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
    parser.add_argument('--mi_temp',           type=float, default=0.1,
                        help='Température InfoNCE pour la MI loss (défaut 0.1)')
    parser.add_argument('--gcn_dropout',       type=float, default=0.8)
    parser.add_argument('--num_layers',        type=int,   default=2)
    parser.add_argument('--run_tag',           type=str,   default='',
                        help='Suffix appended to checkpoint filename (e.g. v3)')
    parser.add_argument('--test_only',         action='store_true', default=False)
    parser.add_argument('--sr_formula',        type=str, default='topic_only',
                        choices=['topic_only', 'fix1', 'vrai_fix', 'orig'],
                        help='Formule SR eval: topic_only=topic_linear(utf,itf), '
                             'fix1=topic_linear(utf,itf)+rating_linear(urf,irf), '
                             'vrai_fix=topic_linear(utf+urf,itf+irf)+rating_linear(urf,irf), '
                             'orig=topic_linear(utf+urf,itf+irf) ancienne formule sans rating_linear')
    parser.add_argument('--no_modal',          action='store_true', default=False)
    parser.add_argument('--no_review',         action='store_true', default=False,
                        help='Désactive les review embeddings dans GCN (attention uniforme) + MI=0')
    parser.add_argument('--lambda_grca',       type=float, default=0.1,
                        help='Poids de la loss GRCA (InfoNCE modal↔collab pondérée par (1-alpha))')
    parser.add_argument('--rand_init',         action='store_true', default=False,
                        help='Init item_embedding aléatoire (user_norm scale) au lieu de direction h_modal — GRCA reste actif')
    parser.add_argument('--bpr_logdeg',        action='store_true', default=False,
                        help='Pondère BPR par (1-alpha) du positive item — cold items reçoivent plus de gradient BPR')

    # ── Params fixes (rarement changés) ─────────────────────────────────────
    parser.add_argument('--device',                      type=int,   default=0)
    parser.add_argument('--epoch',                       type=int,   default=1000)
    parser.add_argument('--train_grad_clip',             type=float, default=1.0)
    parser.add_argument('--train_lr',                    type=float, default=0.001)
    parser.add_argument('--train_min_lr',                type=float, default=0.0001)
    parser.add_argument('--train_lr_decay_factor',       type=float, default=0.5)
    parser.add_argument('--train_decay_patience',        type=int,   default=8)
    parser.add_argument('--train_early_stopping_patience', type=int, default=100)
    parser.add_argument('--sr_freeze_patience',           type=int, default=15,
                        help='Nb epochs sans amélioration SR avant de geler le topic encoder')
    parser.add_argument('--review_feat_size',            type=int,   default=128)
    parser.add_argument('--sent_lr_scale',               type=float, default=0.1,
                        help='LR multiplier for sentence_embedding (0 = frozen)')
    parser.add_argument('--lambda_reg',                  type=float, default=0.0,
                        help='L2 reg toward sent_emb init (0 = disabled)')
    parser.add_argument('--lambda_f',                    type=float, default=0.01)
    parser.add_argument('--floss_mode', type=str, default='sqrt_uscale',
                        choices=['full_uscale', 'sqrt_uscale', 'cosnorm', 'dynorm'],
                        help='f_loss scaling mode (dynorm = F.normalize(h_modal)*item_emb_norm)')
    parser.add_argument('--minmax_alpha',  action='store_true', default=False,
                        help='Use (deg-min)/(max-min) instead of log1p for alpha')
    parser.add_argument('--cosine_lr',     action='store_true', default=False,
                        help='Use CosineAnnealingLR instead of manual step decay')
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
    _nneg_tag = f'_k{args.n_neg}'        if args.n_neg > 1 else ''
    _warm_tag = f'_warm{args.lambda_mi_warmup}' if args.lambda_mi_warmup > 0 else ''
    _nmod_tag  = '_nomodal'               if args.no_modal                 else ''
    _nrev_tag  = '_noreview'              if args.no_review                else ''
    _run_tag   = f'_{args.run_tag}'       if args.run_tag                  else ''
    _grca_tag  = f'_grca{args.lambda_grca}' if args.lambda_grca > 0       else ''
    _rinit_tag = '_randinit'              if args.rand_init                else ''
    _bprw_tag  = '_bprlogdeg'            if args.bpr_logdeg               else ''
    args.model_save_path = (
        f'model_save/{args.dataset_name}/{args.model_short_name}'
        f'_layers_{args.num_layers}_seed{args.seed}'
        f'{_l2_tag}{_bs_tag}{_mi_tag}{_dp_tag}{_neg_tag}{_nneg_tag}{_warm_tag}{_nmod_tag}{_nrev_tag}{_grca_tag}{_rinit_tag}{_bprw_tag}_logdeg_normfuse_topicmi_slowsent01_gpool_binary_grca_emb{_run_tag}.pt'
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
                 dropout_rate=0.0,
                 no_review=False):
        super(GCMCGraphConv, self).__init__()

        self.no_review = no_review
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

            if self.no_review:
                graph.edata['pa'] = torch.ones(graph.num_edges(), 1, device=feat.device)
            else:
                review_feat = self.get_review_feature(graph.edata['review_id'])
                graph.edata['pa'] = torch.sigmoid(self.prob_score(review_feat))

            if self.review_w is not None and not self.no_review:
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

    def __init__(self, review_embedding, user_size, item_size, msg_units, num_layers, aggregate='sum', dropout_rate=0.0, minmax_alpha=False, no_review=False):
        super(MultiLayerHeteroGraphConv, self).__init__()

        assert num_layers > 0, "The number of conv layers must have at least one!"
        self.num_layers = num_layers
        self.minmax_alpha = minmax_alpha
        self.conv_layers = nn.ModuleList()

        self.user_embedding = nn.Parameter(torch.Tensor(user_size, msg_units))
        self.item_embedding = nn.Parameter(torch.Tensor(item_size, msg_units))
        nn.init.xavier_uniform_(self.item_embedding.unsqueeze(0)).squeeze(0)
        self.h_modal = None
        self.item_degree_tensor = None
        self.register_buffer('_pre_ci_global_max', torch.tensor(1.0))

        for l in range(num_layers):
            sub_conv = {
                'interacts': GCMCGraphConv(msg_units, review_embedding,
                                           add_embedding_mapping=l == 0,
                                           add_review=l == (num_layers - 1),
                                           dropout_rate=dropout_rate,
                                           no_review=no_review),
                'rev':       GCMCGraphConv(msg_units, review_embedding,
                                           add_embedding_mapping=l == 0,
                                           add_review=l == (num_layers - 1),
                                           dropout_rate=dropout_rate,
                                           no_review=no_review),
            }
            self.conv_layers.append(nn.ModuleDict(sub_conv))

        self.ufc = nn.Linear(msg_units, msg_units)
        self.ifc = nn.Linear(msg_units, msg_units)
        self.dropout = nn.Dropout(0.5)
        self.agg_act = nn.GELU()

    def reset_epoch_diag(self):
        self._epoch_floss_sum  = 0.0
        self._epoch_grca_sum   = 0.0
        self._epoch_n_batches  = 0
        self._epoch_bpr_batches = []
        self._epoch_sim_sum   = 0.0   # cosine sim h_modal ↔ item_collab post-GCN
        self._epoch_sim_n     = 0

    def forward(self, input_nodes, encoder_blocks):
        curr_user = None
        curr_item = None

        for l in range(len(self.conv_layers)):
            block = encoder_blocks[l]
            conv_layer = self.conv_layers[l]
            if l == 0:
                u_src = self.user_embedding[input_nodes['user']]
                i_src = self.item_embedding[input_nodes['item']]
            else:
                u_src = curr_user
                i_src = curr_item
            curr_item = conv_layer['interacts'](block['user', 'interacts', 'item'], u_src)
            curr_user = conv_layer['rev'](block['item', 'rev', 'user'], i_src)

        user_outputs = curr_user
        item_outputs = curr_item

        _seed_gids       = encoder_blocks[-1].dstdata['_ID']['item']
        _degrees_out     = self.item_degree_tensor[_seed_gids]
        _degree_norm_out = (_degrees_out / self.item_degree_tensor.max().clamp(min=1.0)).unsqueeze(-1)
        _item_modal_out  = self.h_modal[_seed_gids]

        user_outputs = self.agg_act(user_outputs)
        user_outputs = self.dropout(user_outputs)
        user_outputs = self.ufc(user_outputs)

        # pre-ci : uniquement le conv 'interacts' (dst=item)
        _last_conv   = self.conv_layers[-1]
        _pre_ci_sum  = _last_conv['interacts']._last_pre_ci
        _pre_ci_norm = _pre_ci_sum.norm(dim=-1, keepdim=True)
        _pre_ci_norm_scaled = (_pre_ci_norm / self._pre_ci_global_max.clamp(min=1e-8)).clamp(max=1.0)
        # alpha fixé par le degré — cold→0, warm→1, pas de MLP appris
        if self.minmax_alpha:
            _deg_min = self.item_degree_tensor.min().float()
            _deg_max = self.item_degree_tensor.max().float()
            alpha = ((_degrees_out.float() - _deg_min) / (_deg_max - _deg_min + 1e-8)).unsqueeze(-1)
        else:
            alpha = (torch.log1p(_degrees_out.float()) /
                     torch.log1p(self.item_degree_tensor.max().float() + 1e-8)).unsqueeze(-1)
        if self.training:
            self._last_alpha = alpha.squeeze(-1).detach()
        if not self.training:
            _modal_contrib  = ((1 - alpha) * _item_modal_out).norm(dim=-1).detach().cpu()
            _collab_contrib = (alpha * item_outputs).norm(dim=-1).detach().cpu()
            _ratio_modal    = _modal_contrib / (_modal_contrib + _collab_contrib + 1e-8)
            if not hasattr(self, '_alpha_buffer'): self._alpha_buffer = []
            self._alpha_buffer.extend(_ratio_modal.tolist())
            # cache pre-ci norm per item so _get_item_emb_global uses same gate input as training
            if not hasattr(self, '_h_collab_cache'): self._h_collab_cache = {}
            for _g, _n in zip(_seed_gids.cpu().tolist(), _pre_ci_norm_scaled.squeeze(-1).detach().cpu()):
                self._h_collab_cache[_g] = _n

        if self.training:
            # GRCA sur item_embedding pré-GCN (au lieu de h_collab post-GCN)
            _item_emb_seed = self.item_embedding[_seed_gids]  # pré-GCN, gradient direct
            self._grca_item_collab = _item_emb_seed
            self._grca_h_modal     = _item_modal_out
            self._grca_alpha       = alpha
            # cosine sim h_modal ↔ item_embedding pré-GCN
            with torch.no_grad():
                _sim_batch = F.cosine_similarity(
                    _item_modal_out.detach(), _item_emb_seed.detach(), dim=-1
                )
                self._epoch_sim_sum += _sim_batch.sum().item()
                self._epoch_sim_n   += _sim_batch.shape[0]

        # pas de fusion explicite — GCN pur, GRCA fait l'alignement implicitement
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

    def __init__(self, in_units, num_classes, review_embedding, sentence_embedding, dropout_rate=0.0, mi_temp=0.1):
        super(SentenceRetrival, self).__init__()
        self.sentence_embedding = sentence_embedding
        self.review_embedding = review_embedding
        self.mi_temp = mi_temp
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

    def _pick_one_sentence(self, pos_sid):
        # pos_sid: (E, max_sents) — pick one random valid (>0) sentence per edge
        valid_mask = pos_sid > 0
        rand_w = valid_mask.float() * torch.rand(valid_mask.shape, device=pos_sid.device)
        sel_idx = rand_w.argmax(dim=1)
        return pos_sid[torch.arange(pos_sid.shape[0], device=pos_sid.device), sel_idx]  # (E,)

    def calc_sentence_ranking(self, edges):
        th = self.topic_linear(torch.cat([edges.src['tf'], edges.dst['tf']], dim=1))
        pos_sid = edges.data['sentence_id']          # (E, max_sents)
        pos_s   = self._pick_one_sentence(pos_sid)   # (E,) — une phrase individuelle
        pos_feat = self.sentence_embedding(pos_s)    # (E, 128)
        pos_score = (th * pos_feat).sum(1)
        # négatif = phrase d'un autre edge du batch (shuffle)
        perm = torch.randperm(pos_feat.shape[0], device=pos_feat.device)
        neg_score = (th * pos_feat[perm]).sum(1)
        loss = -F.logsigmoid(pos_score - neg_score)
        return {'mi_score': loss, 'ranking_loss': loss}

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
                    th_val = self.topic_linear(torch.cat([edges.src['tf'], edges.dst['tf']], dim=1))
                    return {'_th': th_val, '_pids': edges.data['sentence_id']}
                graph.apply_edges(_store)
                th = graph.edata['_th']
                pos_sid = graph.edata['_pids']          # (N, max_sents)
                N = th.shape[0]
                pos_s    = self._pick_one_sentence(pos_sid)   # (N,) — phrase individuelle
                pos_feat = self.sentence_embedding(pos_s)     # (N, 128)
                # InfoNCE — softmax sur toutes les phrases du batch
                # logits[i,j] = score th_i · phrase_j / temp
                logits = F.normalize(th, dim=-1) @ F.normalize(pos_feat, dim=-1).T / self.mi_temp  # (N, N)
                labels = torch.arange(N, device=th.device)
                loss_t2s = F.cross_entropy(logits,   labels)  # th_i → bonne phrase_i
                loss_s2t = F.cross_entropy(logits.T, labels)  # phrase_i → bon th_i
                mi = (loss_t2s + loss_s2t) / 2
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
    def get_ranking_scores(self, graph, utf, itf, urf, irf, topk=5):
        # SR eval : topic seul (utf/itf) — urf/irf exclus car ils dérivent sous BPR
        graph.nodes['user'].data['tf'] = utf
        graph.nodes['item'].data['tf'] = itf
        dev = next(self.topic_linear.parameters()).device
        def _get(edges):
            th = self.topic_linear(torch.cat([edges.src['tf'].to(dev), edges.dst['tf'].to(dev)], dim=1))
            return {'_th': th}
        graph.apply_edges(_get)
        th       = graph.edata['_th']                      # (E, 128)
        pos_sid  = graph.edata['sentence_id']              # (E, max_sents)
        E = th.shape[0]
        # Prend la première phrase valide (déterministe pour l'eval)
        valid_mask = pos_sid > 0
        first_idx  = valid_mask.long().argmax(dim=1)
        pos_s      = pos_sid[torch.arange(E, device=pos_sid.device), first_idx]  # (E,)
        # Filtre : uniquement les interactions dont la phrase cible a un topic
        if hasattr(self, '_has_topic'):
            topic_mask = self._has_topic[pos_s.cpu()]          # (E,) bool
            valid_idx  = topic_mask.nonzero(as_tuple=True)[0]  # indices originaux conservés
            th    = th[topic_mask.to(dev)]
            pos_s = pos_s[topic_mask.to(dev)]
        else:
            valid_idx = torch.arange(E)
        if pos_s.shape[0] == 0:
            return {'Pre': [], 'Rec': [], 'F1': [], 'nDCG': []}, valid_idx
        pos_feat   = self.sentence_embedding(pos_s.to(dev))                       # (E', 128)
        pos_s_np   = pos_s.cpu().numpy()
        # Score matrix : cosine similarity (F.normalize → invariant à topic_norm)
        th_n       = F.normalize(th,       dim=-1)
        pf_n       = F.normalize(pos_feat, dim=-1)
        score_mat  = (th_n @ pf_n.T).cpu()                # (E', E')
        _, ranked  = torch.sort(score_mat, dim=1, descending=True)
        topk_idx   = ranked[:, :topk].numpy()
        topk_sids  = pos_s_np[topk_idx]
        true_sids  = pos_s_np.reshape(-1, 1)
        return calc_ranking_metrics(topk_sids, true_sids), valid_idx


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
            self.review_embedding,
            params.user_size,
            params.item_size,
            params.gcn_out_units,
            params.num_layers,
            dropout_rate=params.gcn_dropout,
            minmax_alpha=params.minmax_alpha,
            no_review=getattr(params, 'no_review', False)
        )
        self.topic_encoder = TopicGraphEncoder(
            self.sentence_embedding, params.global_topic_size, params.gcn_out_units
        )
        self.topic_decoder = SentenceRetrival(
            params.gcn_out_units, 5, self.review_embedding, self.sentence_embedding,
            mi_temp=params.mi_temp
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

        combined_u = F.normalize(utf, dim=-1) + F.normalize(urf, dim=-1)
        combined_i = F.normalize(itf, dim=-1) + F.normalize(irf, dim=-1)
        # MI training sur utf/itf seuls — isole le signal topic du drift BPR
        ed_mi, ranking_loss = self.topic_decoder(
            pos_graph, urf, irf, utf, itf,
            neg_strategy=self.neg_strategy, n_neg=self.n_neg
        )

        src_pos, dst_pos = pos_graph.edges()
        u_emb     = urf[src_pos]
        i_pos_emb = irf[dst_pos]
        N = i_pos_emb.shape[0]

        h_pos     = self.topic_decoder.rating_linear(torch.cat([u_emb, i_pos_emb], dim=1))
        score_pos = self.topic_decoder.item_scorer(h_pos).squeeze(-1)

        # Binaire : toutes les interactions ont le même poids (pas de weighting par rating)
        bpr_terms = []
        for _ in range(self.n_neg):
            perm = torch.randperm(N, device=i_pos_emb.device)
            clash = perm == torch.arange(N, device=i_pos_emb.device)
            if clash.any():
                perm[clash] = (perm[clash] + 1) % N
            i_neg_emb = i_pos_emb[perm]
            h_neg     = self.topic_decoder.rating_linear(torch.cat([u_emb, i_neg_emb], dim=1))
            score_neg = self.topic_decoder.item_scorer(h_neg).squeeze(-1)
            bpr_terms.append(F.logsigmoid(score_pos - score_neg))

        bpr_raw = -torch.stack(bpr_terms).mean(0)   # (N,)
        if getattr(self, 'bpr_logdeg', False) and self.rating_encoder.item_degree_tensor is not None:
            _pos_global = pos_graph.nodes['item'].data['_ID'][dst_pos]
            _deg = self.rating_encoder.item_degree_tensor[_pos_global].float()
            _alpha = torch.log1p(_deg) / torch.log1p(self.rating_encoder.item_degree_tensor.max().float() + 1e-8)
            _w = (1.0 - _alpha).clamp(min=1e-3)
            _w = _w / _w.mean()   # normalise pour garder la même magnitude globale
            bpr_loss = (bpr_raw * _w).mean()
        else:
            bpr_loss = bpr_raw.mean()
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
                print(
                    f"[BPR_DEBUG] step={self._bpr_step}"
                    f" | n_neg={self.n_neg}"
                    f" | diff mean={diff.mean():.4f} std={diff.std():.4f}"
                    f" | pos={score_pos.detach().mean():.4f} neg={score_neg.detach().mean():.4f}"
                    f" | neg_deg mean={_neg_deg.mean():.1f} min={_neg_deg.min():.0f} max={_neg_deg.max():.0f}",
                    flush=True
                )

        return bpr_loss, ed_mi, ranking_loss, urf, irf

    def _get_item_emb_global(self, global_iids):
        # gate input = [pre-ci norm (cached), log1p-degree (static)] — same as training
        device    = self.rating_encoder.item_embedding.device
        cache     = self.rating_encoder._h_collab_cache
        pre_ci_n  = torch.tensor([cache[i.item()] for i in global_iids], device=device).unsqueeze(-1)
        deg       = self.rating_encoder.item_degree_tensor[global_iids]
        log_deg_n = (torch.log1p(deg.float()) /
                     torch.log1p(self.rating_encoder.item_degree_tensor.max().float() + 1e-8)).unsqueeze(-1)
        collab    = self.rating_encoder.item_embedding[global_iids]
        emb       = collab  # GCN pur, pas de fusion explicite
        return self.rating_encoder.ifc(self.rating_encoder.agg_act(emb))

    @torch.no_grad()
    def evaluate_sentence_ranking(self, dataloader, raw_graph, sampler, etype='valid', topk=5):
        # Test C : pool global — évalue chaque query contre TOUTES les phrases test avec topic
        device = self.review_embedding.weight.device
        has_topic = getattr(self.topic_decoder, '_has_topic', None)

        # Passe 1 : collecter tous les (th, pos_s) du split
        all_th, all_pos_s, all_ratings = [], [], []
        for rating_input_nodes, pos_graph, _neg_graph, rating_encoder_blocks in dataloader:
            decoder_graph = pos_graph[etype].to(device)
            input_nodes, _, blocks = sampler.sample(
                raw_graph,
                {'user': decoder_graph.nodes['user'].data['_ID'].cpu(),
                 'item': decoder_graph.nodes['item'].data['_ID'].cpu()}
            )
            rating_input_nodes    = {k: v.to(device) for k, v in rating_input_nodes.items()}
            input_nodes           = {k: v.to(device) for k, v in input_nodes.items()}
            blocks                = [b.to(device) for b in blocks]
            rating_encoder_blocks = [b.to(device) for b in rating_encoder_blocks]
            urf, irf = self.rating_encoder(rating_input_nodes, rating_encoder_blocks)
            utf, itf = self.topic_encoder(input_nodes, blocks)
            decoder_graph.nodes['user'].data['tf'] = utf
            decoder_graph.nodes['item'].data['tf'] = itf
            dev = next(self.topic_decoder.topic_linear.parameters()).device
            def _get(edges):
                return {'_th': self.topic_decoder.topic_linear(
                    torch.cat([edges.src['tf'].to(dev), edges.dst['tf'].to(dev)], dim=1))}
            decoder_graph.apply_edges(_get)
            th      = decoder_graph.edata['_th']          # (E, D)
            pos_sid = decoder_graph.edata['sentence_id']  # (E, max_sents)
            E = th.shape[0]
            valid_m  = pos_sid > 0
            first_idx = valid_m.long().argmax(dim=1)
            pos_s    = pos_sid[torch.arange(E, device=pos_sid.device), first_idx]
            ratings  = decoder_graph.edata['rating'].cpu().tolist()
            # Filtre topic
            if has_topic is not None:
                tmask = has_topic[pos_s.cpu()]
                th    = th[tmask.to(device)]
                pos_s = pos_s[tmask.to(device)]
                ratings = [ratings[i] for i in tmask.nonzero(as_tuple=True)[0].tolist()]
            all_th.append(th.cpu())
            all_pos_s.append(pos_s.cpu())
            all_ratings.extend(ratings)

        all_th    = torch.cat(all_th,    dim=0)  # (N_total, D)
        all_pos_s = torch.cat(all_pos_s, dim=0)  # (N_total,)

        # Passe 2 : pool global — embeddings de toutes les phrases cibles
        all_pos_feat = self.topic_decoder.sentence_embedding(all_pos_s.to(device)).cpu()  # (N_total, D)
        th_n   = F.normalize(all_th,       dim=-1)
        pf_n   = F.normalize(all_pos_feat, dim=-1)

        # Score matrix (N_total, N_total) — par batch pour éviter OOM
        N = th_n.shape[0]
        chunk = 256
        ndcg_list, pre_list, rec_list = [], [], []
        pos_s_np = all_pos_s.numpy()
        for start in range(0, N, chunk):
            th_chunk  = th_n[start:start+chunk].to(device)
            scores    = (th_chunk @ pf_n.T.to(device)).cpu()  # (chunk, N)
            _, ranked = torch.sort(scores, dim=1, descending=True)
            topk_idx  = ranked[:, :topk].numpy()
            topk_sids = pos_s_np[topk_idx]
            true_sids = pos_s_np[start:start+chunk].reshape(-1, 1)
            m = calc_ranking_metrics(topk_sids, true_sids)
            ndcg_list.extend(m['nDCG']); pre_list.extend(m['Pre']); rec_list.extend(m['Rec'])

        print(f"[GLOBAL_POOL] N_eval={N}", flush=True)
        return {'Pre': np.mean(pre_list), 'Rec': np.mean(rec_list),
                'F1':  np.mean([2*p*r/(p+r) if p+r>0 else 0 for p,r in zip(pre_list,rec_list)]),
                'nDCG': np.mean(ndcg_list)}

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

    def compute_fusion_loss(self, user_emb, h_modal_pos, h_modal_neg, lambda_f=1e-4, weights=None, cosnorm=False):
        if cosnorm:
            u = F.normalize(user_emb,    dim=-1)
            p = F.normalize(h_modal_pos, dim=-1)
            n = F.normalize(h_modal_neg, dim=-1)
        else:
            u, p, n = user_emb, h_modal_pos, h_modal_neg
        pos_score = (u * p).sum(dim=-1)
        neg_score = (u * n).sum(dim=-1)
        per_item  = -F.logsigmoid(pos_score - neg_score)
        if weights is not None:
            return lambda_f * (weights * per_item).mean()
        return lambda_f * per_item.mean()


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
    params.global_topic_size = dataset.graph.nodes['topic'].data['global_topic_id'].max() + 1

    all_sentence_ids = torch.cat([train_sentence_ids, valid_sentence_ids, test_sentence_ids])
    print(f"Sentence IDs — min: {torch.min(all_sentence_ids).item()}, max: {torch.max(all_sentence_ids).item()}")

    net = Net(dataset.review_embedding, dataset.sentence_embedding, params)
    net._sr_formula  = params.sr_formula
    net.bpr_logdeg   = params.bpr_logdeg
    # Masque has_topic : seules les phrases avec topic sont évaluées en SR
    import pickle as _pkl
    _topic_pkl = f'/home/infres/belguith/PFE/HADSF_test/checkpoint/{params.dataset_name}/BERT-Whitening/topic_and_sentence.pkl'
    _sid_to_topic = _pkl.load(open(_topic_pkl, 'rb'))['sid_to_topic']
    _max_sid = max(_sid_to_topic.keys()) + 1
    _has_topic = torch.zeros(_max_sid + 1, dtype=torch.bool)
    for _sid in _sid_to_topic:
        _has_topic[_sid] = True
    net.topic_decoder._has_topic = _has_topic
    print(f"[SR_FILTER] {_has_topic.sum().item():,} sids avec topic sur {_max_sid:,} total", flush=True)
    net = net.to(params.device)

    v_feat, t_feat = load_modal_features('/home/infres/belguith/PFE/bm3_data/musical')
    modal_enc = ModalEncoder(v_feat, t_feat, embed_dim=params.gcn_out_units).to(params.device)

    _sent_params  = list(net.topic_decoder.sentence_embedding.parameters())
    _sent_ids     = set(id(p) for p in _sent_params)
    _other_params = [p for p in list(net.parameters()) + list(modal_enc.parameters())
                     if id(p) not in _sent_ids]
    optimizer = torch.optim.Adam([
        {'params': _other_params,  'lr': params.train_lr},
        {'params': _sent_params,   'lr': params.train_lr * params.sent_lr_scale},
    ])
    print(f"[LR] No scheduler — LR fixe={params.train_lr}  sent_emb_lr={params.train_lr*params.sent_lr_scale} (×{params.sent_lr_scale})", flush=True)
    logger.info("Loading network finished ...\n")

    train_dataloader, valid_dataloader, test_dataloader = dataset.get_dataloaders(
        batch_size=params.batch_size, num_layers=params.num_layers
    )
    graph = dataset.graph
    topic_sampler = dataset.get_topic_sentence_sampler()

    _sent_emb_init = (net.topic_decoder.sentence_embedding.weight.data.detach().clone()
                      if params.lambda_reg > 0 else None)

    best_valid_ndcg = 0.0
    best_test_ndcg  = 0.0
    no_better_valid = 0
    best_iter       = -1
    learning_rate   = params.train_lr
    repr_norm_history = []

    # ── SR freeze tracking ──────────────────────────────────────────────────
    _best_sr_ndcg     = 0.0
    _best_sr_epoch    = -1
    _sr_no_improve    = 0
    _topic_frozen     = False
    _best_sr_snapshot = None

    h_modal, h_v, h_t = modal_enc()

    if not params.no_modal:
        with torch.no_grad():
            _n_items = net.rating_encoder.item_embedding.shape[0]
            _target_norm = net.rating_encoder.user_embedding.norm(dim=-1).mean()
            if params.rand_init:
                _e = net.rating_encoder.item_embedding.data
                _e_norms = _e.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                net.rating_encoder.item_embedding.data.copy_(_e / _e_norms * _target_norm)
                print(f"[INIT] item_embedding ← random direction (user_norm {_target_norm:.3f}) — GRCA actif, pas d'init h_modal", flush=True)
            else:
                _h_dir = h_modal[:_n_items] / h_modal[:_n_items].norm(dim=-1, keepdim=True).clamp(min=1e-8)
                net.rating_encoder.item_embedding.data.copy_(_h_dir * _target_norm)
                print(f"[INIT] item_embedding ← h_modal direction (norm orig→{_target_norm:.3f})", flush=True)
    else:
        with torch.no_grad():
            _target_norm = net.rating_encoder.user_embedding.norm(dim=-1).mean()
            _e = net.rating_encoder.item_embedding.data
            _e_norms = _e.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            net.rating_encoder.item_embedding.data.copy_(_e / _e_norms * _target_norm)
        print(f"[INIT] no_modal=True — item_embedding normalisé à user norm ({_target_norm:.3f})", flush=True)
    net.rating_encoder.h_modal = h_modal

    import pandas as pd
    # Musical_interactions_reviews.csv : IDs alignés sur Musical_reviews_item2id.json (0..4885)
    _df_deg_train = pd.read_csv('/home/infres/belguith/PFE/processed/Musical_interactions_reviews.csv')
    _deg_train = _df_deg_train['iid'].value_counts()
    _n_init = net.rating_encoder.item_embedding.shape[0]
    _deg_tensor_train = torch.zeros(_n_init, dtype=torch.float32)
    for _iid, _cnt in _deg_train.items():
        if int(_iid) < _n_init:
            _deg_tensor_train[int(_iid)] = float(_cnt)
    net.rating_encoder.item_degree_tensor = _deg_tensor_train.to(params.device)
    print("[INIT] item_degree_tensor injected (Musical_interactions_reviews.csv, IDs 0..4885)", flush=True)

    _diag_deg = _deg_tensor_train
    _diag_cold_idx = torch.where((_diag_deg >= 5) & (_diag_deg <= 10))[0]
    _diag_warm_idx = torch.where(_diag_deg > 20)[0]
    print(f"[DIAG_SETUP] cold_items={len(_diag_cold_idx)}, warm_items={len(_diag_warm_idx)}", flush=True)

    net.rating_encoder.reset_epoch_diag()
    net.eval()
    logger.info('Test - ' + format_dict_to_str(
        net.evaluate_sentence_ranking(test_dataloader, graph, topic_sampler, etype='test')
    ))
    net.train()
    logger.info("Start training ...")

    with torch.no_grad():
        _u0 = net.rating_encoder.user_embedding.norm(dim=-1).mean().item()
        _i0 = net.rating_encoder.item_embedding.norm(dim=-1).mean().item()
        repr_norm_history.append((0, _u0, _i0, _u0 / (_i0 + 1e-9)))
        print(f"[REPR_NORM] epoch=  0  user={_u0:.4f}  item={_i0:.4f}  ratio={_u0/(_i0+1e-9):.4f}  (INIT)", flush=True)
        net.rating_encoder._pre_ci_global_max.fill_(
            net.rating_encoder.item_embedding.norm(dim=-1).max().clamp(min=1e-8).item()
        )
        print(f"[INIT] _pre_ci_global_max={net.rating_encoder._pre_ci_global_max.item():.4f}", flush=True)

    _mi_history   = []    # historique des MI moyennes par epoch

    for iter_idx in range(1, params.epoch):
        net.rating_encoder.reset_epoch_diag()
        net.train()
        _epoch_mi_sum   = 0.0
        _epoch_mi_count = 0
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
            # target_norm dynamique = norme actuelle de item_embedding (suit la croissance BPR)
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
                # f_loss pondérée par alpha(item) — warm (α élevé) = signal BPR fiable
                _deg_f   = net.rating_encoder.item_degree_tensor[_item_gids_f]
                if params.minmax_alpha:
                    _dmin = net.rating_encoder.item_degree_tensor.min().float()
                    _dmax = net.rating_encoder.item_degree_tensor.max().float()
                    _alpha_f = (_deg_f.float() - _dmin) / (_dmax - _dmin + 1e-8)
                else:
                    _alpha_f = (torch.log1p(_deg_f.float()) /
                                torch.log1p(net.rating_encoder.item_degree_tensor.max().float() + 1e-8))
                if params.floss_mode == 'cosnorm':
                    _hf_pos = _h_modal_per_edge
                    _hf_neg = _h_modal_per_edge[_perm_f]
                elif params.floss_mode == 'dynorm':
                    # 846863 original: F.normalize(h_modal) * item_emb_norm (dynorm)
                    _target = net.rating_encoder.item_embedding.norm(dim=-1).mean().detach()
                    _hm_n = F.normalize(_h_modal_per_edge, dim=-1) * _target
                    _hf_pos = _hm_n
                    _hf_neg = _hm_n[_perm_f]
                else:
                    _ratio = (_u_emb_per_edge.norm(dim=-1, keepdim=True).detach() /
                              _h_modal_per_edge.norm(dim=-1, keepdim=True).detach().clamp(min=1e-8))
                    if params.floss_mode == 'sqrt_uscale':
                        _ratio = _ratio.sqrt()
                    _hf_pos = _h_modal_per_edge * _ratio
                    _hf_neg = _h_modal_per_edge[_perm_f] * _ratio[_perm_f]
                f_loss = net.compute_fusion_loss(
                    _u_emb_per_edge, _hf_pos, _hf_neg,
                    lambda_f=params.lambda_f, weights=_alpha_f,
                    cosnorm=(params.floss_mode == 'cosnorm')
                )

            _eff_lambda_mi = 0.0 if (_topic_frozen or getattr(params, 'no_review', False)) else (
                params.lambda_mi * min(1.0, iter_idx / params.lambda_mi_warmup) if params.lambda_mi_warmup > 0 else params.lambda_mi)
            mi_term = (_eff_lambda_mi * mi_score if _eff_lambda_mi > 0
                       else torch.tensor(0.0, device=params.device))
            if _eff_lambda_mi > 0:
                _epoch_mi_sum   += mi_score.item()
                _epoch_mi_count += 1

            # ── GRCA loss — InfoNCE(h_modal, item_collab) pondérée par (1-alpha) ──
            grca_loss = torch.tensor(0.0, device=params.device)
            if (not params.no_modal and params.lambda_grca > 0
                    and hasattr(net.rating_encoder, '_grca_item_collab')):
                _col  = net.rating_encoder._grca_item_collab          # (N, 128) GCN pur
                _mod  = net.rating_encoder._grca_h_modal               # (N, 128) modal
                _w    = (1 - net.rating_encoder._grca_alpha).detach().squeeze(-1)  # (N,)
                _a    = F.normalize(_col, dim=-1)
                _b    = F.normalize(_mod, dim=-1)
                _logits = torch.matmul(_a, _b.T) / 0.5                # (N, N)
                _lbl    = torch.arange(_logits.shape[0], device=params.device)
                _lab    = F.cross_entropy(_logits,   _lbl, reduction='none')  # (N,)
                _lba    = F.cross_entropy(_logits.T, _lbl, reduction='none')  # (N,)
                grca_loss = params.lambda_grca * (_w * (_lab + _lba) / 2).mean()

            reg_loss = (params.lambda_reg *
                        ((net.topic_decoder.sentence_embedding.weight - _sent_emb_init) ** 2).mean()
                        if params.lambda_reg > 0 else torch.tensor(0.0, device=params.device))

            loss = r_loss + 0.1 * modal_loss + f_loss + mi_term + grca_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), params.train_grad_clip)
            optimizer.step()

            net.rating_encoder._epoch_floss_sum  += f_loss.item()
            net.rating_encoder._epoch_grca_sum   += grca_loss.item()
            net.rating_encoder._epoch_n_batches  += 1
            net.rating_encoder._epoch_bpr_batches.append(r_loss.item())

            pbar.set_description(
                f"train_loss={r_loss:.4f}, MI={mi_score:.2f}, "
                f"mi_term={mi_term.item():.4f}, modal={modal_loss.item():.4f}, "
                f"f_loss={f_loss.item():.6f}, grca={grca_loss.item():.4f}"
            )

        _nb = net.rating_encoder._epoch_n_batches
        if _nb > 0:
            _avg_floss = net.rating_encoder._epoch_floss_sum / _nb
            _avg_grca  = net.rating_encoder._epoch_grca_sum  / _nb
            print(f"[LOSSES] epoch={iter_idx:>3d}  f_loss={_avg_floss:.6f}  grca={_avg_grca:.6f}", flush=True)
            _bpr_arr = np.array(net.rating_encoder._epoch_bpr_batches)
            print(f"[BPR_STD] epoch={iter_idx:>3d}  mean={_bpr_arr.mean():.4f}  std={_bpr_arr.std():.4f}  "
                  f"min={_bpr_arr.min():.4f}  max={_bpr_arr.max():.4f}", flush=True)

        with torch.no_grad():
            _n = net.rating_encoder.item_embedding.shape[0]
            _h = net.rating_encoder.h_modal
            _e = net.rating_encoder.item_embedding
            _collab_norm = _e.norm(dim=-1).mean()
            _modal_norm  = _h.norm(dim=-1).mean()
            _user_emb_norm = net.rating_encoder.user_embedding.norm(dim=-1).mean().item()
            print(f"[EMB_NORM] epoch={iter_idx:>3d}  user={_user_emb_norm:.4f}  item={_collab_norm.item():.4f}", flush=True)
            print(f"[DIAG] h_modal std_per_dim={_h.std(dim=0).mean():.4f} norm={_modal_norm:.3f} | collab norm={_collab_norm:.3f}", flush=True)
            _sim_pre = F.cosine_similarity(_h[:_n], _e).mean()
            print(f"[DIAG] cosine sim h_modal/item_emb (pré-GCN) = {_sim_pre:.4f}", flush=True)
            _sim_n = net.rating_encoder._epoch_sim_n
            if _sim_n > 0:
                _sim_post = net.rating_encoder._epoch_sim_sum / _sim_n
                print(f"[DIAG] cosine sim h_modal/item_collab (post-GCN) = {_sim_post:.4f}", flush=True)

            with torch.no_grad():
                _hv = (modal_enc.norm_v(F.relu(modal_enc.image_trs(modal_enc.image_embedding.weight)))
                       if hasattr(modal_enc, 'norm_v')
                       else modal_enc.image_trs(modal_enc.image_embedding.weight))
                _ht = (modal_enc.norm_t(F.relu(modal_enc.text_trs(modal_enc.text_embedding.weight)))
                       if hasattr(modal_enc, 'norm_t')
                       else modal_enc.text_trs(modal_enc.text_embedding.weight))
                _sv = modal_enc.query_v(_hv)
                _st = modal_enc.query_t(_ht)
                _ww = torch.softmax(torch.cat([_sv, _st], dim=-1), dim=-1)
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

        if params.lambda_mi > 0 and _epoch_mi_count > 0:
            _epoch_avg_mi = _epoch_mi_sum / _epoch_mi_count
            print(f"[MI_AVG] epoch={iter_idx}  avg_MI={_epoch_avg_mi:.4f}", flush=True)

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
            if no_better_valid >= params.train_early_stopping_patience:
                logger.info("Early stopping threshold reached. Stop training.")
                break

        print(f"[LR] epoch={iter_idx}  lr={learning_rate:.6f}", flush=True)

        logger.info(logging_str)
        net.eval()
        _sr_result = net.evaluate_sentence_ranking(test_dataloader, graph, topic_sampler, etype='test')
        _sr_ndcg   = float(_sr_result.get('nDCG', 0.0))
        logger.info('Test - ' + format_dict_to_str(_sr_result))
        net.train()

        # ── SR freeze logic ─────────────────────────────────────────────────
        if not _topic_frozen and iter_idx >= params.lambda_mi_warmup:
            if _sr_ndcg > _best_sr_ndcg:
                _best_sr_ndcg     = _sr_ndcg
                _best_sr_epoch    = iter_idx
                _sr_no_improve    = 0
                _best_sr_snapshot = {
                    'decoder': copy.deepcopy(net.topic_decoder.state_dict()),
                    'encoder': copy.deepcopy(net.topic_encoder.state_dict()),
                }
                print(f"[SR_TRACK] Nouveau meilleur SR={_best_sr_ndcg:.4f} @ep{iter_idx}", flush=True)
            else:
                _sr_no_improve += 1
                if _sr_no_improve >= params.sr_freeze_patience:
                    net.topic_decoder.load_state_dict(
                        {k: v.to(params.device) for k, v in _best_sr_snapshot['decoder'].items()}
                    )
                    net.topic_encoder.load_state_dict(
                        {k: v.to(params.device) for k, v in _best_sr_snapshot['encoder'].items()}
                    )
                    for p in net.topic_encoder.parameters():
                        p.requires_grad_(False)
                    for p in net.topic_decoder.parameters():
                        p.requires_grad_(False)
                    _topic_frozen = True
                    print(f"[SR_FREEZE] topic_encoder + topic_decoder gelés @ep{iter_idx} "
                          f"(best SR={_best_sr_ndcg:.4f} @ep{_best_sr_epoch}, "
                          f"patience={params.sr_freeze_patience})", flush=True)

    logger.info(f'Best Iter Idx={best_iter}, Best Valid nDCG@10={best_valid_ndcg:.4f}, Best Test nDCG@10={best_test_ndcg:.4f}')
    logger.info(params.model_save_path)
    logger.info("=== Repr norm history ===")
    logger.info(f"{'Epoch':>6} | {'user_norm':>9} | {'item_norm':>9} | {'ratio_u/i':>9}")
    for ep, un, it, rt in repr_norm_history:
        flag = "  <-- DESEQUILIBRE" if rt > 2.0 or rt < 0.5 else ""
        logger.info(f"{ep:>6d} | {un:>9.4f} | {it:>9.4f} | {rt:>9.4f}{flag}")


def test(params):
    import eval_binary as _em
    _em.test(params, net_class=Net)


if __name__ == '__main__':
    config_args = config()
    if config_args.test_only:
        test(config_args)
    else:
        train(config_args)
        test(config_args)

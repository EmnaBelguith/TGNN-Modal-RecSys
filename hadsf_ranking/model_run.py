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
    parser = argparse.ArgumentParser(description='GCMC2')
    parser.add_argument('--device', default='0', type=int,
                        help='Running device. E.g `--device 0`, if using cpu, set `--device -1`')
    parser.add_argument('-dn', '--dataset_name', type=str)
    parser.add_argument('-dp', '--dataset_path', type=str, help='raw dataset file path')
    parser.add_argument('--model_save_path', type=str, help='The model saving path')
    parser.add_argument('--review_feat_size', type=int, default=128)

    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=float, default=10000)
    parser.add_argument('--train_grad_clip', type=float, default=1.0)
    parser.add_argument('--train_lr', type=float, default=0.001)
    parser.add_argument('--train_min_lr', type=float, default=0.0001)
    parser.add_argument('--train_lr_decay_factor', type=float, default=0.5)
    parser.add_argument('--train_decay_patience', type=int, default=8)
    parser.add_argument('--train_early_stopping_patience', type=int, default=10)
    parser.add_argument('--train_classification', type=bool, default=True)

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lambda_f', type=float, default=1e-4)
    parser.add_argument('--gcn_dropout', type=float, default=0.7)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--ed_alpha', type=float, default=.1)
    parser.add_argument('--model_short_name', type=str, default='RHGC4')
    parser.add_argument('--w_deg_init', type=float, default=2.0)
    parser.add_argument('--b_deg_init', type=float, default=-5.0)

    args = parser.parse_args()

    # args.dataset_name = 'Digital_Music_5'
    # args.dataset_path = '/home/d1/shuaijie/data/Digital_Music_5/Digital_Music_5.json'
    # args.review_feat_size = 64
    # args.gcn_dropout = 0.7
    args.device = 0
    # args.num_layers = 2
    args.batch_size = 512

    # args.dataset_name = 'Toys_and_Games_5'
    # args.dataset_path = '/home/d1/shuaijie/data/Toys_and_Games_5/Toys_and_Games_5.json'
    # args.review_feat_size = 64
    # args.gcn_dropout = 0.8
    # args.ed_alpha = 0.1
    # args.device = 0
    # args.num_layers = 2
    args.batch_size = 512
    # args.epoch = 2000

    # args.dataset_name = 'Sports_and_Outdoors_5'
    # args.dataset_path = '/home/d1/shuaijie/data/Sports_and_Outdoors_5/Sports_and_Outdoors_5.json'

 #   args.dataset_name = 'Musical_Instruments'
 #   args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_Musical_Instruments_output/filtered_Musical_Instruments_output.jsonl'
 #   args.dataset_name = 'Musical_Instruments_aspect'
 #   args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_Musical_Instruments_output_aspect/filtered_Musical_Instruments_output.jsonl'
 #   args.dataset_name = 'Industrial_and_Scientific'
 #   args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_Industrial_and_Scientific_output/filtered_Industrial_and_Scientific_output.jsonl' # 修改为您的 JSONL 文件路径
 #   args.dataset_name = 'yelp_reviews'
 #   args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_yelp_reviews_output/filtered_yelp_restaurant_reviews_output.jsonl' # 修改为您的 JSONL 文件路径
 #   args.dataset_name = 'Industrial_and_Scientific_raw_part2'
 #  args.dataset_path = '/home/zheng/RatingTopicGraph/filtered_Industrial_and_Scientific_aspect_raw_10-50/filtered_Industrial_and_Scientific_aspect_raw_10-50.jsonl' # 修改为您的 JSONL 文件路径

  # Nouveaux chemins (à ajouter)
    args.dataset_name = 'Musical_HADSF'
    args.dataset_path = '/home/infres/belguith/PFE/processed/Musical_reviews_with_aspects.jsonl'
    args.gcn_dropout = 0.8
    args.ed_alpha = 2.0
    args.device = 0
    args.num_layers = 2
    args.batch_size = 512

    #     # args.dataset_name = 'Health_and_Personal_Care_5'
    #     # args.dataset_path = '/home/d1/shuaijie/data/Health_and_Personal_Care_5/Health_and_Personal_Care_5.json'
    # args.gcn_dropout = 0.7
    # args.device = 0
    # args.epoch = 200

    # args.dataset_name = 'Yelp2013'
    # args.dataset_path = '/home/d1/shuaijie/data/yelp-recsys-2013/yelp2013.json'
    # args.gcn_dropout = 0.7
    args.device = 0
    args.dataset_name = 'Musical_HADSF'
    args.dataset_path = '/home/infres/belguith/PFE/processed/Musical_reviews_with_aspects.jsonl'
    args.gcn_dropout = 0.8
    args.ed_alpha = 2.0
    # args.epoch = 800
    # args.num_layers = 2

    # args.dataset_name = 'Yelp1'
    # args.dataset_path = '/home/d1/shuaijie/data/yelp1/yelp2013.json'
    # args.gcn_dropout = 0.8
    # args.epoch = 300

    # args.device = torch.device(args.device) if args.device >= 0 else torch.device('cpu')
    args.device = f"cuda:{args.device}" if args.device >= 0 else 'cpu'
    args.model_short_name = 'RHGC4_ranking'

    # configure save_fir to save all the info
    args.model_save_path = f'model_save/{args.dataset_name}/{args.model_short_name}_layers_{args.num_layers}.pt'
    if not os.path.isdir(f'model_save/{args.dataset_name}'):
        os.makedirs(f'model_save/{args.dataset_name}')

    args.gcn_out_units = args.review_feat_size

    return args


def reset_parameters(model):
    em_set = set(['review_embedding.weight', 'sentence_embedding.weight'])

    for n, p in model.named_parameters():
        if n in em_set:
            continue
        if p.dim() > 1 :
            nn.init.xavier_uniform_(p)


def format_dict_to_str(data_dict):
    result = []
    for k, v in data_dict.items():
        result.append(f'{k}: {v:>.4f}')
    return ', '.join(result)


class GCMCGraphConv(nn.Module, ABC):

    def __init__(self, \
                 feature_size, \
                 review_embedding, \
                 add_embedding_mapping=False, \
		 add_review=False, \
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
        max_rid = rid.max().item()
        assert torch.all(rid < num_embeddings), f"存在 review_id 超出范围: 最大值 {rid.max().item()}，num_embeddings {num_embeddings}"
    
        review_feat = self.review_embedding(rid)
        return review_feat

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
            rst = rst * graph.dstdata['ci']
            rst = self.linear(rst)

        return rst 


class MultiLayerHeteroGraphConv(nn.Module):

    def __init__(self, rating_values, review_embedding, user_size, item_size, msg_units, num_layers, aggregate='sum', dropout_rate=0.0):
        super(MultiLayerHeteroGraphConv, self).__init__()
        
        assert num_layers > 0, "The numbder of conv layers must have at least one!"
        self.num_layers = num_layers
        self.conv_layers = nn.ModuleList()
        rating_values = [str(r) for r in rating_values]
        self.rating_values = rating_values

        self.user_embedding = nn.Parameter(torch.Tensor(user_size, msg_units))
        self.item_embedding = nn.Parameter(torch.Tensor(item_size, msg_units))
        nn.init.xavier_uniform_(self.item_embedding.unsqueeze(0)).squeeze(0)
        self.h_modal = None  # sera injecté depuis ModalEncoder
        self.item_degree_tensor = None  # sera injecté depuis l'extérieur
        self.w_deg = nn.Parameter(torch.tensor(2.0))
        self.b_deg = nn.Parameter(torch.tensor(-5.0))
        self.gate_residual = nn.Sequential(
            nn.Linear(msg_units * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )

        sub_conv = nn.ModuleDict()

        for l in range(num_layers):
            sub_conv = {}
            for rating in rating_values:

                rating = str(rating)
                rev_rating = f'rev-{rating}'
                sub_conv[rating] = GCMCGraphConv(msg_units, \
                                                 review_embedding, \
						 add_embedding_mapping = l == 0, \
						 add_review = l == (num_layers - 1), \
						 dropout_rate=dropout_rate)
                sub_conv[rev_rating] = GCMCGraphConv(msg_units, 
                                                     review_embedding, \
						     add_embedding_mapping = l == 0, \
						     add_review = l == (num_layers - 1), \
						     dropout_rate=dropout_rate)

            self.conv_layers.append(nn.ModuleDict(sub_conv))
        
        self.ufc = nn.Linear(msg_units, msg_units)
        self.ifc = nn.Linear(msg_units, msg_units)
        self.dropout = nn.Dropout(0.5)
        self.agg_act = nn.GELU()
        

    def forward(self, input_nodes, encoder_blocks):
        
        user_outputs = []
        item_outputs = []

        # first layer
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
                    item_modal  = self.h_modal[input_nodes['item']]
                    degrees = self.item_degree_tensor[input_nodes['item']]
                    alpha_deg = torch.sigmoid(torch.abs(self.w_deg) * torch.log1p(degrees) + self.b_deg).unsqueeze(-1)
                    gate_input = torch.cat([item_collab, item_modal], dim=-1)
                    residual = self.gate_residual(gate_input) * 0.2
                    alpha = torch.clamp(alpha_deg + residual, 0.0, 1.0)
                    item_init   = alpha * item_collab + (1 - alpha) * item_modal
                    if self.training:
                        self._last_alpha   = alpha.squeeze(-1)
                        self._last_degrees = degrees
                    if not self.training:
                        _iids_cpu = input_nodes['item'].cpu()
                        # Contribution reelle modal vs collab (norme)
                        _modal_contrib  = ((1 - alpha) * item_modal).norm(dim=-1).detach().cpu()
                        _collab_contrib = (alpha * item_collab).norm(dim=-1).detach().cpu()
                        _ratio_modal    = _modal_contrib / (_modal_contrib + _collab_contrib + 1e-8)
                        if not hasattr(self, '_alpha_buffer'): self._alpha_buffer = []
                        self._alpha_buffer.extend(zip(_iids_cpu.tolist(), _ratio_modal.tolist()))
                    u_o = conv_layer[f'rev-{rating}'](block['item', f'rev-{rating}', 'user'], 
                                                      item_init)
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
        # user_outputs = user_outputs.sum(1)
        user_outputs = self.agg_act(user_outputs)
        user_outputs = self.dropout(user_outputs)
        user_outputs = self.ufc(user_outputs)
        # item_outputs = item_outputs.sum(1)
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
        """
        :param x: bs * dim
        :param y: bs * dim
        :param y_neg: bs * dim
        :return:
        """

        # y += torch.zeros_like(y).normal_(0, 0.01)

        if y_neg is None:
            idx = torch.randperm(y.shape[0])
            y_neg = y[idx, :]

        # y += y_sim
        neg_scores = (y_neg @ self.w * x).sum(1)
        neg_labels = neg_scores.new_zeros(neg_scores.shape)
        neg_loss = self.bce_loss(neg_scores, neg_labels)

        scores = (y @ self.w * x).sum(1)
        labels = scores.new_ones(scores.shape)
        pos_loss = self.bce_loss(scores, labels)

        loss = pos_loss + neg_loss
        return loss

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
            # Affine(feature_size),
            nn.GELU(),
            nn.Linear(feature_size, feature_size, bias=False), 
            # Affine(feature_size),
            nn.GELU(),
            nn.Linear(feature_size, feature_size, bias=False), 
        )
        self.gelu = nn.GELU()

        self.sentence_w1 = nn.Parameter(torch.Tensor(topic_size, feature_size))
        self.sentence_score_w = nn.Parameter(torch.Tensor(topic_size, feature_size))
        # self.sentence_w2 = nn.Parameter(torch.Tensor(topic_size, feature_size, feature_size))

        self.sentence_linear = nn.Linear(feature_size, feature_size)

        self.topic_user_linear = nn.Linear(feature_size, feature_size)
        self.topic_item_linear = nn.Linear(feature_size, feature_size)
        self.topic_user_w = nn.Parameter(torch.Tensor(topic_size, feature_size))
        self.topic_item_w = nn.Parameter(torch.Tensor(topic_size, feature_size))

        self.dropout = nn.Dropout(0.5)

    def sentence_to_topic(self, graph, sentence_id):
        sent_feat = self.sentence_embedding(sentence_id)
        # sent_feat = self.sentence_w(sent_feat)

        stid = graph.srcdata['global_topic_id']
        graph.srcdata['h'] = self.sentence_w1[stid] * sent_feat
        # graph.srcdata['attn_score'] = (self.sentence_score_w[stid] * sent_feat).sum(-1, keepdim=True)

        with graph.local_scope():

            graph.update_all(lambda edges: {'m': edges.src['h']},
                             fn.sum(msg='m', out='sum_h'))
            calc_attn = lambda edges: {'attn_score': (edges.src['h'] * edges.dst['sum_h']).sum(1, keepdim=True)}
            graph.apply_edges(calc_attn)
            
            # graph.apply_edges(lambda e: {'attn_score': e.src['attn_score']})
            graph.edata['attn_score'] = edge_softmax(graph, graph.edata['attn_score'])

            # message passing with attention
            graph.update_all(lambda edges: {'m': edges.src['h'] * self.dropout(edges.data['attn_score'])},
                             fn.sum(msg='m', out='h'))
            
            result = graph.dstdata['h']

        result = self.sentence_linear(result)
        return result

    def topic_to_user_item(self, graphs, topic_feat):

        graph = graphs[('topic', 'topic_to_user', 'user')]
        # graph.srcdata['h'] = topic_feat
        stid = graph.srcdata['global_topic_id']
        graph.srcdata['h'] = self.gelu(topic_feat * self.topic_user_w[stid])

        with graph.local_scope():

            # calculate attention weight
            graph.update_all(lambda edges: {'m': edges.src['h']},
                             fn.sum(msg='m', out='sum_h'))
            calc_attn = lambda edges: {'attn_score': (edges.src['h'] * edges.dst['sum_h']).sum(1, keepdim=True)}
            graph.apply_edges(calc_attn)
            e_attn = graph.edata['attn_score']
            graph.edata['attn_score'] = edge_softmax(graph, e_attn)

            # message passing with attention
            graph.update_all(lambda edges: {'m': edges.src['h'] * self.dropout(edges.data['attn_score'])},
                             fn.sum(msg='m', out='h'))
            
            user_feat = graph.dstdata['h']

        user_feat = self.topic_user_linear(user_feat)
        
        # item
        graph = graphs[('topic', 'topic_to_item', 'item')]
        # graph.srcdata['h'] = topic_feat
        stid = graph.srcdata['global_topic_id']
        graph.srcdata['h'] = self.gelu(topic_feat * self.topic_item_w[stid])

        with graph.local_scope():

            # calculate attention weight
            graph.update_all(lambda edges: {'m': edges.src['h']},
                             fn.sum(msg='m', out='sum_h'))
            calc_attn = lambda edges: {'attn_score': (edges.src['h'] * edges.dst['sum_h']).sum(1, keepdim=True)}
            graph.apply_edges(calc_attn)
            e_attn = graph.edata['attn_score']
            graph.edata['attn_score'] = edge_softmax(graph, e_attn)

            # message passing with attention
            graph.update_all(lambda edges: {'m': edges.src['h'] * self.dropout(edges.data['attn_score'])},
                             fn.sum(msg='m', out='h'))
            
            item_feat = graph.dstdata['h']

        item_feat = self.topic_item_linear(item_feat)

        return user_feat, item_feat

    def forward(self, input_nodes, encoder_blocks):
        topic_embedding = self.sentence_to_topic(encoder_blocks[0][('sentence', 'sentence_to_topic', 'topic')], \
                                                 input_nodes['sentence'])
        uo, io = self.topic_to_user_item(encoder_blocks[1], topic_embedding)
        return uo, io


class SentenceRetrival(nn.Module):

    def __init__(self,
                 in_units,
                 num_classes,
                 review_embedding,
                 sentence_embedding,
                 dropout_rate=0.0):
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
        # sid: bs * k
        length = (sid > 0).float().sum(dim=-1, keepdim=True) + 1e-9
        review_feat = self.sentence_embedding(sid).sum(dim=-2)
        review_feat = review_feat / length
        return review_feat

    def calc_sentence_ranking(self, edges):
        rh = self.rating_linear(torch.cat([edges.src['rf'], edges.dst['rf']], dim=1))

        th = self.topic_linear(torch.cat([edges.src['tf'], edges.dst['tf']], dim=1))
        th = th + rh
        pos_sid = edges.data['sentence_id']
        neg_sid = torch.randint(1, self.sentence_embedding.weight.shape[0],
                                pos_sid.shape,
                                device=pos_sid.device)

        pos_review = self.get_review_feature(pos_sid)
        neg_review = self.get_review_feature(neg_sid)

        pos_score = (th * pos_review).sum(1)
        neg_score = (th * neg_review).sum(1)
        loss = -(pos_score - neg_score).sigmoid().log()

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
        
    def forward(self, graph, urf, irf, utf, itf):
        graph.nodes['user'].data['rf'] = urf
        graph.nodes['item'].data['rf'] = irf
        graph.nodes['user'].data['tf'] = utf
        graph.nodes['item'].data['tf'] = itf

        with graph.local_scope():
            graph.apply_edges(self.calc_sentence_ranking)
            mi_score = graph.edata['mi_score']
            ranking_loss = graph.edata['ranking_loss']
        return mi_score.mean(), ranking_loss.mean()

    def measure_sim(self, interaction_feat, sid_list):
        # bs * dim, bs * k
        min_sid = torch.min(sid_list).item()
        max_sid = torch.max(sid_list).item()
        num_embeddings = self.sentence_embedding.num_embeddings

#        print(f"sid_list - min: {min_sid}, max: {max_sid}, num_embeddings: {num_embeddings}")

          # 添加断言确保索引合法
        assert min_sid >= 0, f"sid_list contains negative indices: min_sid={min_sid}"
        assert max_sid < num_embeddings, f"sid_list contains indices >= num_embeddings: max_sid={max_sid}, num_embeddings={num_embeddings}"


        sent_feat = self.sentence_embedding(sid_list)  # bs * k * dim
        scores = torch.einsum('bd,bkd->bk', interaction_feat, sent_feat)
        return scores

    # 分 batch 计算 ranking method
    @staticmethod
    def _rank_batch(_h, _cand, _trues, _measure_func, topk):
        """
        _h: 交互表征
        _cand: 可能的sentence id list
        _trues: 真实的 sentence id list
        _measure_func: 
        """
        _cand_mask = (_cand > 0).float()
        _ml = _cand_mask.int().sum(dim=1).max()
        _cand = _cand[:, :_ml]
        _cand_mask = _cand_mask[:, :_ml]
        _scores = _measure_func(_h, _cand)
        _, _topk_idx = torch.topk(_scores, k=topk, dim=-1)
        _topk_items = torch.gather(_cand, 1, _topk_idx)
        _topk_items = _topk_items.cpu().numpy()
        _trues = _trues.cpu().numpy()
        # import pdb; pdb.set_trace()
        return calc_ranking_metrics(_topk_items, _trues)

    @ torch.no_grad()
    def get_ranking_scores(self, graph, user_feat, item_feat, topk=5):
        graph.nodes['item'].data['th'] = item_feat
        graph.nodes['user'].data['th'] = user_feat

        def _get(edges): 
            h = self.topic_linear(torch.cat([edges.src['th'], edges.dst['th']], dim=1))
            cand_sent = edges.dst['candidate_sentence_id']
            return {'th': h, 'cand_sid': cand_sent}

        graph.apply_edges(_get)

        h = graph.edata['th']
        true_sents = graph.edata['sentence_id']
        cand_sents = graph.edata['cand_sid']

        rank_list = []
        _bs = 2000
        for i in range(0, h.shape[0], _bs):
            _sent_scores = self._rank_batch(h[i: i + _bs], \
                                            cand_sents[i: i + _bs],
                                            true_sents[i: i + _bs], \
                                            self.measure_sim, \
                                            topk=topk)
            rank_list.append(_sent_scores)

        result = {k: sum([list(_rl[k]) for _rl in rank_list], [])
                  for k in rank_list[0].keys()}
        # result = {k: np.mean(v) for k, v in result.items()}
        return result

	    
def calc_ranking_metrics(topk_items, true_list):
    precision, recall = precision_recall_score(topk_items, true_list)
    f1 = [ 2 * p * r / (p + r) if p + r > 0. else 0. for p, r in zip(precision, recall) ]
    ndcg = ndcg_score(topk_items, true_list)
    
    return {'Pre': precision, \
            'Rec': recall, \
            'F1': f1, \
            'nDCG': ndcg}
    

def precision_recall_score(predicts, trues):
    
    def pr_each(ps, ts):
        ps = ps[ps > 0]
        ts = ts[ts > 0]
        if len(ts) < 1 or len(ps) < 1:  # some reviews are empty
            return 0., 0.
        inter = np.intersect1d(ps, ts)
        return len(inter) / len(ps), len(inter) / len(ts)
    
    prs, rcs = zip(*[pr_each(predicts[i], trues[i]) for i in range(len(predicts))])
    return prs, rcs


def ndcg_score(predicts, trues):
    
    def _ndcg(ps, ts):
        ps = ps[ps > 0]
        ts = ts[ts > 0]
        # if len(ts) < 1:
        #     return 0.
        if len(ts) < 1 or len(ps) < 1:  # some reviews are empty
            return 0.
        isin = np.isin(ps, ts)
        if isin.sum() == 0.:
            return 0.
        dcg = isin / np.log2(np.arange(2, len(isin) + 2))
        # idcg = 1 / np.log2(np.arange(2, len(isin) + 2))
        idcg = np.sort(isin)[::-1] / np.log2(np.arange(2, len(isin) + 2))
        return np.sum(dcg) / np.sum(idcg)
    
    return [_ndcg(predicts[i], trues[i]) for i in range(len(predicts))]


class Net(nn.Module):

    def __init__(self, review_embedding, sentence_embedding, params):
        super(Net, self).__init__()

        self.sentence_embedding = sentence_embedding# nn.Embedding.from_pretrained(sentence_embedding)
        self.review_embedding = nn.Embedding.from_pretrained(review_embedding)
        self.rating_encoder = MultiLayerHeteroGraphConv(params.rating_values, \
                                                        self.review_embedding, \
                                                        params.user_size, \
                                                        params.item_size, \
                                                        params.gcn_out_units, \
                                                        params.num_layers, \
                                                        dropout_rate=params.gcn_dropout)

        self.topic_encoder = TopicGraphEncoder(self.sentence_embedding, params.global_topic_size, params.gcn_out_units)
        self.topic_decoder = SentenceRetrival(params.gcn_out_units, 5, self.review_embedding, self.sentence_embedding)

        reset_parameters(self)

    def state_dict(self):
        # exclude review embedding
        sd = super().state_dict()
        pop_keys = []
        for k in sd.keys():
            if 'review_embedding' in k or 'sentence_embedding' in k:
                pop_keys.append(k)
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

        # Sentence ranking loss (topic-based, unchanged)
        ed_mi, ranking_loss = self.topic_decoder(pos_graph,
                                                 urf, irf,
                                                 utf + urf, itf + irf)

        # BPR avec in-batch negatives :
        # pour chaque paire (u_i, pos_i), le négatif est pos_{perm(i)} (item d'une autre paire du batch)
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

        # BPR diagnostic — premiers batches + tous les 100 steps
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
        """Représentation approchée pour items hors-batch (sans propagation GNN)."""
        collab  = self.rating_encoder.item_embedding[global_iids]
        modal   = self.rating_encoder.h_modal[global_iids]
        deg     = self.rating_encoder.item_degree_tensor[global_iids]
        alpha   = torch.sigmoid(
            torch.abs(self.rating_encoder.w_deg) * torch.log1p(deg) + self.rating_encoder.b_deg
        ).unsqueeze(-1)
        gate_in = torch.cat([collab, modal], dim=-1)
        res     = self.rating_encoder.gate_residual(gate_in) * 0.2
        alpha   = torch.clamp(alpha + res, 0.0, 1.0)
        emb     = alpha * collab + (1 - alpha) * modal
        return self.rating_encoder.ifc(self.rating_encoder.agg_act(emb))

    @torch.no_grad()
    def evaluate_sentence_ranking(self, dataloader, raw_graph, sampler, etype='valid', topk=5):
        device = self.review_embedding.weight.device
        # group_scores 用于保存每个评分组下的各指标列表
        group_scores = defaultdict(lambda: defaultdict(list))
        scores_list = []
        for rating_input_nodes, pos_graph, _neg_graph, rating_encoder_blocks in dataloader:
            decoder_graph = pos_graph[etype].to(device)
            input_nodes, _, blocks = sampler.sample(raw_graph,
                                                    {'user': decoder_graph.nodes['user'].data['_ID'].cpu(),
                                                     'item': decoder_graph.nodes['item'].data['_ID'].cpu()})

            rating_input_nodes = {k: v.to(device) for k, v in rating_input_nodes.items()}
            input_nodes = {k: v.to(device) for k, v in input_nodes.items()}
            blocks = [b.to(device) for b in blocks]
            rating_encoder_blocks = [b.to(device) for b in rating_encoder_blocks]

            # 获取每个样本的真实评分（假设取值为 1～5）
            ratings = decoder_graph.edata['rating'].cpu().tolist()
            
            urf, irf = self.rating_encoder(rating_input_nodes, rating_encoder_blocks)
            utf, itf = self.topic_encoder(input_nodes, blocks)
            ranking_scores = self.topic_decoder.get_ranking_scores(decoder_graph, \
                                                                   utf + urf, \
                                                                   # itf + itf, \
                                                                   itf + irf, \
                                                                   topk)
            scores_list.append(ranking_scores)
                    # 将每个样本的各指标按照真实评分分组
            for idx, rating in enumerate(ratings):
                group = int(rating)  # 取值 1～5
                for metric, values in ranking_scores.items():
                    group_scores[group][metric].append(values[idx])

            # 对每个评分组下各指标计算平均值
        group_metrics = {}
        for group, metrics in group_scores.items():
            group_metrics[group] = {metric: np.mean(vals) for metric, vals in metrics.items()}
        
        # 输出每个评分组的排序指标
        print("各评分组（1～5）的排序指标：")
        for group in sorted(group_metrics.keys()):
            metrics = group_metrics[group]
            print("评分组 {}: Pre = {:.4f}, Rec = {:.4f}, F1 = {:.4f}, nDCG = {:.4f}".format(
                group, metrics.get('Pre', 0), metrics.get('Rec', 0), metrics.get('F1', 0), metrics.get('nDCG', 0)
            ))
        
        scores_list = {k: sum([list(_rl[k]) for _rl in scores_list], [])
                       for k in scores_list[0].keys()}
        scores_list = {k: np.mean(v) for k, v in scores_list.items()}
        return scores_list


    @torch.no_grad()
    def evaluate_ranking_ndcg(self, dataloader, dataset, K=10,
                               relevance_threshold=3, etype='valid',
                               n_neg=99, seed=42):
        """
        Protocole standard RecSys : 1 pos + n_neg négatifs par user.
        Négatifs = items jamais vus (ni train ni etype) tirés aléatoirement.
        Identique à evaluate_item_ranking dans evaluate_model_run.py.
        """
        import math
        device = self.review_embedding.weight.device
        self.eval()
        rng = np.random.default_rng(seed)

        # 1. train_seen par user
        graph = dataset.graph
        train_u, train_i = graph['train'].edges()
        train_seen = defaultdict(set)
        for u, i in zip(train_u.tolist(), train_i.tolist()):
            train_seen[u].add(i)

        # 2. Collecter embeddings user/item et items positifs depuis le dataloader
        user_emb  = {}
        item_emb  = {}
        pos_items = defaultdict(dict)  # uid → {iid: rating}

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

        # 3. Scorer (N, dim) paires sur CPU pour éviter les transfers répétés
        rating_linear = self.topic_decoder.rating_linear.to('cpu')
        item_scorer   = self.topic_decoder.item_scorer.to('cpu')

        def score_pairs(u_emb_t, i_embs_t):
            u_rep = u_emb_t.unsqueeze(0).expand(i_embs_t.shape[0], -1)
            return item_scorer(rating_linear(torch.cat([u_rep, i_embs_t], dim=1))).squeeze(-1)

        # 4. Pool de candidats = tous les items dont on a l'embedding
        known_items = np.array(sorted(item_emb.keys()))

        # 5. nDCG@K avec 1 pos + n_neg négatifs
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

        # Remettre les modules sur GPU
        self.topic_decoder.rating_linear.to(device)
        self.topic_decoder.item_scorer.to(device)

        return float(np.mean(ndcg_list)) if ndcg_list else 0.0

    def compute_fusion_loss(self, user_emb, h_modal_pos, h_modal_neg, lambda_f=1e-4):
        """
        Fusion Loss (CFMM Eq.6 adapté).
        Supervise h_modal avec les préférences users directement.
        user_emb    : (n, 128) — représentations users (urf) pour interactions positives
        h_modal_pos : (n, 128) — h_modal des items aimés (rating >= 3)
        h_modal_neg : (n, 128) — h_modal des items non aimés (rating <= 2)
        """
        pos_score = (user_emb * h_modal_pos).sum(dim=-1)
        neg_score = (user_emb * h_modal_neg).sum(dim=-1)
        loss = -F.logsigmoid(pos_score - neg_score).mean()
        return lambda_f * loss


    import random

def train(params):
    import random
    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    print(f"[SEED] {params.seed}", flush=True)

    # global logger

    # save_dir, code_dir =  make_trainging_log_dir(base_dir='logs/', \
    #     					 data_name=params.dataset_name, \
    #     					 model_name=params.model_short_name)
    # params.model_save_path = f"{save_dir}/model.pkl"

    # copy_py_files('./', code_dir)

    # logger = get_logger(params.model_short_name, f"{save_dir}/training.log")

    # global tb_logger
    # tb_logger = get_tensorboard_writer(f"{save_dir}/tb")

    global logger

    logger = get_logger(params.model_short_name, None)

    logger.info(f"Parameters:\n{args_to_str(params)}")

    dataset = GraphData(params.dataset_name,
                        params.dataset_path) 
                        # device='cpu')

    train_sentence_ids = dataset.train_sentence_ids
    valid_sentence_ids = dataset.valid_sentence_ids
    test_sentence_ids = dataset.test_sentence_ids

    params.user_size = dataset.user_size
    params.item_size = dataset.item_size
    params.rating_values = dataset.possible_rating_values

    params.global_topic_size = dataset.graph.nodes['topic'].data['global_topic_id'].max() + 1


  
    all_sentence_ids = torch.cat([train_sentence_ids, valid_sentence_ids, test_sentence_ids])
    max_sentence_id = torch.max(all_sentence_ids).item()
    current_num_embeddings = dataset.sentence_embedding.num_embeddings

    print(f"All sentence IDs - min: {torch.min(all_sentence_ids).item()}, max: {max_sentence_id}")
    print(f"Current sentence_embedding - num_embeddings: {current_num_embeddings}")

    net = Net(dataset.review_embedding, dataset.sentence_embedding, params)
    net = net.to(params.device)
    with torch.no_grad():
        net.rating_encoder.w_deg.fill_(params.w_deg_init)
        net.rating_encoder.b_deg.fill_(params.b_deg_init)
   # === MODAL ENCODER ===
    v_feat, t_feat = load_modal_features('/home/infres/belguith/PFE/bm3_data/musical')
    modal_enc = ModalEncoder(v_feat, t_feat, embed_dim=params.gcn_out_units).to(params.device)
# =====================
    learning_rate = params.train_lr
    optimizer = torch.optim.Adam(
      list(net.parameters()) + list(modal_enc.parameters()),
      lr=learning_rate
    )
    logger.info("Loading network finished ...\n")

    train_dataloader, valid_dataloader, test_dataloader = dataset.get_dataloaders(batch_size=params.batch_size, num_layers=params.num_layers)
    graph = dataset.graph
    topic_sampler = dataset.get_topic_sentence_sampler()

    best_valid_ndcg = 0.0
    best_test_ndcg  = 0.0

    no_better_valid = 0
    best_iter = -1

    repr_norm_history = []  # (epoch, urf_norm, irf_norm, ratio)

    # logger.info('Valid -' + format_dict_to_str(net.evaluate_sentence_ranking(valid_dataloader, graph, topic_sampler, etype='valid')))
    h_modal, h_v, h_t = modal_enc()
    
    if not hasattr(train, '_batch_count'):
       train._batch_count = 0
     
    train._batch_count += 1
    
    if train._batch_count % 50 == 1:
       print(f"[Modal] h_modal norm={h_modal.norm(dim=-1).mean():.4f}")
       print(f"[Modal] h_v={h_v.norm(dim=-1).mean():.4f} | h_t={h_t.norm(dim=-1).mean():.4f}") 
    net.rating_encoder.h_modal = h_modal
    with torch.no_grad():
        h_modal_norm = F.normalize(h_modal, p=2, dim=-1)
        net.rating_encoder.item_embedding.data.copy_(
            h_modal_norm[:net.rating_encoder.item_embedding.shape[0]]
        )
    print(f"[INIT] item_embedding initialisé depuis h_modal normalisé (norme avant={h_modal.norm(dim=-1).mean():.3f} → après=1.0)")
    import pandas as pd
    _df_deg_train = pd.read_csv('/home/infres/belguith/PFE/processed/Musical_interactions.csv')
    _deg_train = _df_deg_train['iid'].value_counts()
    _n_init = net.rating_encoder.item_embedding.shape[0]
    _deg_tensor_train = torch.zeros(_n_init, dtype=torch.float32)
    for _iid, _cnt in _deg_train.items():
        if int(_iid) < _n_init:
            _deg_tensor_train[int(_iid)] = float(_cnt)
    net.rating_encoder.item_degree_tensor = _deg_tensor_train.to(params.device)
    print("[INIT] item_degree_tensor injecté avec vrais degrés", flush=True)

    # Précalcul des indices cold/warm pour les diagnostics
    _diag_deg = _deg_tensor_train  # CPU
    _diag_cold_idx = torch.where((_diag_deg >= 5) & (_diag_deg <= 10))[0]
    _diag_warm_idx  = torch.where(_diag_deg > 20)[0]
    print(f"[DIAG_SETUP] cold_items={len(_diag_cold_idx)}, warm_items={len(_diag_warm_idx)}", flush=True)

    logger.info('Test - '+ format_dict_to_str(net.evaluate_sentence_ranking(test_dataloader, graph, topic_sampler, etype='test')))

    logger.info("Start training ...")

    # Normes à l'initialisation (epoch 0, avant tout entraînement)
    with torch.no_grad():
        _u0 = net.rating_encoder.user_embedding.norm(dim=-1).mean().item()
        _i0 = net.rating_encoder.item_embedding.norm(dim=-1).mean().item()
        _r0 = _u0 / (_i0 + 1e-9)
        repr_norm_history.append((0, _u0, _i0, _r0))
        print(f"[REPR_NORM] epoch=  0  user={_u0:.4f}  item={_i0:.4f}  ratio_u/i={_r0:.4f}  (INIT - avant training)", flush=True)

    for iter_idx in range(1, params.epoch):
        net.train()

        pbar = tqdm(train_dataloader)
        # pbar = train_dataloader
        train_rmse = []
        train_mi = []
        for rating_input_nodes, pos_graph, _neg_graph, rating_blocks in pbar:
            topic_input_nodes, _, topic_blocks = topic_sampler.sample(graph,
                                                                      {'user': pos_graph.nodes['user'].data['_ID'],
                                                                       'item': pos_graph.nodes['item'].data['_ID']})

            rating_input_nodes = {k: v.to(params.device) for k, v in rating_input_nodes.items()}
            topic_input_nodes  = {k: v.to(params.device) for k, v in topic_input_nodes.items()}
            pos_graph_train    = pos_graph['train'].to(params.device)
            rating_blocks      = [b.to(params.device) for b in rating_blocks]
            topic_blocks       = [b.to(params.device) for b in topic_blocks]

            h_modal, h_v, h_t = modal_enc()
            net.rating_encoder.h_modal = h_modal

            # Poids cold-start : upweight les interactions des items peu vus
            _, _dst_idx = pos_graph_train.edges()
            _edge_iids     = pos_graph_train.dstdata['_ID'][_dst_idx]
            _edge_degrees  = net.rating_encoder.item_degree_tensor[_edge_iids].clamp(min=1.0)
            _median_deg    = net.rating_encoder.item_degree_tensor[net.rating_encoder.item_degree_tensor > 0].median()
            _sample_weight = torch.log1p(_median_deg / _edge_degrees)
            _sample_weight = _sample_weight / _sample_weight.mean()

            r_loss, mi_score, ranking_loss, urf, irf = net.calc_loss(
                rating_input_nodes, rating_blocks,
                topic_input_nodes, topic_blocks,
                pos_graph_train,
                sample_weight=_sample_weight
            )

            batch_items = pos_graph_train.nodes['item'].data['_ID']
            modal_loss  = modal_enc.calculate_loss_infonce(h_v, h_t, batch_items)

            # Fusion Loss — BPR sur embeddings modaux (rating >= 3 pos, <= 2 neg)
            _src_idx_f, _dst_idx_f = pos_graph_train.edges()
            _ratings_batch    = pos_graph_train.edata['rating']
            _item_gids_f      = pos_graph_train.dstdata['_ID'][_dst_idx_f]
            _u_emb_per_edge   = urf[_src_idx_f]
            _h_modal_per_edge = h_modal[_item_gids_f]
            _mask_pos = _ratings_batch >= 3
            _mask_neg = _ratings_batch <= 2
            f_loss = torch.tensor(0.0, device=params.device)
            if _mask_pos.sum() > 0 and _mask_neg.sum() > 0:
                _n_f = min(_mask_pos.sum().item(), _mask_neg.sum().item())
                f_loss = net.compute_fusion_loss(
                    _u_emb_per_edge[_mask_pos][:_n_f],
                    _h_modal_per_edge[_mask_pos][:_n_f],
                    _h_modal_per_edge[_mask_neg][:_n_f],
                    lambda_f=params.lambda_f
                )

            loss = r_loss + 0.1 * modal_loss + f_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), params.train_grad_clip)
            optimizer.step()

            # print(f"ranking:{ranking_loss.item():.4f}")
            pbar.set_description(f"train_loss={r_loss:.4f}, MI={mi_score:.2f}, ranking={ranking_loss.item():.4f}, modal={modal_loss.item():.4f}, f_loss={f_loss.item():.6f}")
            
            # with torch.no_grad():
            #     predict_ratings = net.predicts_to_ratings(predicts)
            #     rmse = predict_ratings - edge_subgraph.edata['rating']
            #     train_rmse.extend(rmse.cpu().tolist())
            #     train_mi.append(mi_score.cpu().item())

        # train_rmse = np.sqrt(np.power(np.array(train_rmse), 2).mean())
        # train_mi = np.mean(train_mi)
        train_rmse = r_loss.item()
        train_mi = mi_score.item()
        # print gate alpha stats
        with torch.no_grad():
            _n = net.rating_encoder.item_embedding.shape[0]
            # DIAG cosine similarity
            _h = net.rating_encoder.h_modal
            _e = net.rating_encoder.item_embedding
            _n = _e.shape[0]
            # DIAG variance + normes + comparaison
            _collab_norm = _e.norm(dim=-1).mean()
            _modal_norm  = _h.norm(dim=-1).mean()
            print(f"[DIAG] h_modal std_per_dim={_h.std(dim=0).mean():.4f} std_per_item={_h.std(dim=1).mean():.4f} norm={_modal_norm:.3f} | collab norm={_collab_norm:.3f}", flush=True)
            # DIAG cosine similarity
            _sim = torch.nn.functional.cosine_similarity(_h[:_n], _e).mean()
            print(f"[DIAG] cosine sim h_modal/collab = {_sim:.4f}", flush=True)
            # GATE monitor
            _degrees_all = net.rating_encoder.item_degree_tensor[:_n]
            _alpha = torch.sigmoid(torch.abs(net.rating_encoder.w_deg) * torch.log1p(_degrees_all) + net.rating_encoder.b_deg).unsqueeze(-1)
            _std_items = _alpha.std(dim=0).mean()
            _std_dims  = _alpha.std(dim=1).mean()
            print(f"[GATE] epoch={iter_idx} mean={_alpha.mean():.3f} std_global={_alpha.std():.3f} std_inter_items={_std_items:.3f} std_inter_dims={_std_dims:.3f} min={_alpha.min():.3f} max={_alpha.max():.3f}", flush=True)

            # DIAG 1 : est-ce que img/txt convergent vers 0 ou 1 ?
            _w = torch.softmax(torch.cat([
                net.rating_encoder.h_modal.new_zeros(1),  # placeholder
            ], dim=-1), dim=-1) if False else None
            # On utilise h_modal et h_v/h_t depuis modal_enc
            with torch.no_grad():
                _hv = modal_enc.norm_v(torch.nn.functional.relu(modal_enc.image_trs(modal_enc.image_embedding.weight))) if hasattr(modal_enc, 'norm_v') else modal_enc.image_trs(modal_enc.image_embedding.weight)
                _ht = modal_enc.norm_t(torch.nn.functional.relu(modal_enc.text_trs(modal_enc.text_embedding.weight))) if hasattr(modal_enc, 'norm_t') else modal_enc.text_trs(modal_enc.text_embedding.weight)
                _sv = _hv.norm(dim=-1, keepdim=True)
                _st = _ht.norm(dim=-1, keepdim=True)
                _ww = torch.softmax(torch.cat([_sv, _st], dim=-1), dim=-1)
                _w_img = _ww[:,0].mean().item()
                _w_txt = _ww[:,1].mean().item()
                _w_std = _ww[:,0].std().item()
            print(f"[EPOCH_MODAL] epoch={iter_idx} w_img={_w_img:.3f} w_txt={_w_txt:.3f} std={_w_std:.3f} {'⚠ COLLAPSE' if _w_img > 0.95 or _w_txt > 0.95 else 'OK'}", flush=True)

            # DIAG 2 : corrélation alpha / degré des items
            _item_ids = torch.arange(_e.shape[0])
            _degrees  = torch.tensor([
                net.rating_encoder.item_embedding.shape[0]
            ] * _e.shape[0]).float()  # placeholder
            # Vraie corrélation alpha_mean vs norme collab (proxy du degré)
            _alpha_mean_per_item = _alpha.mean(dim=-1)  # [n_items]
            _collab_norm_per_item = _e.norm(dim=-1)     # [n_items]
            _corr = torch.corrcoef(torch.stack([
                _alpha_mean_per_item, _collab_norm_per_item
            ]))[0,1].item()
            print(f"[CORR] alpha vs collab_norm = {_corr:.4f} {'(gate discrimine par confiance collab)' if abs(_corr) > 0.1 else '(gate independant de la norme collab)'}", flush=True)

            # DIAG 3 : contribution modale — norme de (1-alpha)*h_modal vs alpha*collab
            _modal_contrib  = ((1 - _alpha) * _h[:_n]).norm(dim=-1).mean().item()
            _collab_contrib = (_alpha * _e).norm(dim=-1).mean().item()
            print(f"[CONTRIB] collab={_collab_contrib:.3f} modal={_modal_contrib:.3f} ratio_modal={_modal_contrib/(_modal_contrib+_collab_contrib)*100:.1f}%", flush=True)

            # ── DIAG COLD-START : les 3 hypothèses ──────────────────────────────

            # [H1] collab ≈ modal pour cold ? → si oui, gate est vacuité
            _c_idx = _diag_cold_idx.to(_e.device)
            _w_idx = _diag_warm_idx.to(_e.device)
            _sim_cold = torch.nn.functional.cosine_similarity(_e[_c_idx], _h[_c_idx], dim=-1).mean().item()
            _sim_warm  = torch.nn.functional.cosine_similarity(_e[_w_idx],  _h[_w_idx],  dim=-1).mean().item()
            # item_init pour cold/warm
            _alpha_cold = torch.sigmoid(2.0 * torch.log1p(_diag_deg[_diag_cold_idx].to(_e.device)) - 5.0).unsqueeze(-1)
            _alpha_warm  = torch.sigmoid(2.0 * torch.log1p(_diag_deg[_diag_warm_idx].to(_e.device))  - 5.0).unsqueeze(-1)
            _init_cold = _alpha_cold * _e[_c_idx] + (1 - _alpha_cold) * _h[_c_idx]
            _init_warm  = _alpha_warm  * _e[_w_idx]  + (1 - _alpha_warm)  * _h[_w_idx]
            _sim_init_cold = torch.nn.functional.cosine_similarity(_init_cold, _h[_c_idx], dim=-1).mean().item()
            _sim_init_warm  = torch.nn.functional.cosine_similarity(_init_warm,  _h[_w_idx],  dim=-1).mean().item()
            print(f"[H1_GATE_VACUITE] epoch={iter_idx}", flush=True)
            print(f"  cosine_sim(collab, modal): cold={_sim_cold:.4f}  warm={_sim_warm:.4f}", flush=True)
            print(f"  cosine_sim(item_init, modal): cold={_sim_init_cold:.4f}  warm={_sim_init_warm:.4f}  (cold proche 1.0 → gate vacuité)", flush=True)

            # [H2] cold_reg supprimé (FIX1) — on log f_loss à la place
            print(f"[H2_FUSION_LOSS] epoch={iter_idx} f_loss={f_loss.item():.6f}", flush=True)

            # [H3] drift item_emb depuis h_modal — cold vs warm
            _drift_cold = (_e[_c_idx] - _h[_c_idx]).norm(dim=-1).mean().item()
            _drift_warm  = (_e[_w_idx]  - _h[_w_idx]).norm(dim=-1).mean().item()
            print(f"[H3_DRIFT] epoch={iter_idx} ||collab - modal||: cold={_drift_cold:.4f}  warm={_drift_warm:.4f}", flush=True)
            print(f"  (si cold_drift ≈ warm_drift → collab n'apprend rien de spécifique aux cold items)", flush=True)
            # ────────────────────────────────────────────────────────────────────

            # --- Comparaison norme user vs item (représentations finales pour la prédiction) ---
            _urf_norm = urf.detach().norm(dim=-1).mean().item()
            _irf_norm = irf.detach().norm(dim=-1).mean().item()
            _ratio    = _urf_norm / (_irf_norm + 1e-9)
            repr_norm_history.append((iter_idx, _urf_norm, _irf_norm, _ratio))
            print(f"[REPR_NORM] epoch={iter_idx:>3d}  user={_urf_norm:.4f}  item={_irf_norm:.4f}  ratio_u/i={_ratio:.4f}", flush=True)

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
                    logger.info("\tChange the LR to %g" % new_lr)
                    for p in optimizer.param_groups:
                        p['lr'] = learning_rate
                    no_better_valid = 0

        logger.info(logging_str)
        logger.info('Test - ' + format_dict_to_str(net.evaluate_sentence_ranking(test_dataloader, graph, topic_sampler, etype='test')))
        
    hparam_dict = args_to_dict(params)
    key_hparam_list = ['review_feat_size', 'gcn_dropout', 'num_layers']
    hparam_dict = {k: hparam_dict[k] for k in key_hparam_list}
    # tb_logger.add_hparams(hparam_dict=hparam_dict, \
    #                       metric_dict={'Valid_RMSE': best_valid_rmse, 'Test_RMSE': best_test_rmse}, \
    #                       run_name='metric')

    logger.info(f'Best Iter Idx={best_iter}, Best Valid nDCG@10={best_valid_ndcg:.4f}, Best Test nDCG@10={best_test_ndcg:.4f}')
    logger.info(params.model_save_path)

    # --- Résumé évolution norme user vs item ---
    logger.info("=== Évolution norme représentations finales user vs item par epoch ===")
    logger.info(f"{'Epoch':>6} | {'user_norm':>9} | {'item_norm':>9} | {'ratio_u/i':>9}")
    for ep, un, it, rt in repr_norm_history:
        flag = "  <-- DESEQUILIBRE" if rt > 2.0 or rt < 0.5 else ""
        logger.info(f"{ep:>6d} | {un:>9.4f} | {it:>9.4f} | {rt:>9.4f}{flag}")


def test(params):
    """Déplacé dans evaluate_model_run.py — conservé ici pour compatibilité ascendante."""
    from evaluate_model_run import test as _eval_test
    _eval_test(params)


def _test_legacy(params):
    from nltk.translate.bleu_score import sentence_bleu
    from rouge import Rouge
    # logger = get_logger(params.model_short_name, None)

    dataset = GraphData(params.dataset_name,
                        params.dataset_path) 
                        # device='cpu')

    params.user_size = dataset.user_size
    params.item_size = dataset.item_size
    params.rating_values = dataset.possible_rating_values

    params.global_topic_size = dataset.graph.nodes['topic'].data['global_topic_id'].max() + 1

    train_dataloader, valid_dataloader, test_dataloader = dataset.get_dataloaders(batch_size=params.batch_size, num_layers=params.num_layers)
    graph = dataset.graph
    topic_sampler = dataset.get_topic_sentence_sampler()

    net = Net(dataset.review_embedding, dataset.sentence_embedding, params)
    _ckpt = torch.load(params.model_save_path, weights_only=False)
    _modal_sd = _ckpt.pop('modal_enc', None)
    net.load_state_dict(_ckpt, strict=False)
    net = net.to(params.device)
    # Initialise h_modal pour test() — charger les poids entraînés
    from modal_encoder import ModalEncoder, load_modal_features
    v_feat, t_feat = load_modal_features('/home/infres/belguith/PFE/bm3_data/musical')
    _modal_enc_test = ModalEncoder(v_feat, t_feat, embed_dim=128).to(params.device)
    if _modal_sd is not None:
        missing, unexpected = _modal_enc_test.load_state_dict(_modal_sd, strict=False)
        matched = len(_modal_sd) - len(unexpected)
        print(f"[LOAD] modal_enc: {matched}/{len(_modal_sd)} clés chargées depuis checkpoint", flush=True)
        if missing:
            print(f"[WARN] Clés manquantes (resteront aléatoires): {missing}", flush=True)
    else:
        print("[WARN] modal_enc ABSENT du checkpoint — features aléatoires en test !", flush=True)
    with torch.no_grad():
        _h_modal_test, _, _ = _modal_enc_test()
    net.rating_encoder.h_modal = _h_modal_test

    # Charge les groupes cold/medium/warm pour évaluation
    import pandas as pd
    _df_deg = pd.read_csv('/home/infres/belguith/PFE/processed/Musical_interactions.csv')
    _deg = _df_deg['iid'].value_counts()
    _n_items = net.rating_encoder.item_embedding.shape[0]
    _deg_tensor = torch.zeros(_n_items, dtype=torch.float32)
    for _iid, _cnt in _deg.items():
        if int(_iid) < _n_items:
            _deg_tensor[int(_iid)] = float(_cnt)
    net.rating_encoder.item_degree_tensor = _deg_tensor.to(params.device)
    net.item_degree_groups = (
        set(_deg[(_deg>=5)&(_deg<=10)].index),
        set(_deg[(_deg>=11)&(_deg<=20)].index),
        set(_deg[_deg>20].index)
    )
    net._cs_buffer = []
    net._ranking_buffer = []
    net._alpha_buffer = []
    test_rmse,test_mae,test_mse = net.evaluate_rating(test_dataloader, etype='test')
    print(params.dataset_name)
    print(f'Test RMSE={test_rmse:.4f},Test MAE={test_mae:.4f},Test MSE={test_mse:.4f}')

    # Ranking evaluation depuis _ranking_buffer (uid, iid, pred_score, true_rating)
    if net._ranking_buffer:
        import math
        from collections import defaultdict
        user_items = defaultdict(list)
        for uid, iid, pred, true in net._ranking_buffer:
            user_items[uid].append((iid, pred, true))

        Ks = [3, 5, 10]
        ndcg_buf   = {k: [] for k in Ks}
        recall_buf = {k: [] for k in Ks}
        hr_buf     = {k: [] for k in Ks}
        prec_buf   = {k: [] for k in Ks}

        for uid, items in user_items.items():
            positives = set(iid for iid, _, r in items if r >= 3)
            if not positives:
                continue
            ranked_ids = [x[0] for x in sorted(items, key=lambda x: x[1], reverse=True)]
            n_pos = len(positives)
            for K in Ks:
                top_k = ranked_ids[:K]
                hits = sum(1 for iid in top_k if iid in positives)
                dcg  = sum(1.0/math.log2(i+2) for i, iid in enumerate(top_k) if iid in positives)
                idcg = sum(1.0/math.log2(i+2) for i in range(min(n_pos, K)))
                ndcg_buf[K].append(dcg/idcg if idcg > 0 else 0.0)
                recall_buf[K].append(hits / n_pos)
                hr_buf[K].append(1.0 if hits > 0 else 0.0)
                prec_buf[K].append(hits / K)

        print(f"\n[RANKING] threshold=3 (rating>=3 → positif), {len(user_items)} users évalués")
        print(f"  {'K':>4}  {'nDCG':>7}  {'Recall':>7}  {'HR':>7}  {'Prec':>7}")
        for K in Ks:
            if ndcg_buf[K]:
                print(f"  {K:>4}  {np.mean(ndcg_buf[K]):>7.4f}  {np.mean(recall_buf[K]):>7.4f}  {np.mean(hr_buf[K]):>7.4f}  {np.mean(prec_buf[K]):>7.4f}")
    print('Pre     Rec     F1      nDCG')
    for k in [10, 50]:
        scores = net.evaluate_sentence_ranking(test_dataloader, graph, topic_sampler, etype='test', topk=k)
        print('{Pre:.4f}\t{Rec:.4f}\t{F1:.4f}\t{nDCG:.4f}'.format(**scores))




def calc_bleu_metric(predict_list, true_list):
    # list of string

    b1l, b2l, b4l = [], [], []

    for p, t in zip(predict_list, true_list):
        p = p.split()
        t = t.split()
        b1 = sentence_bleu([t], p, weights=(1, 0, 0, 0))
        b2 = sentence_bleu([t], p, weights=(0.5, 0.5, 0, 0))
        b4 = sentence_bleu([t], p, weights=(0.25, 0.25, 0.25, 0.25))
        b1l.append(b1)
        b2l.append(b2)
        b4l.append(b4)

    return {'BLEU-1': np.mean(b1l), 'BLEU-2': np.mean(b2l), 'BLEU-4': np.mean(b4l)}


def calc_rouge_metric(predict_list, true_list):
    rouge = Rouge()
    predict_list = [' '.join(x) for x in predict_list]
    true_list = [' '.join(x) for x in true_list]
    rouge_scores = rouge.get_scores(predict_list, true_list, avg=True)
    rouge_scores = {k: v['f'] for k, v in rouge_scores.items()}
    return rouge_scores


if __name__ == '__main__':
    config_args = config()
    train(config_args)
    test(config_args)


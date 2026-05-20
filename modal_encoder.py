# coding: utf-8
# Adapté de BM3 
# Rôle : encodeur modal item uniquement (image + texte)
# LightGCN, users, BPR supprimés
# h_v et h_t produits pour injection dans TGNN

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity


class ModalEncoder(nn.Module):
    def __init__(self, v_feat, t_feat, embed_dim=128, dropout=0.1):
        """
        v_feat : numpy array (n_items, 2048) — features visuelles pré-extraites
        t_feat : numpy array (n_items, 384)  — features textuelles pré-extraites
        embed_dim : dimension de sortie (doit = msg_units de TGNN = 128)
        dropout : taux dropout pour les losses SSL
        """
        super(ModalEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.dropout_rate = dropout

        # Features pré-extraites — GELÉES (CNN et SentenceTransformer déjà appliqués)
        self.image_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(v_feat), freeze=True)
        self.text_embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(t_feat), freeze=True)

        # MLP apprenables — projetent vers embed_dim=128
        self.image_trs = nn.Linear(v_feat.shape[1], embed_dim)  # 2048 → 128
        self.text_trs  = nn.Linear(t_feat.shape[1], embed_dim)  # 384  → 128

        # Predictor SSL — comme BM3
        self.predictor = nn.Linear(embed_dim, embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)
        self.norm_t = nn.LayerNorm(embed_dim)
        self.query_v = nn.Linear(embed_dim, 1)
        self.query_t = nn.Linear(embed_dim, 1)
        nn.init.xavier_normal_(self.query_v.weight)
        nn.init.xavier_normal_(self.query_t.weight)
        # Initialisations Xavier — comme BM3
        nn.init.xavier_normal_(self.image_trs.weight)
        nn.init.xavier_normal_(self.text_trs.weight)
        nn.init.xavier_normal_(self.predictor.weight)

    def forward(self):
        """
        Calcule h_v et h_t pour TOUS les items.
        Retourne :
          h_v : (n_items, 128) — représentation visuelle
          h_t : (n_items, 128) — représentation textuelle
        Ces deux vecteurs sont ensuite injectés dans TGNN via h_modal.
        """
        h_v = self.norm_v(F.relu(self.image_trs(self.image_embedding.weight)))
        h_t = self.norm_t(F.relu(self.text_trs(self.text_embedding.weight)))
        score_v = self.query_v(h_v)
        score_t = self.query_t(h_t)
        weights = torch.softmax(torch.cat([score_v, score_t], dim=-1), dim=-1)
        h_modal = weights[:, 0:1] * h_v + weights[:, 1:2] * h_t
        if not hasattr(self, "_fwd_count"): self._fwd_count = 0
        self._fwd_count += 1
        if self._fwd_count % 100 == 0:
            print(f"[ModalEnc] step={self._fwd_count} w_img={weights[:,0].mean():.3f}+-{weights[:,0].std():.3f} w_txt={weights[:,1].mean():.3f}+-{weights[:,1].std():.3f} h_modal_norm={h_modal.norm(dim=-1).mean():.3f}", flush=True)
        return h_modal, h_v, h_t

    def calculate_loss(self, h_v, h_t, items):
        """
        Loss SSL inspirée de BM3.

        Paramètres :
          h_v   : (n_items, 128) — sortie image_trs sur tous les items
          h_t   : (n_items, 128) — sortie text_trs sur tous les items
          irf   : (batch, 128)   — sortie item du Rating GNN de TGNN
                                   c'est notre 'i_target' collaboratif  pas utilise pour le moment 
          items : (batch,)       — indices des items du batch courant

        Retourne : loss scalaire
        """
        # Targets avec dropout — comme BM3 (no_grad car targets fixes)
        with torch.no_grad():
            v_target = F.dropout(h_v[items], p=self.dropout_rate)
            t_target = F.dropout(h_t[items], p=self.dropout_rate)

        # Online via predictor
        v_online = self.predictor(h_v[items])
        t_online = self.predictor(h_t[items])

        # Loss alignement modal 
        # "ce que BM3 voit de l'item" doit ressembler à "ce que TGNN voit de l'item"
        loss_vt = 1 - cosine_similarity(v_online, t_target.detach(), dim=-1).mean()
        loss_tv = 1 - cosine_similarity(t_online, v_target.detach(), dim=-1).mean()
        # Loss intra-modalité — robustesse (L_mask de BM3)
        loss_vv = 1 - cosine_similarity(v_online, v_target.detach(), dim=-1).mean()
        loss_tt = 1 - cosine_similarity(t_online, t_target.detach(), dim=-1).mean()

        return loss_vt + loss_tv + loss_vv + loss_tt 


    def calculate_loss_collab_modal(self, irf, h_modal_batch, temperature=0.5):
        """InfoNCE entre irf (collaboratif) et h_modal (modal) — aligne les deux espaces"""
        a = F.normalize(irf, dim=-1)
        b = F.normalize(h_modal_batch, dim=-1)
        logits = torch.matmul(a, b.T) / temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)
        return (loss_ab + loss_ba) / 2

    def calculate_loss_infonce(self, h_v, h_t, items, temperature=0.07):
        """InfoNCE loss — comme CLIP"""
        v = F.normalize(self.predictor(h_v[items]), dim=-1)
        t = F.normalize(self.predictor(h_t[items]), dim=-1)
        logits = torch.matmul(v, t.T) / temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_vt = F.cross_entropy(logits, labels)
        loss_tv = F.cross_entropy(logits.T, labels)
        return (loss_vt + loss_tv) / 2

def load_modal_features(data_path):
    """
    Charge image_feat.npy et text_feat.npy depuis le dossier BM3.
    Retourne deux numpy arrays.
    """
    v_feat = np.load(f'{data_path}/image_feat.npy')
    t_feat = np.load(f'{data_path}/text_feat.npy')
    print(f"[ModalEncoder] image_feat: {v_feat.shape}, text_feat: {t_feat.shape}")
    return v_feat, t_feat

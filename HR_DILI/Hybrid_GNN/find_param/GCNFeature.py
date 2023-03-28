# _*_ coding: utf-8 _*_
# @Time : $[DATE] $[TIME]
# @Author : G

import torch.nn as nn
from dgllife.model.gnn import GCN
import dgl
import torch

class GCNFeature(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, rdkit_feats=None, gnn_norm=None, activation=None, residual=None,
                 batchnorm=None, dropout=None, classifier_hidden_feats=128, n_task=1, predictor_hidden_feats=128):
        super(GCNFeature, self).__init__()

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       gnn_norm=gnn_norm,
                       activation=activation,
                       residual=residual,
                       batchnorm=batchnorm,
                       dropout=dropout)

        self.predict = nn.Sequential(
            nn.Linear(hidden_feats[0] + rdkit_feats, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, n_task)
        )

    def forward(self, bg, feats, rdkitEF):
        node_feats = self.gnn(bg, feats)
        with bg.local_scope():
            bg.ndata['hv'] = node_feats
            hg = dgl.max_nodes(bg, 'hv')
            _fgt = hg.detach().cpu().numpy()
            Concat_hg = torch.cat([hg, rdkitEF], dim=1)
            Final_feature = self.predict(Concat_hg)
            return Final_feature
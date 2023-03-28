# _*_ coding: utf-8 _*_
# @Time : $[DATE] $[TIME]
# @Author : G


import torch.nn as nn
from dgllife.model.model_zoo.mlp_predictor import MLPPredictor
from dgllife.model.gnn.gat import GAT
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
import dgl
import torch

class GATFeature(nn.Module):
    def __init__(self, in_feats, hidden_feats=None, rdkit_feats=None, num_heads=None, feat_drops=None,
                 attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None,
                 classifier_hidden_feats=128, classifier_dropout=0., n_task=1, dropout=0.,
                 predictor_hidden_feats=128, predictor_dropout=0.):
        super(GATFeature, self).__init__()

        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        self.gnn = GAT(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       num_heads=num_heads,
                       feat_drops=feat_drops,
                       attn_drops=attn_drops,
                       alphas=alphas,
                       residuals=residuals,
                       agg_modes=agg_modes,
                       activations=activations
                       )

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]

        self.predict = nn.Sequential(
            nn.Linear(hidden_feats[0] + rdkit_feats, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, n_task))

        # self.predict = nn.Sequential(
        #     # nn.LSTMCell(graph_feat_size+rdkitEF_size, 100),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(hidden_feats[0] + 2),
        #     nn.Linear(hidden_feats[0] + 2, 64),
        #     nn.LeakyReLU(),
        #     nn.LayerNorm(64),
        #     nn.Linear(64, n_task),
        # )


    def forward(self, bg, feats, rdkitEF):
        node_feats = self.gnn(bg, feats)
        with bg.local_scope():
            bg.ndata['hv'] = node_feats
            # Calculate graph representation by average readout.
            hg = dgl.max_nodes(bg, 'hv')
            _fgt = hg.detach().cpu().numpy()
            # Concat the graph feature and rdkit feature
            Concat_hg = torch.cat([hg, rdkitEF], dim=1)
            Final_feature = self.predict(Concat_hg)
            # print(Final_feature.ndim)
            return Final_feature
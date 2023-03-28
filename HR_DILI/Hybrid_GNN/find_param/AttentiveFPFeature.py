# _*_ coding: utf-8 _*_
# @Time : $[DATE] $[TIME]
# @Author : G

import torch
import torch.nn as nn
from dgllife.model.gnn import AttentiveFPGNN
from dgllife.model.readout import AttentiveFPReadout

class AttentiveFPFeature(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, rdkit_feats=None, num_layers=2, num_timesteps=2,
                 graph_feat_size=200, n_tasks=1, dropout=0.):
        super(AttentiveFPFeature, self).__init__()

        self.gnn = AttentiveFPGNN(node_feat_size=node_feat_size,
                                  edge_feat_size=edge_feat_size,
                                  num_layers=num_layers,
                                  graph_feat_size=graph_feat_size,
                                  dropout=dropout)
        self.readout = AttentiveFPReadout(feat_size=graph_feat_size,
                                          num_timesteps=num_timesteps,
                                          dropout=dropout)

        self.predict = nn.Sequential(
            nn.Linear(graph_feat_size + rdkit_feats, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, n_tasks))

    def forward(self, g, node_feats, edge_feats, rdkitEF, get_node_weight=False):
        node_feats = self.gnn(g, node_feats, edge_feats)
        if get_node_weight:
            g_feats, node_weights = self.readout(g, node_feats, get_node_weight)
            Concat_hg = torch.cat([g_feats, rdkitEF], dim=1)
            Final_feature = self.predict(Concat_hg)
            return Final_feature, node_weights

        else:
            g_feats = self.readout(g, node_feats, get_node_weight)
            Concat_hg = torch.cat([g_feats, rdkitEF], dim=1)
            Final_feature = self.predict(Concat_hg)
            return Final_feature

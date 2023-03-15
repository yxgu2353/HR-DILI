# _*_ coding: utf-8 _*_
# @Time : $[DATE] $[TIME]
# @Author : G

import torch
import dgl
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, AttentiveFPAtomFeaturizer, AttentiveFPBondFeaturizer
from AttentiveFPFeature import AttentiveFPFeature
from GATFeature import GATFeature
from GCNFeature import GCNFeature
from GraphSAGEFeature import GraphSAGEFeature
import pandas as pd
from QSAR_csv_dataset import MoleculeCSVDataset
from functools import partial
from dgllife.utils import smiles_to_bigraph, mol_to_bigraph
from torch.utils.data import DataLoader
import numpy as np
from dgllife.data import UnlabeledSMILES
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, confusion_matrix, f1_score, balanced_accuracy_score
import torch.nn.functional as F
import pickle
import itertools

# Evaluation Metrics
def accuracy(y_true, y_pred):
    return accuracy_score(y_true,y_pred)

def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label=1, average="binary")

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1, average="binary")

def auc(y_true, y_scores):
    return roc_auc_score(y_true, y_scores)

def mcc(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)

def new_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])

def sp(y_true, y_pred):
    cm = new_confusion_matrix(y_true, y_pred)
    return cm[0, 0] * 1.0 / (cm[0, 0] + cm[0, 1])

def BACC(y_true, y_pred):
    return balanced_accuracy_score(y_true, y_pred)

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)

# Device Choose
if torch.cuda.is_available():
    torch.device('cuda:0')
    device = 'cuda'
    print('='*30 + 'use GPU' + '='*30)

else:
    torch.device('cpu')
    device = 'cpu'
    print('='*30 + 'use CPU' + '='*30)

def collate_graphs(data):
    assert len(data[0]) in [4, 5], \
        'Expect the tuple to be of length 4 or 5, got {:d}'.format(len(data[0]))
    if len(data[0]) == 4:
        smiles, graphs, labels = map(list, zip(*data))
        masks = None
    else:
        smiles, graphs, features, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    features = torch.stack(features, dim=0)
    labels = torch.stack(labels, dim=0)

    if masks is None:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)
    return smiles, bg, features, labels, masks

def collate_graphs_unlabeled(data):
    smiles, graphs, features = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    features = torch.stack(features, dim=0)
    return smiles, bg, features

# -- data need to be tested --
Need_test = pd.read_csv('Seed2021_test/seed2021_test_data_descriptor.csv')
# unlabeled_test_data = pd.read_csv('unlabeled.csv')

# feature columns
feat_columns = Need_test.drop(['SMILES', 'label'], axis=1)
feat_size = feat_columns.columns.tolist()
feature_size = len(feat_size)


# ------ GCN -----
# Feature initialize --- No bond feature
node_feature = CanonicalAtomFeaturizer(atom_data_field='hv')
bond_feature = CanonicalBondFeaturizer(bond_data_field='he')
n_feats = node_feature.feat_size('hv')
e_feats = bond_feature.feat_size('he')

# Loading dataset
def load_GCN_data(data, path, feat_column) -> 'DGLGraph Form':
    dataset = MoleculeCSVDataset(df=data,
                                 smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                 node_featurizer=node_feature,
                                 edge_featurizer=None,
                                 smiles_column='SMILES',
                                 features_column=feat_column,
                                 task_names=['label'],
                                 cache_file_path=path + '.bin',
                                 n_jobs=-1,
                                 init_mask=True)
    return dataset

# GCN model structure
gcn_model = GCNFeature(in_feats=n_feats,
                       rdkit_feats=feature_size,
                       hidden_feats=[256],
                       activation=[F.relu])

# Loading GCN model
fn_gcn = 'GCN_model.pt'
gcn_model.load_state_dict(torch.load(fn_gcn, map_location=torch.device('cpu')))
gcn_net = gcn_model.to(device)

# Loading the dataset need to be tested, and transform it to DGL graph
# test data with label
extra_gcn_test = load_GCN_data(Need_test, 'gcn', feat_size)
extra_gcn_loader = DataLoader(extra_gcn_test, batch_size=1000, shuffle=True, collate_fn=collate_graphs)

# # Unlabeled test data
# unlabeled_gcn_test = UnlabeledSMILES(unlabeled_test_data['SMILES'], node_featurizer=node_feature, edge_featurizer=None, mol_to_graph=partial(mol_to_bigraph, add_self_loop=True))
# unlabeled_gcn_loader = DataLoader(unlabeled_gcn_test, batch_size=128, shuffle=True, collate_fn=collate_graphs_unlabeled)

# ------ GAT -----
# Feature initialize --- No bond feature
node_feature = CanonicalAtomFeaturizer(atom_data_field='hv')
bond_feature = CanonicalBondFeaturizer(bond_data_field='he')
n_feats = node_feature.feat_size('hv')
e_feats = bond_feature.feat_size('he')

# Loading dataset
def load_GAT_data(data, path, feat_column) -> 'DGLGraph Form':
    dataset = MoleculeCSVDataset(df=data,
                                 smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                 node_featurizer=node_feature,
                                 edge_featurizer=None,
                                 smiles_column='SMILES',
                                 features_column=feat_column,
                                 task_names=['label'],
                                 cache_file_path=path + '.bin',
                                 n_jobs=-1,
                                 init_mask=True)
    return dataset

# GAT model structure
gat_model = GATFeature(in_feats=n_feats,
                       rdkit_feats=feature_size,
                       hidden_feats=[128],
                       num_heads=[4],
                       alphas=[0.030304],
                       predictor_hidden_feats=256,
                       )


# Loading GAT model
fn_gat = 'GAT_model.pt'
gat_model.load_state_dict(torch.load(fn_gat, map_location=torch.device('cpu')))
gat_net = gat_model.to(device)

# Loading the dataset need to be tested, and transform it to DGL graph
# test data with label
extra_gat_test = load_GAT_data(Need_test, 'gat', feat_size)
extra_gat_loader = DataLoader(extra_gat_test, batch_size=1000, shuffle=True, collate_fn=collate_graphs)

# # Unlabeled test data
# unlabeled_gat_test = UnlabeledSMILES(unlabeled_test_data['SMILES'], node_featurizer=node_feature, edge_featurizer=bond_feature, mol_to_graph=partial(mol_to_bigraph))
# unlabeled_gat_loader = DataLoader(unlabeled_gat_test, batch_size=128, shuffle=True, collate_fn=collate_graphs_unlabeled)

# ------ AttentiveFP -----
# Feature initialize --- No bond feature
node_feature_1 = AttentiveFPAtomFeaturizer(atom_data_field='hv')
bond_feature_1 = AttentiveFPBondFeaturizer(bond_data_field='he')
n_feats_1 = node_feature_1.feat_size('hv')
e_feats_1 = bond_feature_1.feat_size('he')

# Loading dataset
def load_AttentiveFP_data(data, path, feat_column) -> 'DGLGraph Form':
    dataset = MoleculeCSVDataset(df=data,
                                 smiles_to_graph=partial(smiles_to_bigraph),
                                 node_featurizer=node_feature_1,
                                 edge_featurizer=bond_feature_1,
                                 smiles_column='SMILES',
                                 features_column=feat_column,
                                 task_names=['label'],
                                 cache_file_path=path + '.bin',
                                 n_jobs=-1,
                                 init_mask=True)
    return dataset

# AttentiveFP model structure
AttentiveFP_model = AttentiveFPFeature(node_feat_size=n_feats_1,
                                       edge_feat_size=e_feats_1,
                                       rdkit_feats=feature_size,
                                       num_layers=2,
                                       num_timesteps=2,
                                       graph_feat_size=128,
                                       n_tasks=1,
                                       dropout=0.)
# Loading AttentiveFP model
fn_attentiveFP = 'AttentiveFP_model.pt'
AttentiveFP_model.load_state_dict(torch.load(fn_attentiveFP, map_location=torch.device('cpu')))
AttentiveFP_net = AttentiveFP_model.to(device)

# Loading the dataset need to be tested, and transform it to DGL graph
# test data with label
extra_attentiveFP_test = load_AttentiveFP_data(Need_test, 'AttentiveFP', feat_size)
extra_attentiveFP_loader = DataLoader(extra_attentiveFP_test, batch_size=1000, shuffle=True, collate_fn=collate_graphs)

# # Unlabeled test data
# unlabeled_attentiveFP_test = UnlabeledSMILES(unlabeled_test_data['SMILES'], node_featurizer=node_feature_1, edge_featurizer=bond_feature_1, mol_to_graph=partial(mol_to_bigraph))
# unlabeled_attentiveFP_loader = DataLoader(unlabeled_attentiveFP_test, batch_size=128, shuffle=True, collate_fn=collate_graphs_unlabeled)

# ------ GraphSAGE -----
# Feature initialize --- No bond feature
node_feature = CanonicalAtomFeaturizer(atom_data_field='hv')
bond_feature = CanonicalBondFeaturizer(bond_data_field='he')
n_feats = node_feature.feat_size('hv')
e_feats = bond_feature.feat_size('he')

# Loading dataset
def load_GraphSAGE_data(data, path, feat_column) -> 'DGLGraph Form':
    dataset = MoleculeCSVDataset(df=data,
                                 smiles_to_graph=partial(smiles_to_bigraph),
                                 node_featurizer=node_feature,
                                 edge_featurizer=bond_feature,
                                 smiles_column='SMILES',
                                 features_column=feat_column,
                                 task_names=['label'],
                                 cache_file_path=path + '.bin',
                                 n_jobs=-1,
                                 init_mask=True)
    return dataset

# GraphSAGE model structure
GraphSAGE_model = GraphSAGEFeature(in_feats=n_feats,
                                   rdkit_feats=feature_size,
                                   hidden_feats=[256],
                                   activation=[F.relu],
                                   aggregator_type=['mean'])

# Loading GraphSAGE model
fn_GraphSAGE = 'GraphSAGE_model.pt'
GraphSAGE_model.load_state_dict(torch.load(fn_GraphSAGE, map_location=torch.device('cpu')))
GraphSAGE_net = GraphSAGE_model.to(device)

# Loading the dataset need to be tested, and transform it to DGL graph
# test data with label
extra_GraphSAGE_test = load_GraphSAGE_data(Need_test, 'GraphSAGE', feat_size)
extra_GraphSAGE_loader = DataLoader(extra_GraphSAGE_test, batch_size=1000, shuffle=True, collate_fn=collate_graphs)
#
# # Unlabeled test data
# unlabeled_GraphSAGE_test = UnlabeledSMILES(unlabeled_test_data['SMILES'], node_featurizer=node_feature, edge_featurizer=bond_feature, mol_to_graph=partial(mol_to_bigraph))
# unlabeled_GraphSAGE_loader = DataLoader(unlabeled_GraphSAGE_test, batch_size=128, shuffle=True, collate_fn=collate_graphs_unlabeled)



# ------ Eval Setting ------
# test data with label
# ---> GCN model
for batch_id, batch_data in enumerate(extra_gcn_loader):
    test_smiles, test_bg, test_features, test_labels, test_mask = batch_data
    test_bg = test_bg.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    test_masks = test_mask.to(device)
    test_n_feats = test_bg.ndata.pop('hv').to(device)
    test_logits_GCN = gcn_net(test_bg, test_n_feats, test_features)

prob_GCN = torch.sigmoid(test_logits_GCN).detach().cpu().numpy().flatten().tolist()
pred_y_GCN = []
for i in prob_GCN:
    if i > 0.5:
        i = 1
        pred_y_GCN.append(i)
    if i <= 0.5:
        i = 0
        pred_y_GCN.append(i)
GCN_prediction = np.array(pred_y_GCN).reshape(-1, 1)
GCN_probs = np.array(prob_GCN).reshape(-1, 1)
Y_label = np.array(test_labels)
Num_test = len(Y_label)
print('-'*30 + 'hybrid_GCN' + '-'*30)
print('ACC: {:.4f}'.format(accuracy(Y_label, GCN_prediction)))
print('Precision: {:.4f}'.format(precision(Y_label, GCN_prediction)))
print('Recall: {:.4f}'.format(recall(Y_label, GCN_prediction)))
print('MCC: {:.4f}'.format(accuracy(Y_label, GCN_prediction)))
print('SP: {:.4f}'.format(sp(Y_label, GCN_prediction)))
print('BACC: {:.4f}'.format(BACC(Y_label, GCN_prediction)))
print('F1: {:.4f}'.format(f1(Y_label, GCN_prediction)))
print('AUC: {:.4f}'.format(auc(Y_label, GCN_probs)))


# ---> GAT model
for batch_id, batch_data in enumerate(extra_gat_loader):
    test_smiles, test_bg, test_features, test_labels, test_mask = batch_data
    test_bg = test_bg.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    test_masks = test_mask.to(device)
    test_n_feats = test_bg.ndata.pop('hv').to(device)
    test_logits_GAT = gcn_net(test_bg, test_n_feats, test_features)

prob_GAT = torch.sigmoid(test_logits_GAT).detach().cpu().numpy().flatten().tolist()
pred_y_GAT = []
for i in prob_GAT:
    if i > 0.5:
        i = 1
        pred_y_GAT.append(i)
    if i <= 0.5:
        i = 0
        pred_y_GAT.append(i)
GAT_prediction = np.array(pred_y_GAT).reshape(-1, 1)
GAT_probs = np.array(prob_GAT).reshape(-1, 1)
print('-'*30 + 'hybrid_GAT' + '-'*30)
print('ACC: {:.4f}'.format(accuracy(Y_label, GAT_prediction)))
print('Precision: {:.4f}'.format(precision(Y_label, GAT_prediction)))
print('Recall: {:.4f}'.format(recall(Y_label, GAT_prediction)))
print('MCC: {:.4f}'.format(accuracy(Y_label, GAT_prediction)))
print('SP: {:.4f}'.format(sp(Y_label, GAT_prediction)))
print('BACC: {:.4f}'.format(BACC(Y_label, GAT_prediction)))
print('F1: {:.4f}'.format(f1(Y_label, GAT_prediction)))
print('AUC: {:.4f}'.format(auc(Y_label, GAT_probs)))

# ---> AttentiveFP model
for batch_id, batch_data in enumerate(extra_attentiveFP_loader):
    test_smiles, test_bg, test_features, test_labels, test_mask = batch_data
    test_bg = test_bg.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    test_masks = test_mask.to(device)
    test_n_feats = test_bg.ndata.pop('hv').to(device)
    test_e_feats = test_bg.edata.pop('he').to(device)
    test_logits_attentiveFP = AttentiveFP_net(test_bg, test_n_feats, test_e_feats, test_features)

prob_attentiveFP = torch.sigmoid(test_logits_attentiveFP).detach().cpu().numpy().flatten().tolist()
pred_y_attentiveFP = []
for i in prob_attentiveFP:
    if i > 0.5:
        i = 1
        pred_y_attentiveFP.append(i)
    if i <= 0.5:
        i = 0
        pred_y_attentiveFP.append(i)
attentiveFP_prediction = np.array(pred_y_attentiveFP).reshape(-1, 1)
attentiveFP_probs = np.array(prob_attentiveFP).reshape(-1, 1)
print('-'*30 + 'hybrid_AttentiveFP' + '-'*30)
print('ACC: {:.4f}'.format(accuracy(Y_label, attentiveFP_prediction)))
print('Precision: {:.4f}'.format(precision(Y_label, attentiveFP_prediction)))
print('Recall: {:.4f}'.format(recall(Y_label, attentiveFP_prediction)))
print('MCC: {:.4f}'.format(accuracy(Y_label, attentiveFP_prediction)))
print('SP: {:.4f}'.format(sp(Y_label, attentiveFP_prediction)))
print('BACC: {:.4f}'.format(BACC(Y_label, attentiveFP_prediction)))
print('F1: {:.4f}'.format(f1(Y_label, attentiveFP_prediction)))
print('AUC: {:.4f}'.format(auc(Y_label, attentiveFP_probs)))

# ---> GraphSAGE model
for batch_id, batch_data in enumerate(extra_GraphSAGE_loader):
    test_smiles, test_bg, test_features, test_labels, test_mask = batch_data
    test_bg = test_bg.to(device)
    test_features = test_features.to(device)
    test_labels = test_labels.to(device)
    test_masks = test_mask.to(device)
    test_n_feats = test_bg.ndata.pop('hv').to(device)
    test_logits_GraphSAGE = GraphSAGE_net(test_bg, test_n_feats, test_features)

prob_GraphSAGE = torch.sigmoid(test_logits_GraphSAGE).detach().cpu().numpy().flatten().tolist()
pred_y_GraphSAGE= []
for i in prob_GraphSAGE:
    if i > 0.5:
        i = 1
        pred_y_GraphSAGE.append(i)
    if i <= 0.5:
        i = 0
        pred_y_GraphSAGE.append(i)
GraphSAGE_prediction = np.array(pred_y_GraphSAGE).reshape(-1, 1)
GraphSAGE_probs = np.array(prob_GraphSAGE).reshape(-1, 1)
print('-'*30 + 'hybrid_GraphSAGE' + '-'*30)
print('ACC: {:.4f}'.format(accuracy(Y_label, GraphSAGE_prediction)))
print('Precision: {:.4f}'.format(precision(Y_label, GraphSAGE_prediction)))
print('Recall: {:.4f}'.format(recall(Y_label, GraphSAGE_prediction)))
print('MCC: {:.4f}'.format(accuracy(Y_label, GraphSAGE_prediction)))
print('SP: {:.4f}'.format(sp(Y_label, GraphSAGE_prediction)))
print('BACC: {:.4f}'.format(BACC(Y_label, GraphSAGE_prediction)))
print('F1: {:.4f}'.format(f1(Y_label, GraphSAGE_prediction)))
print('AUC: {:.4f}'.format(auc(Y_label, GraphSAGE_probs)))

# ---> Traditional ML Model
# MACCS data
test_MACCS_data = pd.read_csv('Seed2021_test/seed2021_test_data_MACCS.csv', header=None).values
x_test_MACCS = test_MACCS_data[:, 1:]
y_test_MACCS = test_MACCS_data[:, 0]
Num_test_1 = len(y_test_MACCS)

# ML_model_one
model_load_1 = open('Hepa_rf_MACCS.model', 'rb')
model_1 = pickle.load(model_load_1)
model_load_1.close()
y_pred_1 = model_1.predict(x_test_MACCS)
y_pred_score_1 = model_1.predict_proba(x_test_MACCS)[:, 1]
print('-'*30 + 'rf_MACCS' + '-'*30)
print('ACC: {:.4f}'.format(accuracy(y_test_MACCS, y_pred_1)))
print('Precision: {:.4f}'.format(precision(y_test_MACCS, y_pred_1)))
print('Recall: {:.4f}'.format(recall(y_test_MACCS, y_pred_1)))
print('MCC: {:.4f}'.format(accuracy(y_test_MACCS, y_pred_1)))
print('SP: {:.4f}'.format(sp(y_test_MACCS, y_pred_1)))
print('BACC: {:.4f}'.format(BACC(y_test_MACCS, y_pred_1)))
print('F1: {:.4f}'.format(f1(y_test_MACCS, y_pred_1)))
print('AUC: {:.4f}'.format(auc(y_test_MACCS, y_pred_score_1)))

# Morgan data
test_Morgan_data = pd.read_csv('Seed2021_test/seed2021_test_data_Morgan.csv', header=None).values
x_test_Morgan = test_Morgan_data[:, 1:]
y_test_Morgan = test_Morgan_data[:, 0]
Num_test_2 = len(y_test_Morgan)

# model_two
model_load_2 = open('Hepa_rf_Morgan.model', 'rb')
model_2 = pickle.load(model_load_2)
model_load_2.close()
y_pred_2 = model_2.predict(x_test_Morgan)
y_pred_score_2 = model_2.predict_proba(x_test_Morgan)[:, 1]
print('-'*30 + 'rf_Morgan' + '-'*30)
print('ACC: {:.4f}'.format(accuracy(y_test_Morgan, y_pred_2)))
print('Precision: {:.4f}'.format(precision(y_test_Morgan, y_pred_2)))
print('Recall: {:.4f}'.format(recall(y_test_Morgan, y_pred_2)))
print('MCC: {:.4f}'.format(accuracy(y_test_Morgan, y_pred_2)))
print('SP: {:.4f}'.format(sp(y_test_Morgan, y_pred_2)))
print('BACC: {:.4f}'.format(BACC(y_test_Morgan, y_pred_2)))
print('F1: {:.4f}'.format(f1(y_test_Morgan, y_pred_2)))
print('AUC: {:.4f}'.format(auc(y_test_Morgan, y_pred_score_2)))

# ---> Gather these Models' prediction & prob
SMILES = pd.DataFrame(test_smiles, columns=['SMILES'])
Labels = pd.DataFrame(np.array(test_labels), columns=['label'])
# GCN
Label_GCN_pred = pd.DataFrame(GCN_prediction, columns=['label_GCN_pred'])
Prediction_GCN_prob = pd.DataFrame(GCN_probs, columns=['Prob_GCN'])
# GAT
Label_GAT_pred = pd.DataFrame(GAT_prediction, columns=['label_GAT_pred'])
Prediction_GAT_prob = pd.DataFrame(GAT_probs, columns=['Prob_GAT'])
# AttentiveFP
Label_AttentiveFP_pred = pd.DataFrame(attentiveFP_prediction, columns=['label_AttentiveFP_pred'])
Prediction_AttentiveFP_prob = pd.DataFrame(attentiveFP_probs, columns=['Prob_AttentiveFP'])
# GraphSAGE
Label_GraphSAGE_pred = pd.DataFrame(GraphSAGE_prediction, columns=['label_GraphSAGE_pred'])
Prediction_GraphSAGE_prob = pd.DataFrame(GraphSAGE_probs, columns=['Prob_GraphSAGE'])
# RF_MACCS
Label_RF_MACCS_pred = pd.DataFrame(y_pred_2, columns=['label_RF_MACCS_pred'])
Prediction_RF_MACCS_prob = pd.DataFrame(y_pred_score_2, columns=['Prob_RF_MACCS'])
# RF_Morgan
Label_RF_Morgan_pred = pd.DataFrame(y_pred_1, columns=['label_RF_Morgan_pred'])
Prediction_RF_Morgan_prob = pd.DataFrame(y_pred_score_1, columns=['Prob_RF_Morgan'])

Final_results = pd.concat([SMILES, Labels, Label_GCN_pred, Prediction_GCN_prob, Label_GAT_pred, Prediction_GAT_prob,
                           Label_AttentiveFP_pred, Prediction_AttentiveFP_prob, Label_GraphSAGE_pred, Prediction_GraphSAGE_prob,
                           Label_RF_MACCS_pred, Prediction_RF_MACCS_prob, Label_RF_Morgan_pred, Prediction_RF_Morgan_prob], axis=1)
Final_results.to_csv('Consensus_test.csv', index=None)
print('== '*10 + ' Predict extra-validation File Finished ' + ' =='*10)


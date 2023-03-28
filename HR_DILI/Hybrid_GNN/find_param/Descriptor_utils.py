# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import errno
import json
import os
import torch
import torch.nn.functional as F
from dgllife.utils import smiles_to_bigraph, ScaffoldSplitter, RandomSplitter, mol_to_bigraph
from functools import partial

def init_featurizer(args):
    """Initialize node/edge featurizer
    Parameters
    ----------
    args : dict
        Settings
    Returns
    -------
    args : dict
        Settings with featurizers updated
    """
    if args['model'] in ['gin_supervised_contextpred', 'gin_supervised_infomax',
                         'gin_supervised_edgepred', 'gin_supervised_masking']:
        from dgllife.utils import PretrainAtomFeaturizer, PretrainBondFeaturizer
        args['atom_featurizer_type'] = 'pre_train'
        args['bond_featurizer_type'] = 'pre_train'
        args['node_featurizer'] = PretrainAtomFeaturizer()
        args['edge_featurizer'] = PretrainBondFeaturizer()
        return args

    if args['atom_featurizer_type'] == 'canonical':
        from dgllife.utils import CanonicalAtomFeaturizer
        args['node_featurizer'] = CanonicalAtomFeaturizer()
    elif args['atom_featurizer_type'] == 'attentivefp':
        from dgllife.utils import AttentiveFPAtomFeaturizer
        args['node_featurizer'] = AttentiveFPAtomFeaturizer()
    elif args['atom_featurizer_type'] == 'weave':
        from dgllife.utils import WeaveAtomFeaturizer
        args['node_featurizer'] = WeaveAtomFeaturizer()
    else:
        return ValueError(
            "Expect node_featurizer to be in ['canonical', 'attentivefp','Weavefp'], "
            "got {}".format(args['atom_featurizer_type']))

    if args['model'] in ['Weave', 'MPNN', 'AttentiveFP']:
        if args['bond_featurizer_type'] == 'canonical':
            from dgllife.utils import CanonicalBondFeaturizer
            args['edge_featurizer'] = CanonicalBondFeaturizer(self_loop=True)
        elif args['bond_featurizer_type'] == 'attentivefp':
            from dgllife.utils import AttentiveFPBondFeaturizer
            args['edge_featurizer'] = AttentiveFPBondFeaturizer(self_loop=True)
        elif args['bond_featurizer_type'] == 'weave':
            from dgllife.utils import WeaveEdgeFeaturizer
            args['edge_featurizer'] = WeaveEdgeFeaturizer()
    else:
        args['edge_featurizer'] = None

    return args


from QSAR_csv_dataset import MoleculeCSVDataset
def load_dataset(args, df):
    dc_listings = df.drop(['SMILES', 'label'], axis=1)
    features = dc_listings.columns.tolist()
    args['rdkitEF_s'] = len(features)
    dataset = MoleculeCSVDataset(df=df,
                                 features_column=features,
                                 smiles_to_graph=partial(smiles_to_bigraph, add_self_loop=True),
                                 node_featurizer=args['node_featurizer'],
                                 edge_featurizer=args['edge_featurizer'],
                                 smiles_column=args['smiles_column'],
                                 cache_file_path=args['result_path'] + '/graph.bin',
                                 task_names=['label'],
                                 load=False, init_mask=True, n_jobs=1)

    return dataset

def get_configure(model):
    """Query for the manually specified configuration
    Parameters
    ----------
    model : str
        Model type
    Returns
    -------
    dict
        Returns the manually specified configuration
    """
    with open('model_configures/{}.json'.format(model), 'r') as f:
        config = json.load(f)
    return config

def mkdir_p(path):
    """Create a folder for the given path.
    Parameters
    ----------
    path: str
        Folder to create
    """
    try:
        os.makedirs(path)
        print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            print('Directory {} already exists.'.format(path))
        else:
            raise

def init_trial_path(args):
    """Initialize the path for a hyperparameter setting
    Parameters
    ----------
    args : dict
        Settings
    Returns
    -------
    args : dict
        Settings with the trial path updated
    """
    trial_id = 0
    path_exists = True
    while path_exists:
        trial_id += 1
        path_to_results = args['result_path'] + '/{:d}'.format(trial_id)
        path_exists = os.path.exists(path_to_results)
    args['trial_path'] = path_to_results
    mkdir_p(args['trial_path'])

    return args

def split_dataset(args, dataset):
    train_ratio, val_ratio, test_ratio = map(float, args['split_ratio'].split(','))
    if args['split'] == 'scaffold_decompose':
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
            scaffold_func='decompose')
    elif args['split'] == 'scaffold_smiles':
        train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio,
            scaffold_func='smiles')
    elif args['split'] == 'random':
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=train_ratio, frac_val=val_ratio, frac_test=test_ratio)
    else:
        return ValueError("Expect the splitting method to be 'scaffold', got {}".format(args['split']))

    return train_set, val_set, test_set

def collate_molgraphs(data):
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

def collate_molgraphs_unlabeled(data):
    """Batching a list of datapoints without labels
    Parameters
    ----------
    data : list of 2-tuples.
        Each tuple is for a single datapoint, consisting of
        a SMILES and a DGLGraph.
    Returns
    -------
    smiles : list
        List of smiles
    bg : DGLGraph
        The batched DGLGraph.
    """
    smiles, graphs = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)

    return smiles, bg

def load_model(exp_configure):
    if exp_configure['model'] == 'GCN':
        from GCNFeature import GCNFeature
        model = GCNFeature(
            in_feats=exp_configure['in_node_feats'],
            rdkit_feats=exp_configure['rdkitEF_s'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            activation=[F.relu] * exp_configure['num_gnn_layers'],
            residual=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            batchnorm=[exp_configure['batchnorm']] * exp_configure['num_gnn_layers'],
            dropout=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            n_task=exp_configure['n_tasks'])

    elif exp_configure['model'] == 'GAT':
        from GATFeature import GATFeature
        model = GATFeature(
            in_feats=exp_configure['in_node_feats'],
            rdkit_feats=exp_configure['rdkitEF_s'],
            hidden_feats=[exp_configure['gnn_hidden_feats']] * exp_configure['num_gnn_layers'],
            num_heads=[exp_configure['num_heads']] * exp_configure['num_gnn_layers'],
            feat_drops=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            attn_drops=[exp_configure['dropout']] * exp_configure['num_gnn_layers'],
            alphas=[exp_configure['alpha']] * exp_configure['num_gnn_layers'],
            residuals=[exp_configure['residual']] * exp_configure['num_gnn_layers'],
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            predictor_dropout=exp_configure['dropout'],
            n_task=exp_configure['n_tasks']
        )
    elif exp_configure['model'] == 'GraphSAGE':
        # from GraphSAGEPredictor import GraphSAGEPredictor
        from GraphSAGEFeature import GraphSAGEFeature
        model = GraphSAGEFeature(
            in_feats=exp_configure['in_node_feats'],
            rdkit_feats=exp_configure['rdkitEF_s'],
            hidden_feats=[exp_configure['gnn_hidden_feats']]*2,
            dropout=[exp_configure['dropout']] * 2,
            activation=[F.relu] * 2,
            aggregator_type=[exp_configure['aggregator_type']]*2,
            predictor_hidden_feats=exp_configure['predictor_hidden_feats'],
            n_task=exp_configure['n_tasks']
        )


    elif exp_configure['model'] == 'AttentiveFP':
        from AttentiveFPFeature import AttentiveFPFeature
        model = AttentiveFPFeature(
            node_feat_size=exp_configure['in_node_feats'],
            edge_feat_size=exp_configure['in_edge_feats'],
            rdkit_feats=exp_configure['rdkitEF_s'],
            num_layers=exp_configure['num_layers'],
            num_timesteps=exp_configure['num_timesteps'],
            graph_feat_size=exp_configure['graph_feat_size'],
            dropout=exp_configure['dropout'],
            n_tasks=exp_configure['n_tasks']
        )

    else:
        return ValueError("Expect model to be from ['GCN', 'GAT', 'AttentiveFP', 'GraphSAGE'], "
                          "got {}".format(exp_configure['model']))

    return model

def predict(args, model, bg, features):
    if args['edge_featurizer'] is None:
        bg = bg.to(args['device'])
        features=features.to(args['device'])
        node_feats = bg.ndata.pop('h').to(args['device'])
        return model(bg, node_feats, features)

    else:
        bg = bg.to(args['device'])
        features=features.to(args['device'])
        node_feats = bg.ndata.pop('h').to(args['device'])
        edge_feats = bg.edata.pop('e').to(args['device'])
        return model(bg, node_feats, edge_feats, features)
    
    # bg = bg.to(args['device'])
    # features=features.to(args['device'])
    # node_feats = bg.ndata.pop('h').to(args['device'])
    # return model(bg, node_feats, features)

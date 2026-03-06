import argparse
import numpy as np
import scipy.sparse as sp
import torch
import random
import networkx as nx
import dgl
from dgl import DGLGraph
from dgl.data import *

import os
import scipy
from sklearn.preprocessing import label_binarize
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from dgl import function as fn
import torch.nn as nn
import torch.nn.functional as F
import time
# from google_drive_downloader import GoogleDriveDownloader as gdd

from ogb.nodeproppred import Evaluator

from ogb.nodeproppred import DglNodePropPredDataset

def entropy(array):
    array = array + 1e-9
    entropy = -torch.sum(array * torch.exp(array))
    return entropy

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def save(save_path, dataname, graphs, labels, feats):
    # save graphs and labels
    graph_path = os.path.join(save_path, dataname + '_dgl_graph.bin')
    save_graphs(graph_path, graphs, {'labels': labels, 'feats': feats})

def load(save_path, dataname):
    # load processed data from directory `self.save_path`
    graph_path = os.path.join(save_path, dataname + '_dgl_graph.bin')
    graphs, label_dict = load_graphs(graph_path)
    labels = label_dict['labels']
    feats = label_dict['feats']
    
    return graphs, labels, feats

def random_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0, seed=0):
    index = [i for i in range(0, data['label'].shape[0])]
    trn_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data['label'].cpu() == c)[0]
        if len(class_idx)<percls_trn:
            trn_idx.extend(class_idx)
        else:
            trn_idx.extend(rnd_state.choice(class_idx, percls_trn,replace=False))
    rest_idx = [i for i in index if i not in trn_idx]
    val_idx = rnd_state.choice(rest_idx,val_lb,replace=False)
    tst_idx = [i for i in rest_idx if i not in val_idx]
    
    trn_idx = torch.LongTensor(trn_idx)
    val_idx = torch.LongTensor(val_idx)
    tst_idx = torch.LongTensor(tst_idx)
    return trn_idx, val_idx, tst_idx


def preprocess_data_3(dataset, train_ratio, val_ratio, seed):
    
    if dataset in ['film', 'cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel', 'new_chameleon', 'new_squirrel']:
        file_path = '../high_freq/{}'.format(dataset)
        graph, labels, feats = load(file_path, dataset)
        nclass = len(set(labels.cpu().numpy()))
        g = graph[0]
        data = g.ndata
        data['label'] = labels
        data['feat'] = feats
        start = time.time()
        
        percls_trn = int(round(train_ratio*len(data['label'])/nclass))
        val_lb = int(round(val_ratio*len(data['label'])))
        
        trn_idx, val_idx, tst_idx = random_splits(data, nclass, percls_trn, val_lb, seed = seed)
        #print('--------------------------------- data preprocessing: ',time.time()-start,'------------------------------------------')
        return g, nclass, feats, labels, trn_idx, val_idx, tst_idx
    
    elif dataset in ['computers', 'photo', 'cs', 'physics']:
        if dataset == 'computers':
            dataset = dgl.data.AmazonCoBuyComputerDataset()
        elif dataset == 'photo':
            dataset = dgl.data.AmazonCoBuyPhotoDataset()
        elif dataset == 'cs':
            dataset = dgl.data.CoauthorCSDataset()
        elif dataset == 'physics':
            dataset = dgl.data.CoauthorPhysicsDataset()
            
        g = dataset[0]
        feat = normalize_features(g.ndata['feat'])
        labels = g.ndata['label']
        feat = torch.FloatTensor(feat)
        labels = torch.LongTensor(labels)
        g = dgl.to_simple(g)
        g = dgl.to_bidirected(g)
        g = dgl.add_self_loop(g)
        nclass = len(set(labels.tolist()))
        
        g.ndata['label'] = labels
        g.ndata['feat'] = feat
        data = g.ndata
        percls_trn = int(round(train_ratio*len(labels)/nclass))
        val_lb = int(round(val_ratio*len(labels)))
        train, val, test = random_splits(data, nclass, percls_trn, val_lb, seed = seed)
        
        return g, nclass, feat, labels, train, val, test
    
    elif dataset in ['arxiv', 'products']:
        if dataset == 'arxiv':
            data = DglNodePropPredDataset(name = 'ogbn-arxiv', root = '../../../OGB')
        elif dataset == 'products':
            data = DglNodePropPredDataset(name = 'ogbn-products', root = '../../../OGB')
        
        splitted_idx = data.get_idx_split()
        idx_train, idx_val, idx_test = (
            splitted_idx["train"],
            splitted_idx["valid"],
            splitted_idx["test"],
        )
        
        g, labels = data[0]
        labels = labels.squeeze()
        nclass = len(set(labels.tolist()))
        
        g.ndata['label'] = labels
        feat = g.ndata['feat']
        
        if dataset == "arxiv":
            srcs, dsts = g.all_edges()
            g.add_edges(dsts, srcs)
        g = g.remove_self_loop().add_self_loop()
        
        return g, nclass, feat, labels, idx_train, idx_val, idx_test
        
    if dataset in ['cora', 'citeseer', 'pubmed']:

        edge = np.loadtxt('../low_freq/{}.edge'.format(dataset), dtype=int).tolist()
        feat = np.loadtxt('../low_freq/{}.feature'.format(dataset))
        labels = np.loadtxt('../low_freq/{}.label'.format(dataset), dtype=int)
        train = np.loadtxt('../low_freq/{}.train'.format(dataset), dtype=int)
        val = np.loadtxt('../low_freq/{}.val'.format(dataset), dtype=int)
        test = np.loadtxt('../low_freq/{}.test'.format(dataset), dtype=int)
        nclass = len(set(labels.tolist()))
        #print(dataset, nclass)

        U = [e[0] for e in edge]
        V = [e[1] for e in edge]
        g = dgl.graph((U, V))
        g = dgl.to_simple(g)
        g = dgl.remove_self_loop(g)
        
        if dataset == 'citeseer':
            g = dgl.add_self_loop(g)
        g = dgl.to_bidirected(g)

        feat = normalize_features(feat)
        feat = torch.FloatTensor(feat)
        labels = torch.LongTensor(labels)
        nclass = len(set(labels.cpu().numpy()))
#         trn_idx = torch.LongTensor(train)
#         val_idx = torch.LongTensor(val)
#         tst_idx = torch.LongTensor(test)
        
        g.ndata['label'] = labels
        g.ndata['feat'] = feat
        data = g.ndata
        
        percls_trn = int(round(train_ratio*len(data['label'])/nclass))
        val_lb = int(round(val_ratio*len(data['label'])))
        trn_idx, val_idx, tst_idx = random_splits(data, nclass, percls_trn, val_lb, seed = seed)   

        return g, nclass, feat, labels, trn_idx, val_idx, tst_idx
    
    if dataset in ['pokec', 'penn94']:
        g, labels, train, val, test = load_nonhom_data(dataset, '../high_freq/', split_idx = seed)
        nclass = labels.max()+1
        feat = g.ndata['feat']
        return g, nclass, feat, labels, train, val, test


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def norm_degree_matrix(g):
    deg = g.in_degrees().float().clamp(min=1)
    return torch.pow(deg, -0.5)


""" For NonHom"""
dataset_drive_url = {"pokec": "1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y"}
splits_drive_url = {"pokec": "1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_"}

def load_nonhom_data(dataset, dataset_path, split_idx):
    data_path = os.path.join(dataset_path, f"{dataset}.mat")
    data_split_path = os.path.join(
        dataset_path, "splits", f"{dataset}-splits.npy"
    )

    if dataset == "pokec":
        g, features, labels = load_pokec_mat(data_path)
    elif dataset == "penn94":
        g, features, labels = load_penn94_mat(data_path)
    else:
        raise ValueError("Invalid dataname")

    g = g.remove_self_loop().add_self_loop()
    g.ndata["feat"] = features
    labels = torch.LongTensor(labels)

    splitted_idx = load_fixed_splits(dataset, data_split_path, split_idx)
    idx_train, idx_val, idx_test = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    return g, labels, idx_train, idx_val, idx_test

def load_penn94_mat(data_path):
    mat = scipy.io.loadmat(data_path)
    A = mat["A"]
    metadata = mat["local_info"]

    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    metadata = metadata.astype(np.int)

    # make features into one-hot encodings
    feature_vals = np.hstack((np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    g = dgl.graph((edge_index[0], edge_index[1]))
    g = dgl.to_bidirected(g)

    features = torch.tensor(features, dtype=torch.float)
    labels = torch.tensor(metadata[:, 1] - 1)  # gender label, -1 means unlabeled
    return g, features, labels


def load_pokec_mat(data_path):
    if not os.path.exists(data_path):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url["pokec"], dest_path=data_path, showsize=True
        )

    fulldata = scipy.io.loadmat(data_path)
    edge_index = torch.tensor(fulldata["edge_index"], dtype=torch.long)
    g = dgl.graph((edge_index[0], edge_index[1]))
    g = dgl.to_bidirected(g)

    features = torch.tensor(fulldata["node_feat"]).float()
    labels = fulldata["label"].flatten()
    return g, features, labels

def load_fixed_splits(dataset, data_split_path="", split_idx=0):
    if not os.path.exists(data_split_path):
        assert dataset in splits_drive_url.keys()
        gdd.download_file_from_google_drive(
            file_id=splits_drive_url[dataset], dest_path=data_split_path, showsize=True
        )

    splits_lst = np.load(data_split_path, allow_pickle=True)
    splits = splits_lst[split_idx]

    for key in splits:
        if not torch.is_tensor(splits[key]):
            splits[key] = torch.as_tensor(splits[key])

    return splits

def edge_applying(edges):
    h_i = edges.dst['d'].unsqueeze(-1) * edges.dst['h']
    h_j = edges.src['d'].unsqueeze(-1) * edges.src['h']
    diff = torch.pow(h_i-h_j, 2).sum(-1)
    return {'e': diff}

def smoothness(g, h):
    g.ndata['h'] = h
    g.apply_edges(edge_applying)
    g.update_all(fn.copy_e('e', 'm'), fn.sum('m', 'z'))
    return g.ndata['z'].sum() * 0.5 / g.ndata['z'].shape[0]

def oh_encoding_logit(logp):
    _, indices = torch.max(logp, dim=1)
    oh_logp = F.one_hot(indices)
    return oh_logp

class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))
import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
import sys
import argparse
import logging
import random
import datetime
import torch
import torch.nn.functional as F
from torch.optim import Adam
import models
import train
import data
from scipy.special import logsumexp


def calc_log_h_prime(
    data_vectors,
    C_1,  # 1
    C_2,  # 4/9*\sqrt(3)
    lambda_max,
    M,
    n_dim_e,
    n_dim_m,
    n_dim_d,
):
    x_norms = np.linalg.norm(data_vectors, axis=1)
    x_mat = x_norms.reshape((-1, 1)).dot(x_norms.reshape((1, -1)))
    x_mat_ = np.triu(x_mat, k=1).flatten()
    x_mat_ = x_mat_[np.where(x_mat_ != 0)[0]]
    log_x_i_x_j = logsumexp(np.log(x_mat_))
    log_x_i_x_j_2 = logsumexp(np.log(x_mat_**2))

    x_sum_mat = x_mat * (x_norms[:, np.newaxis] + x_norms[np.newaxis, :])
    x_sum_mat_ = np.triu(x_sum_mat, k=1).flatten()
    x_sum_mat_ = x_sum_mat_[np.where(x_sum_mat_ != 0)[0]]

    log_x_i_x_j_3 = logsumexp(np.log(x_sum_mat_))

    def calc_log_h_A():
        first_term = np.log(2) + 2 * np.log(C_1) + 2 * np.log(M) + 0.5 * np.log(n_dim_m * n_dim_d) + log_x_i_x_j + np.log(
            lambda_max * ((n_dim_m * n_dim_d)**0.5) + M * n_dim_m * ((n_dim_e * n_dim_d)**0.5) + 2 * lambda_max * ((n_dim_e * n_dim_m)**0.5))
        second_term = np.log(2) + 4 * np.log(C_1) + 5 * np.log(M) + np.log(lambda_max) + 0.5 * np.log(n_dim_e) + 1.5 * np.log(n_dim_m) + np.log(n_dim_d) + \
            log_x_i_x_j_2 + np.log(lambda_max * ((n_dim_m * n_dim_d)**0.5) + M * n_dim_m * (
                (n_dim_e * n_dim_d)**0.5) + 2 * lambda_max * ((n_dim_e * n_dim_m)**0.5))

        return logsumexp([first_term, second_term])

    def calc_log_h_B():
        first_term = np.log(2) + 2 * np.log(C_1) + 2 * np.log(M) + 0.5 * np.log(n_dim_e * n_dim_m) + log_x_i_x_j + np.log(lambda_max * ((n_dim_m * n_dim_d)**0.5) + M * n_dim_m * (
            (n_dim_e * n_dim_d)**0.5) + C_1 * lambda_max * ((n_dim_e * n_dim_m)**0.5))
        second_term = np.log(C_1 * C_2) + 3 * np.log(M) + np.log(lambda_max) + np.log(
            n_dim_e * n_dim_m) + 0.5 * np.log(n_dim_m * n_dim_d) + log_x_i_x_j_3
        third_term = np.log(2) + 4 * np.log(C_1) + 6 * np.log(M) + 1.5 * np.log(n_dim_e * n_dim_m) + np.log(n_dim_m * n_dim_d) + log_x_i_x_j_2 + np.log(lambda_max * ((n_dim_m * n_dim_d)**0.5) + M * n_dim_m * (
            (n_dim_e * n_dim_d)**0.5) + 2 * lambda_max * ((n_dim_e * n_dim_m)**0.5))
        return logsumexp([first_term, second_term, third_term])

    def calc_log_h_Lambda():
        first_term = np.log(2) + 2 * np.log(C_1) + 3 * np.log(M) + 0.5 * \
            np.log(n_dim_e * n_dim_m) + np.log(n_dim_m * n_dim_d) + log_x_i_x_j
        second_term = np.log(2) + 2 * np.log(C_1) + 3 * np.log(M) + np.log(
            n_dim_e * n_dim_m) + 0.5 * np.log(n_dim_m * n_dim_d) + log_x_i_x_j_3
        third_term = 4 * np.log(C_1) + 7 * np.log(M) + 1.5 * np.log(n_dim_e * (n_dim_m**2) * n_dim_d) + log_x_i_x_j_2 + np.log(lambda_max * ((n_dim_m * n_dim_d)**0.5) + M * n_dim_m * (
            (n_dim_e * n_dim_d)**0.5) + 2 * lambda_max * ((n_dim_e * n_dim_m)**0.5))
        return logsumexp([first_term, second_term, third_term])

    log_h_A = calc_log_h_A()
    log_h_B = calc_log_h_B()
    log_h_Lambda = calc_log_h_Lambda()

    log_h_prime = 0.5 * logsumexp([2 * log_h_A, 2 * log_h_B, 2 * log_h_Lambda])

    return log_h_prime


def lossfn(preds, target):
    # does not use target variable
    # one positive sample at the first dimension, and negative samples for remaining dimension.
    # loss function
    pos_score = preds.narrow(1, 0, 1)
    neg_score = preds.narrow(1, 1, preds.size(1) - 1)
    pos_loss = F.logsigmoid(pos_score).squeeze().sum()
    neg_loss = F.logsigmoid(-1 * neg_score).squeeze().sum()
    loss = pos_loss + neg_loss
    return -1 * loss


# from https://github.com/jrios6/graph-neural-networks/blob/master/

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_data2(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix (not required in RGGCN)
    #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    #adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))

    labels = torch.LongTensor(np.where(labels)[1])
    #adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def load_data3(dir_path, dataset_str):
    """
    FROM https://github.com/tkipf/gcn
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/ind.{}.{}".format(dir_path, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "{}/ind.{}.test.index".format(dir_path, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row,
                                          sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

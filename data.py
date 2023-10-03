import os
import numpy as np
import torch
from nltk.corpus import wordnet as wn
from collections import defaultdict
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset, Sampler
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from utils import load_data3
import sys


class EdgeSampler(Sampler):

    def __init__(
        self,
        edges,
        edge_freq,
        smoothing_rate_for_edge,
        edge_table_size,
        node_freq,
        verbose=True
    ):
        self.edge_num = len(edges)
        self.edge_freq = edge_freq
        self.smoothing_rate_for_edge = smoothing_rate_for_edge
        self.edge_table_size = int(edge_table_size)

        c = self.edge_freq ** self.smoothing_rate_for_edge

        self.sample_edge_table = np.zeros(self.edge_table_size, dtype=int)
        index = 0
        # negative samplingの確率込み
        p = c / c.sum()
        d = p[index]
        # sample_edge_tableにサンプリング用の配列を作成する。
        # self.edge_table_size(凄く大きな数)の長さの配列に、
        # 要素がiであるものが、self.edge_table_size * p[i]だけある
        # 配列に仕上げる。randpermを組み合わせると、negative samplingが可能となる。
        for i in tqdm(range(self.edge_table_size)):
            self.sample_edge_table[i] = index
            if i / self.edge_table_size > d:
                index += 1
                d += p[index]
            if index >= self.edge_num:
                index = self.edge_num - 1
        # print(self.sample_edge_table)
        # print(self.sample_edge_table.shape)

    def __iter__(self):
        # randperm is a function that returns a random permutation of 0 to n-1
        return (self.sample_edge_table[i] for i in torch.randperm(self.edge_table_size))

    def __len__(self):
        return self.edge_table_size


class GraphDataset(Dataset):
    ntries = 10
    smoothing_rate_for_edge = 1.0
    # node_table_size = int(1e7)
    # edge_table_size = int(5e7)
    node_table_size = int(1e5)
    edge_table_size = int(5e5)

    def __init__(
        self,
        node2id,
        id2freq,
        edges2freq,
        nnegs,
        smoothing_rate_for_node,
        data_vectors=None,
        task="reconst",
        seed=0
    ):
        assert task in ["reconst", "linkpred", "nodeclf"]

        self.smoothing_rate_for_node = smoothing_rate_for_node
        self.nnegs = nnegs
        self.task = task

        if task == "linkpred":
            assert data_vectors is not None

            # ノードを訓練テストvalidationに分ける
            train_node, test_node = train_test_split(
                list(node2id.keys()), test_size=0.2, random_state=seed)
            train_node, valid_node = train_test_split(
                train_node, test_size=0.2, random_state=seed)

            # self.train_ids = []
            # self.valid_ids = []
            # self.test_ids = []

            # for n in train_node:
            #     self.train_ids.append(int(node2id[n]))
            # for n in valid_node:
            #     self.valid_ids.append(int(node2id[n]))
            # for n in test_node:
            #     self.test_ids.append(int(node2id[n]))

            # self.train_ids=np.array(self.train_ids)
            # self.valid_ids=np.array(self.valid_ids)
            # self.test_ids=np.array(self.test_ids)

            train_node_set = set(train_node)
            node_freq = list()
            valid_node_set = set(valid_node)
            test_node_set = set(test_node)
            print(f"len(train_node) : {len(train_node)}, len(valid_node) : {len(valid_node)}, len(test_node) : {len(test_node)}")
            new_node2id = defaultdict(lambda: len(new_node2id))
            new_data_vectors = np.empty(data_vectors.shape)

            # data vectorの振り直し?
            for i in train_node:
                new_data_vectors[new_node2id[i]] = data_vectors[node2id[i]]
                node_freq.append(id2freq[node2id[i]])
            for i in valid_node:
                new_data_vectors[new_node2id[i]] = data_vectors[node2id[i]]
            for i in test_node:
                new_data_vectors[new_node2id[i]] = data_vectors[node2id[i]]
            new_node2id = dict(new_node2id)

            id2node = dict((y, x) for x, y in node2id.items())
            train_edges = list()
            edge_freq = list()

            neighbor_train = defaultdict(lambda: set())
            neighbor_valid = defaultdict(lambda: set())
            neighbor_test = defaultdict(lambda: set())

            for edge, edgefreq in edges2freq.items():
                i, j = [id2node[k] for k in edge]
                if i in train_node_set and j in train_node_set:
                    train_edges.append((new_node2id[i], new_node2id[j]))
                    edge_freq.append(edgefreq)
                    neighbor_train[new_node2id[i]].add(new_node2id[j])
                    neighbor_train[new_node2id[j]].add(new_node2id[i])
                else:
                    if i in test_node_set or j in test_node_set:
                        if i in test_node_set:
                            neighbor_test[new_node2id[i]].add(new_node2id[j])
                        if j in test_node_set:
                            neighbor_test[new_node2id[j]].add(new_node2id[i])
                    else:
                        if i in valid_node_set:
                            neighbor_valid[new_node2id[i]].add(new_node2id[j])
                        if j in valid_node_set:
                            neighbor_valid[new_node2id[j]].add(new_node2id[i])

            train_edges = np.array(train_edges, dtype=np.int)
            neighbor_train = dict(neighbor_train)
            neighbor_valid = dict(neighbor_valid)
            neighbor_test = dict(neighbor_test)

            self.node2id = new_node2id
            self.data_vectors = new_data_vectors
            self.total_node_num = len(node2id)
            self.train_node_num = len(train_node)
            self.edges = train_edges
            self.total_edge_num = len(edges2freq)
            self.train_edge_num = len(self.edges)
            self.node_freq = np.array(node_freq, dtype=np.float)
            self.edge_freq = np.array(edge_freq, dtype=np.float)
            self.max_tries = self.nnegs * self.ntries
            self.neighbor_train = neighbor_train
            self.neighbor_valid = neighbor_valid
            self.neighbor_test = neighbor_test

        # elif task == "reconst":
        else:
            if task == "nodeclf":
                # ノードを訓練テストvalidationに分ける
                train_node, test_node = train_test_split(
                    list(node2id.keys()), test_size=0.2, random_state=seed)
                train_node, valid_node = train_test_split(
                    train_node, test_size=0.2, random_state=seed)

                self.train_ids = []
                self.valid_ids = []
                self.test_ids = []

                for n in train_node:
                    self.train_ids.append(int(node2id[n]))
                for n in valid_node:
                    self.valid_ids.append(int(node2id[n]))
                for n in test_node:
                    self.test_ids.append(int(node2id[n]))

                self.train_ids = np.array(self.train_ids)
                self.valid_ids = np.array(self.valid_ids)
                self.test_ids = np.array(self.test_ids)

            self.node2id = node2id
            self.data_vectors = data_vectors
            self.total_node_num = len(node2id)
            # the number of train nodes are the number of total nodes if
            # reconst mode
            self.train_node_num = self.total_node_num

            # edge-related variables
            self.edges = list()
            self.edge_freq = list()
            for edge, freq in edges2freq.items():
                self.edges.append(edge)
                self.edge_freq.append(freq)
            self.edges = np.array(self.edges, dtype=np.int)
            self.edge_freq = np.array(self.edge_freq, dtype=np.float)
            self.total_edge_num = len(self.edges)
            self.train_edge_num = self.total_edge_num

            # node-related varibales
            self.node_freq = np.zeros(self.train_node_num, dtype=np.float)
            for i, f in id2freq.items():
                self.node_freq[i] = f

            self.max_tries = self.nnegs * self.ntries
            self.neighbor_train = defaultdict(lambda: set())
            for i, j in self.edges:
                self.neighbor_train[i].add(j)
                self.neighbor_train[j].add(i)

            self.neighbor_train = dict(self.neighbor_train)
            # None for validation and test
            self.neighbor_valid = None
            self.neighbor_test = None
            # print(len(self.neighbor_train))
            # print(self.train_node_num)
            assert len(self.neighbor_train) == self.train_node_num

        # else:

            # ノードの選択確率?negative samplingっぽい感じにはなっている
        c = self.node_freq ** self.smoothing_rate_for_node
        self.sample_node_table = np.zeros(self.node_table_size, dtype=int)
        index = 0
        p = c / c.sum()
        d = p[index]
        for i in tqdm(range(self.node_table_size)):
            self.sample_node_table[i] = index
            if i / self.node_table_size > d:
                index += 1
                d += p[index]
            if index >= self.train_node_num:
                index = self.train_node_num - 1

        self.sampler = EdgeSampler(
            self.edges, self.edge_freq, self.smoothing_rate_for_edge, self.edge_table_size, self.node_freq)

    def __len__(self):
        return self.train_edge_num

    def __getitem__(self, i):
        # positive sample
        i, j = [int(x) for x in self.edges[i]]
        if np.random.randint(2) == 1:
            i, j = j, i

        # negative sampling. iもjも含まないものとする.
        negs = set()
        ntries = 0
        nnegs = self.nnegs
        while ntries < self.max_tries and len(negs) < nnegs:
            n = np.random.randint(0, self.node_table_size)
            n = int(self.sample_node_table[n])
            if n != i and n != j:
                negs.add(n)
            ntries += 1
        ix = [i, j] + list(negs)

        # negative sampleが足りない時は、既にサンプリングしたものを複製する。
        while len(ix) < nnegs + 2:
            ix.append(ix[np.random.randint(2, len(ix))])

        # The target variable is dummy. The first element of ix is the positive
        # sample, and the remaining is the set of negative samples.
        return torch.LongTensor(ix).view(1, len(ix)), torch.zeros(1).long()

    @classmethod
    def collate(cls, batch):
        inputs, targets = zip(*batch)
        return Variable(torch.cat(inputs, 0)), Variable(torch.cat(targets, 0))


def preprocess_hierarchy(word2vec_path, use_rich_information=False, verbose=True):
    if word2vec_path is not None:
        assert use_rich_information == False

    def _clean(word):
        if use_rich_information:
            return word
        else:
            word = word.split(".n.")[0]
            word = word.lower()
            return word

    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    # word2vec_vocab = set(word2vec.vocab.keys())
    word2vec_vocab = set(list(word2vec.index_to_key))
    word2data_vector = dict()
    for word in word2vec_vocab:
        lowerword = word.lower()
        if lowerword in word2vec_vocab:
            word2data_vector[lowerword] = word2vec[lowerword]
        else:
            word2data_vector[lowerword] = word2vec[word]

    word2id = defaultdict(lambda: len(word2id))
    id2freq = defaultdict(lambda: 0)
    edges2freq = defaultdict(lambda: 0)

    def _memo(word1, word2):
        if word1 in word2data_vector and word2 in word2data_vector:
            id_1 = word2id[word1]
            id_2 = word2id[word2]
            id2freq[id_1] += 1
            id2freq[id_2] += 1
            if id_1 > id_2:
                id_1, id_2 = id_2, id_1
            edges2freq[(id_1, id_2)] = 1

    if verbose:
        pbar = tqdm(total=len(list(wn.all_synsets(pos='n'))))
    for synset in wn.all_synsets(pos='n'):
        if verbose:
            pbar.update(1)
        for hyper in synset.closure(lambda s: s.hypernyms()):
            word1 = _clean(hyper.name())
            word2 = _clean(synset.name())
            _memo(word1, word2)
        for instance in synset.instance_hyponyms():
            for hyper in instance.closure(lambda s: s.instance_hypernyms()):
                word1 = _clean(hyper.name())
                word2 = _clean(instance.name())
                _memo(word1, word2)
                for h in hyper.closure(lambda s: s.hypernyms()):
                    word1 = _clean(h.name())
                    _memo(word1, word2)
    if verbose:
        pbar.close()

    word2id = dict(word2id)
    id2freq = dict(id2freq)
    edges2freq = dict(edges2freq)
    vectors = np.empty((len(word2id), 300))
    for word, index in word2id.items():
        vectors[index] = word2data_vector[word]

    print(f"""Node num : {len(word2id)},
    Node frequency max :{np.max(list(id2freq.values()))} min :{np.min(list(id2freq.values()))} mean :{np.mean(list(id2freq.values()))},
    Edge num : {len(edges2freq)},
    Edge frequency max :{np.max(list(edges2freq.values()))} min :{np.min(list(edges2freq.values()))} mean :{np.mean(list(edges2freq.values()))}""")

    return word2id, id2freq, edges2freq, vectors


def iter_line(fname, sep='\t', type=tuple, comment='#', return_idx=False, convert=None):
    with open(fname, 'r') as fin:
        if return_idx:
            index = -1
        for line in fin:
            if line[0] == comment:
                continue
            if convert is not None:
                d = [convert(i) for i in line.strip().split(sep)]
            else:
                d = line.strip().split(sep)
            out = type(d)
            if out is not None:
                if return_idx:
                    index += 1
                    yield (index, out)
                else:
                    yield out


def preprocess_co_author_network(dir_path, undirect=True, seed=0):
    author2id = defaultdict(lambda: len(author2id))
    edges2freq = dict()
    for _i, _j in iter_line(dir_path + "/graph_dblp.txt", sep='\t', type=tuple, convert=int):
        i = author2id[_i]
        j = author2id[_j]
        if i > j:
            j, i = i, j
        edges2freq[(i, j)] = 1

    author2id = dict(author2id)

    vectors = np.empty((len(author2id), 33))
    selected_attributes = np.array([0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
                                    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 34, 35, 37])
    for i, vec in iter_line(dir_path + "/db_normalz_clus.txt", sep=',', type=np.array, convert=float, return_idx=True):
        assert vec.shape[0] == 38
        if i in author2id:
            vec = vec.astype(np.float32)[selected_attributes]
            vectors[author2id[i]] = vec

    id2freq = defaultdict(lambda: 0)
    for key, value in edges2freq.items():
        id2freq[key[0]] += value
        id2freq[key[1]] += value
    id2freq = dict(id2freq)

    print(f"""Node num : {len(author2id)},
    Node frequency max :{np.max(list(id2freq.values()))} min :{np.min(list(id2freq.values()))} mean :{np.mean(list(id2freq.values()))},
    Edge num : {len(edges2freq)},
    Edge frequency max :{np.max(list(edges2freq.values()))} min :{np.min(list(edges2freq.values()))} mean :{np.mean(list(edges2freq.values()))}""")

    return author2id, id2freq, edges2freq, vectors

# from torch_geometric.io import read_npz

# import os.path as osp
# from typing import Callable, Optional

# import torch

# from torch_geometric.data import InMemoryDataset, download_url
# from torch_geometric.io import read_npz


# class Amazon(InMemoryDataset):
#     r"""The Amazon Computers and Amazon Photo networks from the
#     `"Pitfalls of Graph Neural Network Evaluation"
#     <https://arxiv.org/abs/1811.05868>`_ paper.
#     Nodes represent goods and edges represent that two goods are frequently
#     bought together.
#     Given product reviews as bag-of-words node features, the task is to
#     map goods to their respective product category.

#     Args:
#         root (str): Root directory where the dataset should be saved.
#         name (str): The name of the dataset (:obj:`"Computers"`,
#             :obj:`"Photo"`).
#         transform (callable, optional): A function/transform that takes in an
#             :obj:`torch_geometric.data.Data` object and returns a transformed
#             version. The data object will be transformed before every access.
#             (default: :obj:`None`)
#         pre_transform (callable, optional): A function/transform that takes in
#             an :obj:`torch_geometric.data.Data` object and returns a
#             transformed version. The data object will be transformed before
#             being saved to disk. (default: :obj:`None`)

#     **STATS:**

#     .. list-table::
#         :widths: 10 10 10 10 10
#         :header-rows: 1

#         * - Name
#           - #nodes
#           - #edges
#           - #features
#           - #classes
#         * - Computers
#           - 13,752
#           - 491,722
#           - 767
#           - 10
#         * - Photo
#           - 7,650
#           - 238,162
#           - 745
#           - 8
#     """

#     url = 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/'

#     def __init__(
#         self,
#         root: str,
#         name: str,
#         transform: Optional[Callable] = None,
#         pre_transform: Optional[Callable] = None,
#     ):
#         self.name = name.lower()
#         assert self.name in ['computers', 'photo']
#         super().__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_dir(self) -> str:
#         return osp.join(self.root, self.name.capitalize(), 'raw')

#     @property
#     def processed_dir(self) -> str:
#         return osp.join(self.root, self.name.capitalize(), 'processed')

#     @property
#     def raw_file_names(self) -> str:
#         return f'amazon_electronics_{self.name.lower()}.npz'

#     @property
#     def processed_file_names(self) -> str:
#         return 'data.pt'

#     def download(self):
#         download_url(self.url + self.raw_file_names, self.raw_dir)

#     def process(self):
#         # data = read_npz(self.raw_paths[0], to_undirected=True)
#         data = read_npz(self.raw_paths[0])
#         data = data if self.pre_transform is None else self.pre_transform(data)
#         data, slices = self.collate([data])
#         torch.save((data, slices), self.processed_paths[0])

#     def __repr__(self) -> str:
#         return f'{self.__class__.__name__}{self.name.capitalize()}()'

# def preprocess_amazon(
#     dir_path
# ):
#     dataset = read_npz("data/amazon_electronics_photo.npz")
#     # dataset = np.load("data/amazon_electronics_photo.npz")
#     # print(dataset)
#     # print(dataset.x)
#     # print(dataset.y)
#     # print(dataset.edge_index)

#     features = dataset.x.numpy()
#     labels = dataset.y.numpy()

#     node2id = {}
#     id2freq = {}
#     edges2freq = {}

#     n = dataset.x.shape[0]

#     for i in range(n):
#         node2id[str(i)] = i
#         id2freq[i] = 0
#         # id2freq[i] = freq[i]


#     for e in dataset.edge_index.numpy().T:
#         id2freq[e[0]] += 1
#         id2freq[e[1]] += 1
#         if e[0] < e[1]:
#             edges2freq[(e[0], e[1])] = 1
#         else:
#             edges2freq[(e[1], e[0])] = 1

#     # print(node2id)
#     # print(id2freq)
#     # print(edges2freq)
#     # print(labels)
#     # print(features)

#     # dataset = Amazon("data", "photo")
#     # print(dataset)

#     print(f"""Node num : {len(node2id)},
#     Node frequency max :{np.max(list(id2freq.values()))} min :{np.min(list(id2freq.values()))} mean :{np.mean(list(id2freq.values()))},
#     Edge num : {len(edges2freq)},
#     Edge frequency max :{np.max(list(edges2freq.values()))} min :{np.min(list(edges2freq.values()))} mean :{np.mean(list(edges2freq.values()))}""")

#     # print(set(labels))

#     # return node2id, id2freq, edges2freq, features, labels

def preprocess_amazon(
    dir_path
):

    node2id = np.load("data/amazon_electronics_photo_node2id.npy", allow_pickle=True).item()
    id2freq = np.load("data/amazon_electronics_photo_id2freq.npy", allow_pickle=True).item()
    edges2freq = np.load("data/amazon_electronics_photo_edges2freq.npy", allow_pickle=True).item()
    features = np.load("data/amazon_electronics_photo_features.npy", allow_pickle=True)
    labels = np.load("data/amazon_electronics_photo_labels.npy", allow_pickle=True)


    print(f"""Node num : {len(node2id)},
    Node frequency max :{np.max(list(id2freq.values()))} min :{np.min(list(id2freq.values()))} mean :{np.mean(list(id2freq.values()))},
    Edge num : {len(edges2freq)},
    Edge frequency max :{np.max(list(edges2freq.values()))} min :{np.min(list(edges2freq.values()))} mean :{np.mean(list(edges2freq.values()))}""")

    return node2id, id2freq, edges2freq, features, labels


def preprocess_citeseer(dir_path):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data3(
        dir_path, 'citeseer')
    adj = adj.toarray().astype(float)

    # ラベルがついていないものはとりのぞく
    idx_with_label = np.where(labels)[0]

    labels = labels[idx_with_label, :]
    labels = np.where(labels)[1]
    features = features[idx_with_label]
    adj = adj[idx_with_label, :][:, idx_with_label]

    features = features.numpy()

    # for input data

    node2id = {}
    id2freq = {}
    edges2freq = {}

    n = adj.shape[0]
    freq = np.sum(adj, axis=1)

    for i in range(n):
        node2id[str(i)] = i
        id2freq[i] = freq[i]

    adj = np.triu(adj)

    for e in np.array(np.where(adj == 1)).T:
        if e[0] < e[1]:
            edges2freq[(e[0], e[1])] = 1
        else:
            edges2freq[(e[1], e[0])] = 1

    print(f"""Node num : {len(node2id)},
    Node frequency max :{np.max(list(id2freq.values()))} min :{np.min(list(id2freq.values()))} mean :{np.mean(list(id2freq.values()))},
    Edge num : {len(edges2freq)},
    Edge frequency max :{np.max(list(edges2freq.values()))} min :{np.min(list(edges2freq.values()))} mean :{np.mean(list(edges2freq.values()))}""")

    return node2id, id2freq, edges2freq, features, labels


# def preprocess_pubmed(dir_path):
#     adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data3(
#         dir_path, 'pubmed')
#     adj = adj.toarray().astype(float)

#     # ラベルがついていないものはとりのぞく
#     idx_with_label = np.where(labels)[0]

#     labels = labels[idx_with_label]
#     labels = np.where(labels)[1]

#     features = features[idx_with_label]
#     adj = adj[idx_with_label, :][:, idx_with_label]

#     features = features.numpy()

#     # for input data
#     node2id = {}
#     id2freq = {}
#     edges2freq = {}

#     n = adj.shape[0]
#     freq = np.sum(adj, axis=1)

#     for i in range(n):
#         node2id[str(i)] = i
#         id2freq[i] = freq[i]

#     adj = np.triu(adj)

#     for e in np.array(np.where(adj == 1)).T:
#         if e[0] < e[1]:
#             edges2freq[(e[0], e[1])] = 1
#         else:
#             edges2freq[(e[1], e[0])] = 1

#     print(f"""Node num : {len(node2id)},
#     Node frequency max :{np.max(list(id2freq.values()))} min :{np.min(list(id2freq.values()))} mean :{np.mean(list(id2freq.values()))},
#     Edge num : {len(edges2freq)},
#     Edge frequency max :{np.max(list(edges2freq.values()))} min :{np.min(list(edges2freq.values()))} mean :{np.mean(list(edges2freq.values()))}""")

#     return node2id, id2freq, edges2freq, features, labels

def preprocess_pubmed(dir_path):
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, labels = load_data3(
        dir_path, 'pubmed')
    adj = adj.toarray().astype(float)

    # ラベルがついていないものはとりのぞく
    idx_with_label = np.where(labels)[0]
    idx_with_label = idx_with_label[:int(len(idx_with_label)*0.4)]

    labels = labels[idx_with_label]
    labels = np.where(labels)[1]

    features = features[idx_with_label]
    adj = adj[idx_with_label, :][:, idx_with_label]

    features = features.numpy()

    # print(set(labels))

    # for input data
    node2id = {}
    id2freq = {}
    edges2freq = {}

    n = adj.shape[0]
    freq = np.sum(adj, axis=1)

    for i in range(n):
        node2id[str(i)] = i
        id2freq[i] = freq[i]

    adj = np.triu(adj)

    for e in np.array(np.where(adj == 1)).T:
        if e[0] < e[1]:
            edges2freq[(e[0], e[1])] = 1
        else:
            edges2freq[(e[1], e[0])] = 1

    print(f"""Node num : {len(node2id)},
    Node frequency max :{np.max(list(id2freq.values()))} min :{np.min(list(id2freq.values()))} mean :{np.mean(list(id2freq.values()))},
    Edge num : {len(edges2freq)},
    Edge frequency max :{np.max(list(edges2freq.values()))} min :{np.min(list(edges2freq.values()))} mean :{np.mean(list(edges2freq.values()))}""")

    # sys.exit()

    return node2id, id2freq, edges2freq, features, labels

def preprocess_cora(dir_path):
    all_data = []
    all_edges = []

    with open(dir_path + "/cora.content", 'r') as f:
        all_data.extend(f.read().splitlines())
    with open(dir_path + "/cora.cites", 'r') as f:
        all_edges.extend(f.read().splitlines())

    labels = []
    nodes = []
    X = []

    for i, data in enumerate(all_data):
        elements = data.split('\t')
        labels.append(elements[-1])
        X.append(elements[1:-1])
        nodes.append(elements[0])

    X = np.array(X, dtype=int)
    N = X.shape[0]  # the number of nodes
    F = X.shape[1]  # the size of node features
    print('X shape: ', X.shape)

    # parse the edge
    edge_list = []
    for edge in all_edges:
        e = edge.split('\t')
        edge_list.append((e[0], e[1]))

    nodes = np.array(nodes)
    labels = np.array(labels)
    edge_list = np.array(edge_list)

    # for input data

    nodes2id = {}
    id2freq = {}
    edges2freq = {}

    for i in range(len(nodes)):
        id2freq[i] = 0

    for i, node in enumerate(nodes):
        nodes2id[str(node)] = i

    for edge in edge_list:
        i, j = edge
        id_i = nodes2id[i]
        id_j = nodes2id[j]

        # update edges2freq
        if id_i > id_j:
            id_i, id_j = id_j, id_i
        edges2freq[(id_i, id_j)] = 1

        # update id2freq
        id2freq[id_i] += 1
        id2freq[id_j] += 1

    print('\nNumber of nodes (N): ', N)
    print('\nNumber of features (F) of each node: ', F)
    print('\nCategories: ', set(labels))

    num_classes = len(set(labels))
    print('\nNumber of classes: ', num_classes)

    return nodes2id, id2freq, edges2freq, X, labels


def preprocess_webkb_network(dir_path):
    print(dir_path)
    node2id = defaultdict(lambda: len(node2id))
    edges2freq = dict()
    for _i, _j in iter_line(dir_path + "/WebKB.cites", sep='\t', type=tuple, convert=str):
        i = node2id[_i]
        j = node2id[_j]
        if i > j:
            j, i = i, j
        edges2freq[(i, j)] = 1
    node2id = dict(node2id)
    id2freq = defaultdict(lambda: 0)
    for key, value in edges2freq.items():
        id2freq[key[0]] += value
        id2freq[key[1]] += value
    id2freq = dict(id2freq)
    vectors = np.empty((len(node2id), 1703), dtype=np.float)
    lines = open(dir_path + "/WebKB.content").readlines()
    labels = np.empty(len(node2id), dtype=int)
    categories = []

    for line in lines:
        elements = line.strip().split()
        categories.append(elements[-1])
    categories = list(set(categories))
    print(categories)
    # print(lines)

    cat2id = {}
    for i, cat in enumerate(categories):
        cat2id[cat] = i

    for line in lines:
        elements = line.strip().split()
        # print(len(elements))
        node = str(elements[0])
        vec = np.array([int(i) for i in elements[1:-1]], dtype=np.float)
        assert len(vec) == 1703
        vectors[node2id[node]] = vec
        # print(node2id[node])
        # print(elements[-1])
        labels[node2id[node]] = cat2id[elements[-1]]

    # print(labels)
    # print(set(labels))

    print(f"""Node num : {len(node2id)},
    Node frequency max :{np.max(list(id2freq.values()))} min :{np.min(list(id2freq.values()))} mean :{np.mean(list(id2freq.values()))},
    Edge num : {len(edges2freq)},
    Edge frequency max :{np.max(list(edges2freq.values()))} min :{np.min(list(edges2freq.values()))} mean :{np.mean(list(edges2freq.values()))}""")

    return node2id, id2freq, edges2freq, vectors, labels

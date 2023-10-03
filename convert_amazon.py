import os
import numpy as np
import torch
from nltk.corpus import wordnet as wn
from collections import defaultdict
from tqdm import tqdm
# from torch.autograd import Variable
# from torch.utils.data import Dataset, Sampler
# from sklearn.model_selection import train_test_split
# from gensim.models import KeyedVectors
# from utils import load_data3
import sys
from torch_geometric.io import read_npz

def save_amazon(
):
    dataset = read_npz("data/amazon_electronics_photo.npz")
    # dataset = np.load("data/amazon_electronics_photo.npz")
    # print(dataset)
    # print(dataset.x)
    # print(dataset.y)
    # print(dataset.edge_index)

    features = dataset.x.numpy()
    labels = dataset.y.numpy()

    node2id = {}
    id2freq = {}
    edges2freq = {}

    n = dataset.x.shape[0]

    for i in range(n):
        node2id[str(i)] = i
        id2freq[i] = 0
        # id2freq[i] = freq[i]


    for e in dataset.edge_index.numpy().T:
        id2freq[e[0]] += 1
        id2freq[e[1]] += 1
        if e[0] < e[1]:
            edges2freq[(e[0], e[1])] = 1
        else:
            edges2freq[(e[1], e[0])] = 1

    print(f"""Node num : {len(node2id)},
    Node frequency max :{np.max(list(id2freq.values()))} min :{np.min(list(id2freq.values()))} mean :{np.mean(list(id2freq.values()))},
    Edge num : {len(edges2freq)},
    Edge frequency max :{np.max(list(edges2freq.values()))} min :{np.min(list(edges2freq.values()))} mean :{np.mean(list(edges2freq.values()))}""")

    np.save("data/amazon_electronics_photo_node2id.npy", node2id)
    np.save("data/amazon_electronics_photo_id2freq.npy", id2freq)
    np.save("data/amazon_electronics_photo_edges2freq.npy", edges2freq)
    np.save("data/amazon_electronics_photo_features.npy", features)
    np.save("data/amazon_electronics_photo_labels.npy", labels)

    # return node2id, id2freq, edges2freq, features, labels


if __name__=="__main__":
    save_amazon()

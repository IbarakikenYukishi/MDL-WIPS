import sys
import argparse
import logging
import random
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import models
import train
import data
from torch.optim import Adam
from scipy.special import logsumexp
from utils import calc_log_h_prime


def main():
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-debug', help='Print debug output',
                        action='store_true', default=True)
    parser.add_argument(
        '-save_dir', help='Path for saving checkpoints', type=str, required=True)
    parser.add_argument('-exp_name', help='Experiment name', type=str,
                        default=str(datetime.datetime.now()).replace(" ", "_"))
    parser.add_argument('-graph_type', help='Graph type: "webkb", "collab" or "hierarchy"',
                        type=str, required=True, choices=["hierarchy", "collab", "webkb", "cora", "citeseer", "pubmed", "amazon"])
    # parser.add_argument(
    #     '-neproc', help='Number of eval processes', type=int, default=32)
    parser.add_argument(
        '-neproc', help='Number of eval processes', type=int, default=8)
    # parser.add_argument('-seed', help='Random seed', type=int, required=False)
    parser.add_argument('-seed', help='Random seed',
                        type=int, default=1, required=False)

    parser.add_argument('-dblp_path', help='dblp_path',
                        type=str, required=False)
    parser.add_argument('-webkb_path', help='webkb_path',
                        type=str, required=False)
    parser.add_argument('-cora_path', help='cora_path',
                        type=str, required=False)
    parser.add_argument('-citeseer_path', help='citeseer_path',
                        type=str, required=False)
    parser.add_argument('-pubmed_path', help='pubmed_path',
                        type=str, required=False)
    parser.add_argument('-amazon_path', help='amazon_path',
                        type=str, required=False)

    # parser.add_argument(
    #     '-word2vec_path', help='word2vec_path', type=str, required=False)
    parser.add_argument(
        '-word2vec_path',
        help='word2vec_path',
        type=str,
        default="data/word2vec/GoogleNews-vectors-negative300.bin",
        required=False
    )

    # parser.add_argument(
    #     '-iter', help='Number of iterations', type=int, default=10)
    parser.add_argument(
        '-iter', help='Number of iterations', type=int, default=100000)
    parser.add_argument(
        '-eval_each', help='Run evaluation at every n-th iter', type=int, default=5000)
    parser.add_argument(
        '-init_lr', help='Initial learning rate', type=float, default=0.1)
    # parser.add_argument('-batchsize', help='Batchsize', type=int, default=2)
    parser.add_argument('-batchsize', help='Batchsize', type=int, default=64)
    parser.add_argument(
        '-negs', help='Number of negative samples', type=int, default=10)

    # negative sampling's smoothing rate, but I am not sure whether it is
    # necessary or not for simple graphs
    parser.add_argument('-smoothing_rate_for_node',
                        help='Smoothing rate for the negative sampling', type=float, default=1.0)

    # 考えているモデルなら1に固定したい。
    # parser.add_argument('-hidden_layer_num',
    #                     help='Number of hidden layers', type=int, default=2)
    parser.add_argument('-hidden_layer_num',
                        help='Number of hidden layers', type=int, default=1)

    parser.add_argument(
        '-hidden_size', help='Number of units in a hidden layer', type=int, default=1000)
    # parser.add_argument(
    #     '-hidden_size', help='Number of units in a hidden layer', type=int, default=100)

    # parser.add_argument('-model_name', help='Model: "IPS", "SIPS", "NPD", "IPDS" or "WIPS"',
    # type=str, required=True, choices=["IPS", "SIPS", "NPD", "IPDS", "WIPS"])
    parser.add_argument('-model_name', help='Model: "IPS", "SIPS", "NPD", "IPDS", "WIPS" or "MDL-WIPS"',
                        type=str, required=True, choices=["IPS", "SIPS", "NPD", "IPDS", "WIPS", "MDL_WIPS"])
    parser.add_argument('-task', help='', type=str, default="reconst")
    # parser.add_argument(
    #     '-parameter_num', help='Parameter number K for each node', type=int, default=100)
    parser.add_argument(
        '-parameter_num', help='Parameter number K for each node', type=int, default=100)

    parser.add_argument(
        '-cuda', help='cuda device', type=int, default=0, required=False)

    parser.add_argument('-neg_ratio', help='Dimension ratio for negative IPS in IPDS',
                        type=float, default=0.0, required=False)
    parser.add_argument('-neg_dim', help='Dimension for negative IPS in IPDS',
                        type=int, default=0, required=False)

    opt = parser.parse_args()

    if opt.seed == None:
        opt.seed = random.randint(1, 1000000)
    torch.manual_seed(opt.seed)
    # opt.cuda = torch.cuda.is_available()
    # if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    # if opt.cuda:
    #     torch.cuda.manual_seed(opt.seed)
    #     torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    torch.set_default_tensor_type('torch.FloatTensor')
    # enable the line below if you want to add the date to the output file
    # opt.exp_name += "@" + str(datetime.datetime.now()).replace(" ", "_")
    if opt.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    log = logging.getLogger(opt.exp_name)
    fileHandler = logging.FileHandler(f'{opt.save_dir}/{opt.exp_name}.log')
    streamHandler = logging.StreamHandler()
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    log.setLevel(log_level)

    log.info(f"Experiment {opt.exp_name} start with the following setting:\n{str(opt.__dict__)}")

    if opt.graph_type == "hierarchy":
        word2id, id2freq, edges2freq, vectors = data.preprocess_hierarchy(
            opt.word2vec_path, use_rich_information=False)
    # elif opt.graph_type == "collab":
    #     word2id, id2freq, edges2freq, vectors = data.preprocess_co_author_network(
    #         opt.dblp_path)
    elif opt.graph_type == "webkb":
        word2id, id2freq, edges2freq, vectors, labels = data.preprocess_webkb_network(
            opt.webkb_path)
    elif opt.graph_type == "cora":
        word2id, id2freq, edges2freq, vectors, labels = data.preprocess_cora(
            opt.cora_path)
    elif opt.graph_type == "citeseer":
        word2id, id2freq, edges2freq, vectors, labels = data.preprocess_citeseer(
            opt.citeseer_path)
    elif opt.graph_type == "pubmed":
        word2id, id2freq, edges2freq, vectors, labels = data.preprocess_pubmed(
            opt.pubmed_path)
    elif opt.graph_type == "amazon":
        word2id, id2freq, edges2freq, vectors, labels = data.preprocess_amazon(
            opt.amazon_path)

    vectors = np.concatenate((vectors, np.ones((vectors.shape[0], 1))), axis=1)
    # print(vectors.shape)

    # word2id: 単語とidの対応関係のdict
    # print(word2id)

    # id2freq: edgeに出てくるidのfrequency?
    # print(id2freq)

    # edgeのペアのfrequency?
    # print(edges2freq)

    # vectors: GraphDatasetに使う。reconstの場合はdata vectorになり、link
    # predictionの場合は何らかの処理を加え、data vectorの元になる。
    # print(vectors.shape)
    # print(vectors.shape)

    dataset = data.GraphDataset(word2id, id2freq, edges2freq, opt.negs,
                                opt.smoothing_rate_for_node, vectors, opt.task, opt.seed)
    # optにデータを加える
    opt.data_vectors = dataset.data_vectors
    opt.total_node_num = dataset.total_node_num
    opt.train_node_num = dataset.train_node_num

    opt.lik_pos_ratio = len(edges2freq) / opt.batchsize
    opt.lik_neg_ratio = (
        opt.train_node_num * (opt.train_node_num - 1) / 2 - len(edges2freq)) / (opt.batchsize * opt.negs)

    # print(opt.lik_neg_ratio)
    # print(opt.lik_pos_ratio)

    # IPDSの場合neg_ratioからneg_dimを計算する
    if opt.model_name == "IPDS":
        if opt.neg_dim == 0:
            opt.neg_dim = np.round(opt.neg_ratio * opt.parameter_num)
        if opt.neg_dim == 0:
            opt.neg_dim = 1
        if opt.neg_dim == opt.parameter_num:
            opt.neg_dim = opt.parameter_num - 1
        opt.neg_dim = int(opt.neg_dim)

    model = getattr(models, opt.model_name)(opt.__dict__)

    filtered_parameters = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            filtered_parameters.append(param)
    # print(filtered_parameters)
    params_num = sum([np.prod(p.size()) for p in filter(
        lambda p: p.requires_grad, model.parameters())])

    # if opt.cuda:
    #     model = model.cuda()
    model = model.cuda(opt.cuda)

    log_h_prime = calc_log_h_prime(
        data_vectors=opt.data_vectors,
        C_1=1,  # 1 for tanh
        C_2=4 * np.sqrt(3) / 9,  # 4/9*\sqrt(3) for tanh
        lambda_max=10000.0,
        M=10000.0,
        n_dim_e=opt.parameter_num,
        n_dim_m=opt.hidden_size,
        n_dim_d=opt.data_vectors.shape[1],
    )

    if opt.model_name=="MDL_WIPS":
        optimizer = train.ProxSGD(
            params=filtered_parameters,
            lr=opt.init_lr,
            log_h_prime=log_h_prime,
            batchsize=opt.batchsize,
            M=10000.0,
            lambda_max=10000.0,
            n_nodes=dataset.train_node_num,
            n_dim_e=opt.parameter_num,
            n_dim_m=opt.hidden_size,
            n_dim_d=opt.data_vectors.shape[1],
            device=opt.cuda
        )
    else:
        optimizer = Adam(
            filtered_parameters,
            lr=opt.init_lr
        )

    # train the model
    train.trainer_for_ablation(
        model,
        dataset,
        lossfn,
        optimizer,
        opt.__dict__,
        log,
        opt.cuda,
        labels=labels,
    )


def lossfn(preds, target):
    # does not use target variable
    # one positive sample at the first dimension, and negative samples for remaining dimension.
    # loss function
    pos_score = preds.narrow(1, 0, 1)
    neg_score = preds.narrow(1, 1, preds.size(1) - 1)
    pos_loss = F.logsigmoid(pos_score).squeeze().sum()
    neg_loss = F.logsigmoid(-1 * neg_score).squeeze().sum()
    # loss = pos_loss + neg_loss
    # return -1 * loss
    return -pos_loss, -neg_loss


# def lossfn(preds, target):
#     # does not use target variable
#     # one positive sample at the first dimension, and negative samples for remaining dimension.
#     # loss function
#     pos_score = preds.narrow(1, 0, 1)
#     neg_score = preds.narrow(1, 1, preds.size(1) - 1)
#     pos_loss = F.logsigmoid(pos_score).squeeze().sum()
#     neg_loss = F.logsigmoid(-1 * neg_score).squeeze().sum()
#     loss = pos_loss + neg_loss
#     return -1 * loss

if __name__ == '__main__':
    main()

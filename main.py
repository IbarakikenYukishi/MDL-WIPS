import sys
import argparse
import logging
import random
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import models
import train
import data
from scipy.special import logsumexp


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
                        type=str, required=True, choices=["hierarchy", "collab", "webkb"])
    parser.add_argument(
        '-neproc', help='Number of eval processes', type=int, default=32)
    parser.add_argument('-seed', help='Random seed', type=int, required=False)

    parser.add_argument('-dblp_path', help='dblp_path',
                        type=str, required=False)
    parser.add_argument('-webkb_path', help='webkb_path',
                        type=str, required=False)
    parser.add_argument(
        '-word2vec_path', help='word2vec_path', type=str, required=False)

    # parser.add_argument(
    #     '-iter', help='Number of iterations', type=int, default=10)
    parser.add_argument(
        '-iter', help='Number of iterations', type=int, default=100000)
    parser.add_argument(
        '-eval_each', help='Run evaluation at every n-th iter', type=int, default=5000)
    parser.add_argument(
        '-init_lr', help='Initial learning rate', type=float, default=0.1)
    # parser.add_argument('-batchsize', help='Batchsize', type=int, default=2)
    parser.add_argument('-batchsize', help='Batchsize', type=int, default=32)
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
        '-hidden_size', help='Number of units in a hidden layer', type=int, default=2000)
    # parser.add_argument(
    #     '-hidden_size', help='Number of units in a hidden layer', type=int, default=100)

    # parser.add_argument('-model_name', help='Model: "IPS", "SIPS", "NPD", "IPDS" or "WIPS"',
    # type=str, required=True, choices=["IPS", "SIPS", "NPD", "IPDS", "WIPS"])
    parser.add_argument('-model_name', help='Model: "IPS", "SIPS", "NPD", "IPDS" or "WIPS"',
                        type=str, required=True, choices=["IPS", "SIPS", "NPD", "IPDS", "WIPS", "MDL_WIPS"])
    parser.add_argument('-task', help='', type=str, default="reconst")
    # parser.add_argument(
    #     '-parameter_num', help='Parameter number K for each node', type=int, default=100)
    parser.add_argument(
        '-parameter_num', help='Parameter number K for each node', type=int, default=100)

    parser.add_argument('-neg_ratio', help='Dimension ratio for negative IPS in IPDS',
                        type=float, default=0.0, required=False)
    parser.add_argument('-neg_dim', help='Dimension for negative IPS in IPDS',
                        type=int, default=0, required=False)

    opt = parser.parse_args()

    if opt.seed == None:
        opt.seed = random.randint(1, 1000000)
    torch.manual_seed(opt.seed)
    opt.cuda = torch.cuda.is_available()
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    torch.set_default_tensor_type('torch.FloatTensor')
    opt.exp_name += "@" + str(datetime.datetime.now()).replace(" ", "_")
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
    elif opt.graph_type == "collab":
        word2id, id2freq, edges2freq, vectors = data.preprocess_co_author_network(
            opt.dblp_path)
    elif opt.graph_type == "webkb":
        word2id, id2freq, edges2freq, vectors = data.preprocess_webkb_network(
            opt.webkb_path)
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
    # print(vectors)

    dataset = data.GraphDataset(word2id, id2freq, edges2freq, opt.negs,
                                opt.smoothing_rate_for_node, vectors, opt.task, opt.seed)
    # optにデータを加える
    opt.data_vectors = dataset.data_vectors
    opt.total_node_num = dataset.total_node_num
    opt.train_node_num = dataset.train_node_num

    # IPDSの場合neg_ratioからneg_dimを計算する
    if opt.model_name == "IPDS":
        if opt.neg_dim == 0:
            opt.neg_dim = np.round(opt.neg_ratio * opt.parameter_num)
        if opt.neg_dim == 0:
            opt.neg_dim = 1
        if opt.neg_dim == opt.parameter_num:
            opt.neg_dim = opt.parameter_num - 1
        opt.neg_dim = int(opt.neg_dim)

    # print("opt")
    # print(opt)
    # print(opt.__dict__)

    model = getattr(models, opt.model_name)(opt.__dict__)

    filtered_parameters = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            filtered_parameters.append(param)
    # print(filtered_parameters)
    params_num = sum([np.prod(p.size()) for p in filter(
        lambda p: p.requires_grad, model.parameters())])

    if opt.cuda:
        model = model.cuda()

    # optimizer = Adam(
    #     filtered_parameters,
    #     lr=opt.init_lr
    # )

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
    optimizer = train.ProxSGD(
        params=filtered_parameters,
        # lr=opt.init_lr * opt.batchsize,
        lr=opt.init_lr,
        # C_1=1.0,
        # C_2=1.0,
        # data_vectors=np.array([1]),
        log_h_prime=log_h_prime,
        batchsize=opt.batchsize,
        M=10000.0,
        lambda_max=10000.0,
        n_nodes=dataset.train_node_num,
        n_dim_e=opt.parameter_num,
        n_dim_m=opt.hidden_size,
        n_dim_d=opt.data_vectors.shape[1],
        device="cuda:0"
    )

    # train the model
    train.trainer(model, dataset, lossfn, optimizer,
                  opt.__dict__, log, opt.cuda)


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
    # log_x_i_x_j = logsumexp(np.log(np.triu(x_mat, k=1).flatten()))
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

    # print(log_h_A)
    # print(log_h_B)
    # print(log_h_Lambda)

    log_h_prime = 0.5 * logsumexp([2 * log_h_A, 2 * log_h_B, 2 * log_h_Lambda])

    # print(log_h_prime)

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


if __name__ == '__main__':
    main()

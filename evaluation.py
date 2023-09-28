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


def ablation_study(
    path,
    dataset_name,
    init_lr_list,
    n_dim_e_list
):
    prefix = path + "/" + dataset_name
    for n_dim_e in n_dim_e_list:
        best_effective_dim = 0
        best_AUC_valid = -1
        best_AUC_test = -1
        best_lr = 0
        best_iter = 0
        for init_lr in init_lr_list:
            filepath = prefix + "_" + \
                str(n_dim_e) + "_" + str(init_lr) + ".pth"
            data = torch.load(filepath)
            # print(data)
            # print(data["model"]["ips_weight"])
            # print(data["best_rocauc_model"]["ips_weight"])
            ips_weight = data["best_rocauc_model"][
                "ips_weight"].data.cpu().numpy()
            effective_dim = np.sum(ips_weight != 0)
            AUC_valid = data["best_rocauc_valid"]
            AUC_test = data["best_rocauc_valid_test"]
            # print(n_dim_e, init_lr, data["best_rocauc_valid_iteration"])
            # # print(init_lr)
            # print(effective_dim)
            # print(AUC)
            if AUC_valid > best_AUC_valid:
                best_AUC_valid = AUC_valid
                best_AUC_test = AUC_test
                best_effective_dim = effective_dim
                best_lr = init_lr
                best_iter = data["best_rocauc_valid_iteration"]
        print("n_dim_e:", n_dim_e, best_lr, best_iter)
        print("effective_dim:", best_effective_dim)
        print("AUC:", best_AUC_test)


if __name__ == '__main__':
    print("WebKB")
    ablation_study(
        path="results",
        dataset_name="webkb",
        init_lr_list=[0.05, 0.1, 0.2, 0.4, 0.8],
        # n_dim_e_list=[10, 20, 30, 40, 50, 60,
        #               70, 80, 90, 100, 150, 200, 250, 300]
        n_dim_e_list=[20, 30, 40, 60, 80, 100]
        # n_dim_e_list=[10, 20, 30]
    )
    print("Cora")
    ablation_study(
        path="results",
        dataset_name="cora",
        init_lr_list=[0.05, 0.1, 0.2, 0.4, 0.8],
        # n_dim_e_list=[10, 20, 30, 40, 50, 60,
        #               70, 80, 90, 100, 150, 200, 250, 300]
        n_dim_e_list=[20, 30, 40, 60, 80, 100]
        # n_dim_e_list=[10, 20, 30]
    )
    print("PubMed")
    ablation_study(
        path="results",
        dataset_name="pubmed",
        init_lr_list=[0.05, 0.1, 0.2, 0.4, 0.8],
        # n_dim_e_list=[10, 20, 30, 40, 50, 60,
        #               70, 80, 90, 100, 150, 200, 250, 300]
        n_dim_e_list=[20, 30, 40, 60, 80, 100]
        # n_dim_e_list=[10, 20, 30]
    )
    print("Citeseer")
    ablation_study(
        path="results",
        dataset_name="citeseer",
        init_lr_list=[0.05, 0.1, 0.2, 0.4, 0.8],
        # n_dim_e_list=[10, 20, 30, 40, 50, 60,
        #               70, 80, 90, 100, 150, 200, 250, 300]
        n_dim_e_list=[20, 30, 40, 60, 80, 100]
        # n_dim_e_list=[10, 20, 30]
    )

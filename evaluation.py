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
        best_AUC = -1
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
            AUC = data["best_rocauc_valid_test"]
            print(n_dim_e)
            print(init_lr)
            print(effective_dim)
            print(AUC)
            if AUC > best_AUC:
                best_AUC = AUC
                best_effective_dim = effective_dim
        # print("n_dim_e:", n_dim_e)
        # print("effective_dim:", best_effective_dim)
        # print("AUC:", best_AUC)


if __name__ == '__main__':
    ablation_study(
        path="results",
        dataset_name="webkb",
        init_lr_list=[0.05, 0.1, 0.2, 0.4, 0.8],
        n_dim_e_list=[10, 20, 30, 40, 50, 60,
                      70, 80, 90, 100, 150, 200, 250, 300]
        # n_dim_e_list=[10, 20, 30]
    )

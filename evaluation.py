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
import matplotlib.pyplot as plt


def ablation_study(
    path,
    dataset_name,
    task,
    init_lr_list,
    n_dim_e_list
):
    prefix = path + "/MDL_WIPS_" + task + "_" + dataset_name
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


# def comparison(
#     path,
#     dataset_name,
#     n_dim_e,
#     task,
#     models,
#     init_lr_list,
#     neg_ratios
# ):
#     # basic information
#     print(dataset_name)
#     print(task)
#     print(n_dim_e)

#     MDL_WIPS_AUCs = []
#     WIPS_AUCs = []
#     SIPS_AUCs = []
#     IPDS_AUCs = []
#     IPS_AUCs = []
#     MDL_WIPS_AUCs.append([0, 0])
#     WIPS_AUCs.append([0, 0])
#     SIPS_AUCs.append([0, 0])
#     IPDS_AUCs.append([0, 0])
#     IPS_AUCs.append([0, 0])

#     for model in models:
#         print(model)
#         prefix = path + "/comparison_" + model + "_" + task + "_" + dataset_name
#         if model == "MDL_WIPS":
#             best_effective_dim = 0
#             best_AUC_valid = -1
#             best_AUC_test = -1
#             best_lr = 0
#             best_iter = 0
#             for init_lr in init_lr_list:
#                 filepath = prefix + "_" + \
#                     str(n_dim_e) + "_" + str(init_lr) + ".pth"

#                 data = torch.load(filepath)
#                 ROCAUC_valid_list = np.array(data["rocauc_valid_list"])
#                 ROCAUC_test_list = np.array(data["rocauc_test_list"])
#                 sparsity_list = np.array(data["sparsity_list"])

#                 # set required minimum sparsity
#                 required_sparsity = 0.05

#                 idx_sparse = np.where(sparsity_list >= required_sparsity)[0]
#                 ROCAUC_valid_list = ROCAUC_valid_list[idx_sparse]
#                 ROCAUC_test_list = ROCAUC_test_list[idx_sparse]
#                 sparsity_list = sparsity_list[idx_sparse]

#                 if len(sparsity_list) != 0:
#                     idx = np.argmax(ROCAUC_valid_list)
#                     AUC_valid = ROCAUC_valid_list[idx]
#                     AUC_test = ROCAUC_test_list[idx]
#                     effective_dim = int((1 - sparsity_list[idx]) * n_dim_e)
#                 else:
#                     AUC_valid = -1
#                     AUC_test = -1
#                     effective_dim = 0

#                 # print(data)
#                 # print(data["rocauc_train_list"])
#                 # print(data["rocauc_valid_list"])
#                 # print(data["rocauc_test_list"])
#                 # print(data["sparsity_list"])

#                 # ips_weight = data["best_rocauc_model"][
#                 #     "ips_weight"].data.cpu().numpy()
#                 # effective_dim = np.sum(ips_weight != 0)
#                 # AUC_valid = data["best_rocauc_valid"]
#                 # AUC_test = data["best_rocauc_valid_test"]

#                 # validation dataでのAUCが最もよいもので更新
#                 if AUC_valid > best_AUC_valid:
#                     best_AUC_valid = AUC_valid
#                     best_AUC_test = AUC_test
#                     best_effective_dim = effective_dim
#             print("n_dim_e:", n_dim_e)
#             print("effective_dim:", best_effective_dim)
#             print("AUC:", best_AUC_test)
#             MDL_WIPS_AUCs.append([best_effective_dim, best_AUC_test])
#         elif model == "IPDS":  # neg_ratio as a hyperparameter for IPDS
#             best_AUC_valid = -1
#             best_AUC_test = -1
#             best_lr = 0
#             best_iter = 0
#             for init_lr in init_lr_list:
#                 for neg_ratio in neg_ratios:
#                     filepath = prefix + "_" + \
#                         str(n_dim_e) + "_" + str(init_lr * 0.001) + \
#                         "_" + str(neg_ratio) + ".pth"
#                     data = torch.load(filepath)
#                     AUC_valid = data["best_rocauc_valid"]
#                     AUC_test = data["best_rocauc_valid_test"]
#                     if AUC_valid > best_AUC_valid:
#                         best_AUC_valid = AUC_valid
#                         best_AUC_test = AUC_test
#                         best_lr = init_lr
#                         best_iter = data["best_rocauc_valid_iteration"]
#             print("n_dim_e:", n_dim_e, best_lr, best_iter)
#             print("AUC:", best_AUC_test)
#             IPDS_AUCs.append([n_dim_e, best_AUC_test])
#         else:  # others
#             best_AUC_valid = -1
#             best_AUC_test = -1
#             best_lr = 0
#             best_iter = 0
#             for init_lr in init_lr_list:
#                 filepath = prefix + "_" + \
#                     str(n_dim_e) + "_" + str(init_lr * 0.001) + ".pth"
#                 data = torch.load(filepath)
#                 AUC_valid = data["best_rocauc_valid"]
#                 AUC_test = data["best_rocauc_valid_test"]
#                 if AUC_valid > best_AUC_valid:
#                     best_AUC_valid = AUC_valid
#                     best_AUC_test = AUC_test
#                     best_lr = init_lr
#                     best_iter = data["best_rocauc_valid_iteration"]
#             print("n_dim_e:", n_dim_e, best_lr, best_iter)
#             print("AUC:", best_AUC_test)
#             if model == "WIPS":
#                 WIPS_AUCs.append([n_dim_e, best_AUC_test])
#             elif model == "SIPS":
#                 SIPS_AUCs.append([n_dim_e, best_AUC_test])
#             elif model == "IPS":
#                 IPS_AUCs.append([n_dim_e, best_AUC_test])

#     MDL_WIPS_AUCs = np.array(MDL_WIPS_AUCs)
#     WIPS_AUCs = np.array(WIPS_AUCs)
#     SIPS_AUCs = np.array(SIPS_AUCs)
#     IPDS_AUCs = np.array(IPDS_AUCs)
#     IPS_AUCs = np.array(IPS_AUCs)

#     print(MDL_WIPS_AUCs)

#     # draw graphs
#     plt.clf()
#     plt.style.use('ggplot')
#     plt.xlabel('Dimensionality')
#     plt.ylabel('AUC')
#     plt.legend(loc='upper right')
#     plt.xlim(0, max(n_dim_e_list))
#     plt.ylim(0, 1)

#     plt.plot(WIPS_AUCs[0, :], WIPS_AUCs[1, :],
#              marker='o', color='blue', label='WIPS')
#     plt.plot(SIPS_AUCs[0, :], SIPS_AUCs[1, :],
#              marker='o', color='red', label='SIPS')
#     plt.plot(IPDS_AUCs[0, :], IPDS_AUCs[1, :],
#              marker='o', color='green', label='IPDS')
#     plt.plot(IPS_AUCs[0, :], IPS_AUCs[1, :],
#              marker='o', color='black', label='IPS')
#     plt.savefig("figs/" + dataset_name + "_" + task + ".jpg")

def comparison(
    path,
    dataset_name,
    n_dim_e_list,
    task,
    models,
    init_lr_list,
    neg_ratios
):
    # basic information
    print(dataset_name)
    print(task)
    # print(n_dim_e)

    MDL_WIPS_AUCs = []
    WIPS_AUCs = []
    SIPS_AUCs = []
    IPDS_AUCs = []
    IPS_AUCs = []
    # MDL_WIPS_AUCs.append([0, 0])
    WIPS_AUCs.append([0, 0])
    SIPS_AUCs.append([0, 0])
    IPDS_AUCs.append([0, 0])
    IPS_AUCs.append([0, 0])

    for n_dim_e in n_dim_e_list:

        for model in models:
            print(model)
            prefix = path + "/comparison_" + model + "_" + task + "_" + dataset_name
            if model == "MDL_WIPS":
                best_effective_dim = 0
                best_AUC_valid = -1
                best_AUC_test = -1
                best_lr = 0
                best_iter = 0
                for init_lr in init_lr_list:
                    filepath = prefix + "_" + \
                        str(n_dim_e) + "_" + str(init_lr) + ".pth"

                    data = torch.load(filepath)
                    ROCAUC_valid_list = np.array(data["rocauc_valid_list"])
                    ROCAUC_test_list = np.array(data["rocauc_test_list"])
                    sparsity_list = np.array(data["sparsity_list"])

                    # set required minimum sparsity
                    required_sparsity = 0.05

                    idx_sparse = np.where(
                        sparsity_list >= required_sparsity)[0]
                    ROCAUC_valid_list = ROCAUC_valid_list[idx_sparse]
                    ROCAUC_test_list = ROCAUC_test_list[idx_sparse]
                    sparsity_list = sparsity_list[idx_sparse]

                    if len(sparsity_list) != 0:
                        idx = np.argmax(ROCAUC_valid_list)
                        AUC_valid = ROCAUC_valid_list[idx]
                        AUC_test = ROCAUC_test_list[idx]
                        effective_dim = int((1 - sparsity_list[idx]) * n_dim_e)
                    else:
                        AUC_valid = -1
                        AUC_test = -1
                        effective_dim = 0

                    # print(data)
                    # print(data["rocauc_train_list"])
                    # print(data["rocauc_valid_list"])
                    # print(data["rocauc_test_list"])
                    # print(data["sparsity_list"])

                    # ips_weight = data["best_rocauc_model"][
                    #     "ips_weight"].data.cpu().numpy()
                    # effective_dim = np.sum(ips_weight != 0)
                    # AUC_valid = data["best_rocauc_valid"]
                    # AUC_test = data["best_rocauc_valid_test"]

                    # validation dataでのAUCが最もよいもので更新
                    if AUC_valid > best_AUC_valid:
                        best_AUC_valid = AUC_valid
                        best_AUC_test = AUC_test
                        best_effective_dim = effective_dim
                        best_lr = init_lr

                print("n_dim_e:", n_dim_e)
                print("best_lr:", best_lr)
                print("effective_dim:", best_effective_dim)
                print("AUC:", best_AUC_test)
                MDL_WIPS_AUCs.append([best_effective_dim, best_AUC_test])
            elif model == "IPDS":  # neg_ratio as a hyperparameter for IPDS
                best_AUC_valid = -1
                best_AUC_test = -1
                best_lr = 0
                best_iter = 0
                for init_lr in init_lr_list:
                    for neg_ratio in neg_ratios:
                        filepath = prefix + "_" + \
                            str(n_dim_e) + "_" + str(init_lr * 0.001) + \
                            "_" + str(neg_ratio) + ".pth"
                        data = torch.load(filepath)
                        AUC_valid = data["best_rocauc_valid"]
                        AUC_test = data["best_rocauc_valid_test"]
                        if AUC_valid > best_AUC_valid:
                            best_AUC_valid = AUC_valid
                            best_AUC_test = AUC_test
                            best_lr = init_lr
                            best_iter = data["best_rocauc_valid_iteration"]
                print("n_dim_e:", n_dim_e, best_lr, best_iter)
                print("AUC:", best_AUC_test)
                IPDS_AUCs.append([n_dim_e, best_AUC_test])
            else:  # others
                best_AUC_valid = -1
                best_AUC_test = -1
                best_lr = 0
                best_iter = 0
                for init_lr in init_lr_list:
                    filepath = prefix + "_" + \
                        str(n_dim_e) + "_" + str(init_lr * 0.001) + ".pth"
                    data = torch.load(filepath)
                    AUC_valid = data["best_rocauc_valid"]
                    AUC_test = data["best_rocauc_valid_test"]
                    if AUC_valid > best_AUC_valid:
                        best_AUC_valid = AUC_valid
                        best_AUC_test = AUC_test
                        best_lr = init_lr
                        best_iter = data["best_rocauc_valid_iteration"]
                print("n_dim_e:", n_dim_e, best_lr, best_iter)
                print("AUC:", best_AUC_test)
                if model == "WIPS":
                    WIPS_AUCs.append([n_dim_e, best_AUC_test])
                elif model == "SIPS":
                    SIPS_AUCs.append([n_dim_e, best_AUC_test])
                elif model == "IPS":
                    IPS_AUCs.append([n_dim_e, best_AUC_test])

    MDL_WIPS_AUCs = np.array(MDL_WIPS_AUCs)
    WIPS_AUCs = np.array(WIPS_AUCs)
    SIPS_AUCs = np.array(SIPS_AUCs)
    IPDS_AUCs = np.array(IPDS_AUCs)
    IPS_AUCs = np.array(IPS_AUCs)

    print(MDL_WIPS_AUCs)
    print(WIPS_AUCs)
    print(SIPS_AUCs)
    print(IPDS_AUCs)
    print(IPS_AUCs)

    # draw graphs
    fig = plt.figure(figsize=(7, 7))
    plt.clf()
    plt.style.use('ggplot')
    plt.xlabel('Dimensionality')
    plt.ylabel('AUC')
    plt.xlim(0, max(n_dim_e_list))
    plt.ylim(0.5, 1)

    plt.scatter(MDL_WIPS_AUCs[:, 0], MDL_WIPS_AUCs[:, 1],
                marker='o', color='black', label='WIPS')
    plt.plot(WIPS_AUCs[:, 0], WIPS_AUCs[:, 1],
             marker='o', color='blue', label='WIPS')
    plt.plot(SIPS_AUCs[:, 0], SIPS_AUCs[:, 1],
             marker='o', color='red', label='SIPS')
    plt.plot(IPDS_AUCs[:, 0], IPDS_AUCs[:, 1],
             marker='o', color='green', label='IPDS')
    plt.plot(IPS_AUCs[:, 0], IPS_AUCs[:, 1],
             marker='o', color='purple', label='IPS')

    plt.legend(loc='upper right')
    plt.savefig("figs/" + dataset_name + "_" + task + ".jpg")

if __name__ == '__main__':
    n_dim_e_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300]
    dataset_name_list = ["webkb", "cora", "citeseer"]
    init_lr_list = [0.4, 0.8, 1.6]
    neg_ratios = [0.01, 0.25, 0.50, 0.75, 0.99]
    task_list = ["linkpred", "nodeclf"]
    models = ["IPS", "SIPS", "IPDS", "WIPS", "MDL_WIPS"]
    # models = ["MDL_WIPS"]

    for task in task_list:
        # print(task)
        for dataset_name in dataset_name_list:
            # print(dataset_name)
            # for n_dim_e in n_dim_e_list:
            comparison(
                path="results",
                dataset_name=dataset_name,
                n_dim_e_list=n_dim_e_list,
                task=task,
                models=models,
                init_lr_list=init_lr_list,
                neg_ratios=neg_ratios
            )

    # print("n_dim_e = 100")
    # # comparison(
    # #     path="results",
    # #     dataset_name="webkb",
    # #     n_dim_e=100,
    # #     task="linkpred",
    # #     models=models,
    # #     init_lr_list=[0.4, 0.8, 1.6],
    # #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # # )
    # # comparison(
    # #     path="results",
    # #     dataset_name="cora",
    # #     n_dim_e=100,
    # #     task="linkpred",
    # #     models=models,
    # #     init_lr_list=[0.4, 0.8, 1.6],
    # #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # # )
    # # comparison(
    # #     path="results",
    # #     dataset_name="citeseer",
    # #     n_dim_e=100,
    # #     task="linkpred",
    # #     models=models,
    # #     init_lr_list=[0.4, 0.8, 1.6],
    # #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # # )
    # comparison(
    #     path="results",
    #     dataset_name="webkb",
    #     n_dim_e=100,
    #     task="nodeclf",
    #     models=models,
    #     init_lr_list=[0.4, 0.8, 1.6],
    #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # )
    # comparison(
    #     path="results",
    #     dataset_name="cora",
    #     n_dim_e=100,
    #     task="nodeclf",
    #     models=models,
    #     init_lr_list=[0.4, 0.8, 1.6],
    #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # )
    # comparison(
    #     path="results",
    #     dataset_name="citeseer",
    #     n_dim_e=100,
    #     task="nodeclf",
    #     models=models,
    #     init_lr_list=[0.4, 0.8, 1.6],
    #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # )
    # print("n_dim_e = 200")
    # # comparison(
    # #     path="results",
    # #     dataset_name="webkb",
    # #     n_dim_e=200,
    # #     task="linkpred",
    # #     models=models,
    # #     init_lr_list=[0.4, 0.8, 1.6],
    # #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # # )
    # # comparison(
    # #     path="results",
    # #     dataset_name="cora",
    # #     n_dim_e=200,
    # #     task="linkpred",
    # #     models=models,
    # #     init_lr_list=[0.4, 0.8, 1.6],
    # #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # # )
    # # comparison(
    # #     path="results",
    # #     dataset_name="citeseer",
    # #     n_dim_e=200,
    # #     task="linkpred",
    # #     models=models,
    # #     init_lr_list=[0.4, 0.8, 1.6],
    # #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # # )
    # comparison(
    #     path="results",
    #     dataset_name="webkb",
    #     n_dim_e=200,
    #     task="nodeclf",
    #     models=models,
    #     init_lr_list=[0.4, 0.8, 1.6],
    #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # )
    # comparison(
    #     path="results",
    #     dataset_name="cora",
    #     n_dim_e=200,
    #     task="nodeclf",
    #     models=models,
    #     init_lr_list=[0.4, 0.8, 1.6],
    #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # )
    # comparison(
    #     path="results",
    #     dataset_name="citeseer",
    #     n_dim_e=200,
    #     task="nodeclf",
    #     models=models,
    #     init_lr_list=[0.4, 0.8, 1.6],
    #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # )
    # print("n_dim_e = 300")
    # # comparison(
    # #     path="results",
    # #     dataset_name="webkb",
    # #     n_dim_e=300,
    # #     task="linkpred",
    # #     models=models,
    # #     init_lr_list=[0.4, 0.8, 1.6],
    # #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # # )
    # # comparison(
    # #     path="results",
    # #     dataset_name="cora",
    # #     n_dim_e=300,
    # #     task="linkpred",
    # #     models=models,
    # #     init_lr_list=[0.4, 0.8, 1.6],
    # #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # # )
    # # comparison(
    # #     path="results",
    # #     dataset_name="citeseer",
    # #     n_dim_e=300,
    # #     task="linkpred",
    # #     models=models,
    # #     init_lr_list=[0.4, 0.8, 1.6],
    # #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # # )
    # comparison(
    #     path="results",
    #     dataset_name="webkb",
    #     n_dim_e=300,
    #     task="nodeclf",
    #     models=models,
    #     init_lr_list=[0.4, 0.8, 1.6],
    #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # )
    # comparison(
    #     path="results",
    #     dataset_name="cora",
    #     n_dim_e=300,
    #     task="nodeclf",
    #     models=models,
    #     init_lr_list=[0.4, 0.8, 1.6],
    #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # )
    # comparison(
    #     path="results",
    #     dataset_name="citeseer",
    #     n_dim_e=300,
    #     task="nodeclf",
    #     models=models,
    #     init_lr_list=[0.4, 0.8, 1.6],
    #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # )
    # # comparison(
    # #     path="results",
    # #     dataset_name="amazon",
    # #     n_dim_e=300,
    # #     task="nodeclf",
    # #     models=models,
    # #     init_lr_list=[0.4, 0.8, 1.6],
    # #     neg_ratios=[0.01, 0.25, 0.50, 0.75, 0.99]
    # # )

    # print("WebKB")
    # ablation_study(
    #     path="results",
    #     dataset_name="webkb",
    #     init_lr_list=[0.4, 0.8, 1.6],
    #     task="linkpred",
    #     n_dim_e_list=[20, 30, 40, 60, 80, 100, 150, 200, 250, 300]
    # )
    # ablation_study(
    #     path="results",
    #     dataset_name="webkb",
    #     init_lr_list=[0.4, 0.8, 1.6],
    #     task="nodeclf",
    #     n_dim_e_list=[20, 30, 40, 60, 80, 100, 150, 200, 250, 300]
    # )
    # print("Cora")
    # ablation_study(
    #     path="results",
    #     dataset_name="cora",
    #     init_lr_list=[0.4, 0.8, 1.6],
    #     task="linkpred",
    #     n_dim_e_list=[20, 30, 40, 60, 80, 100, 150, 200, 250, 300]
    # )
    # ablation_study(
    #     path="results",
    #     dataset_name="cora",
    #     init_lr_list=[0.4, 0.8, 1.6],
    #     task="nodeclf",
    #     n_dim_e_list=[20, 30, 40, 60, 80, 100, 150, 200, 250, 300]
    # )
    # print("Citeseer")
    # ablation_study(
    #     path="results",
    #     dataset_name="citeseer",
    #     init_lr_list=[0.4, 0.8, 1.6],
    #     task="linkpred",
    #     n_dim_e_list=[20, 30, 40, 60, 80, 100, 150, 200, 250, 300]
    # )
    # ablation_study(
    #     path="results",
    #     dataset_name="citeseer",
    #     init_lr_list=[0.4, 0.8, 1.6],
    #     task="nodeclf",
    #     n_dim_e_list=[20, 30, 40, 60, 80, 100, 150, 200, 250, 300]
    # )

    # print("Cora")
    # ablation_study(
    #     path="results",
    #     dataset_name="cora",
    #     init_lr_list=[0.05, 0.1, 0.2, 0.4, 0.8],
    #     # n_dim_e_list=[10, 20, 30, 40, 50, 60,
    #     #               70, 80, 90, 100, 150, 200, 250, 300]
    #     n_dim_e_list=[20, 30, 40, 60, 80, 100]
    # )
    # # print("PubMed")
    # # ablation_study(
    # #     path="results",
    # #     dataset_name="pubmed",
    # #     # init_lr_list=[0.05, 0.1, 0.2, 0.4, 0.8],
    # #     init_lr_list=[0.2, 0.4, 0.8, 1.6, 3.2],
    # #     # n_dim_e_list=[10, 20, 30, 40, 50, 60,
    # #     #               70, 80, 90, 100, 150, 200, 250, 300]
    # #     # n_dim_e_list=[20, 30, 40, 60, 80, 100],
    # #     # n_dim_e_list=[20, 30, 40, 60, 80, 100, 150, 200, 250, 300]
    # #     n_dim_e_list=[150, 200, 250, 300]
    # # )
    # print("Citeseer")
    # ablation_study(
    #     path="results",
    #     dataset_name="citeseer",
    #     init_lr_list=[0.05, 0.1, 0.2, 0.4, 0.8],
    #     # n_dim_e_list=[10, 20, 30, 40, 50, 60,
    #     #               70, 80, 90, 100, 150, 200, 250, 300]
    #     # n_dim_e_list=[20, 30, 40, 60, 80, 100]
    #     n_dim_e_list=[20, 30, 40, 60, 80, 100, 150, 200, 250, 300]
    # )

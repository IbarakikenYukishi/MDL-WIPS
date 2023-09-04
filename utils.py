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


if __name__ == '__main__':
    main()

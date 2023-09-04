import os
import sys
import timeit
import numpy as np
import torch
import torch.multiprocessing as mp
import math
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim


class ProxSGD(optim.Optimizer):

    def __init__(
        self,
        params,
        lr,
        log_h_prime,
        batchsize,
        M,
        lambda_max,
        n_nodes,
        n_dim_e,
        n_dim_m,
        n_dim_d,
        device
    ):
        defaults = {
            "lr": lr,
            "log_h_prime": log_h_prime,
            "batchsize": batchsize,
            "M": M,
            "lambda_max": lambda_max,
            "n_nodes": n_nodes,
            "n_dim_e": n_dim_e,
            "n_dim_m": n_dim_m,
            "n_dim_d": n_dim_d,
            "alpha": torch.tensor(np.ones(n_dim_e) * 0.001),
            "device": device
        }
        super().__init__(params, defaults=defaults)

    def optimize_alpha(
        self
    ):
        group = self.param_groups[0]
        param_l = group["params"][0]
        # param_B = group["params"][1]
        param_A = group["params"][2]
        A_tilde = torch.cat(
            (param_l.reshape((-1, 1)), param_A), dim=1)
        # print(group["alpha"])
        for i in range(group["n_dim_e"]):
            print(torch.norm(A_tilde[i, :]))
            group["alpha"][i] = 4 * (group["n_dim_m"] + 1) / (4 * torch.norm(A_tilde[i, :]) + 2) + \
                torch.exp(- group["log_h_prime"] + 2 * np.log(
                    (group["n_dim_m"] + 1)) - 2 * torch.log(4 * torch.norm(A_tilde[i, :]) + 2))
        print(group["alpha"])

    def step(
        self
    ):
        group = self.param_groups[0]
        param_l = group["params"][0]
        param_B = group["params"][1]
        param_A = group["params"][2]

        def normalized_grad(grad, threshold):  # normalize grad not to diverge the parameters
            grad_norm = torch.norm(grad)
            grad_norm = torch.where(
                grad_norm > threshold, grad_norm, torch.tensor(threshold, device=grad.device))
            return grad / grad_norm

        threshold = 5.0
        l_grad = normalized_grad(param_l.grad.data, threshold)
        B_grad = normalized_grad(param_B.grad.data, threshold)
        A_grad = normalized_grad(param_A.grad.data, threshold)

        param_B_update = param_B.data - \
            (group["lr"] / group["batchsize"]) * B_grad
        param_l_update = param_l.data - \
            (group["lr"] / group["batchsize"]) * l_grad
        param_A_update = param_A.data - \
            (group["lr"] / group["batchsize"]) * A_grad

        # Proximal operator of group lasso
        A_tilde = torch.cat(
            (param_l_update.reshape((-1, 1)), param_A_update), dim=1)
        for i in range(group["n_dim_e"]):
            # old one
            # threshold = group["lr"] * group["alpha"][i]
            # new update rule
            threshold = group["lr"] * 2 * group["alpha"][i] / \
                (group["n_nodes"] * (group["n_nodes"] - 1))

            if torch.norm(A_tilde[i, :]) <= threshold:
                A_tilde[i, :] = 0
            else:
                A_tilde[i, :] -= threshold * \
                    A_tilde[i, :] / torch.norm(A_tilde[i, :])

        param_l_update = A_tilde[:, 0]
        param_A_update = A_tilde[:, 1:]

        def update_param(update, original, bound):
            is_nan_inf = torch.isnan(update) | torch.isinf(update)
            update = torch.where(is_nan_inf, original, update)
            update = torch.where(
                update > bound,
                torch.tensor(bound, device=update.device),
                update
            )
            update = torch.where(
                update < -bound,
                torch.tensor(-bound, device=update.device),
                update
            )
            return update

        param_l_update = update_param(
            param_l_update, param_l, group["lambda_max"])
        param_B_update = update_param(param_B_update, param_B, group["M"])
        param_A_update = update_param(param_A_update, param_A, group["M"])

        # # raw param
        # print("raw param")
        # print(param_l)
        # print(param_B)
        # print(param_A)

        # # grad
        # print("grad")
        # print(param_l.grad.data)
        # print(param_B.grad.data)
        # print(param_A.grad.data)

        # # update
        # print("update")
        # print(param_l_update)
        # print(param_B_update)
        # print(param_A_update)

        param_l.data.copy_(param_l_update)
        param_B.data.copy_(param_B_update)
        param_A.data.copy_(param_A_update)


def trainer(model, dataset, lossfn, optimizer, opt, log, cuda):
    print(dataset)

    # dataloader
    loader = DataLoader(
        dataset,
        batch_size=opt["batchsize"],
        collate_fn=dataset.collate,
        sampler=dataset.sampler
    )

    # (maximum ROCAUC, the iteratioin at which the maximum one was achieved)
    max_ROCAUC = (-1, -1)
    max_ROCAUC_model = None
    max_ROCAUC_model_on_test = None

    iter_counter = 0
    former_loss = np.Inf

    t_start = timeit.default_timer()

    assert opt["iter"] % opt["eval_each"] == 0
    pbar = tqdm(total=opt["eval_each"])

    while True:
        train_loss = []
        loss = None

        for inputs, targets in loader:
            pbar.update(1)

            # inputとtargetが何を表しているかわからない
            # link predictionかreconstructionかでも異なる気はする
            # print(inputs)
            # print(targets)
            if cuda:
                inputs = inputs.cuda()
                targets = targets.cuda()

            # update parameters until reaching to the predefined number of
            # iterations
            optimizer.zero_grad()
            preds = model(inputs)
            loss = lossfn(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            iter_counter += 1
            # optimize alpha for every x iteration
            if iter_counter % 10000 == 0:
                optimizer.optimize_alpha()

            # evaluate validation and test for each "eval_each" iterations.
            if iter_counter % opt["eval_each"] == 0:

                pbar.close()
                model.eval()
                ROCAUC_train, ROCAUC_valid, ROCAUC_test, eval_elapsed = evaluation(
                    model, dataset.neighbor_train, dataset.neighbor_valid, dataset.neighbor_test, dataset.task, log, opt["neproc"], cuda, True)
                model.train()

                # update maximum performance
                if ROCAUC_valid > max_ROCAUC[0]:
                    max_ROCAUC = (ROCAUC_valid, iter_counter)
                    max_ROCAUC_model = model.state_dict()
                    embeds = model.embed()
                    max_ROCAUC_model_embed = embeds
                    max_ROCAUC_model_on_test = ROCAUC_test

                log.info(
                    ('[%s] Eval: {'
                     '"iter": %d, '
                     '"loss": %.6f, '
                     '"elapsed (for %d iter.)": %.2f, '
                     '"elapsed (for eval.)": %.2f, '
                     '"rocauc_train": %.6f, '
                     '"rocauc_valid": %.6f, '
                     '"rocauc_test": %.6f, '
                     '"best_rocauc_valid": %.6f, '
                     '"best_rocauc_valid_iter": %d, '
                     '"best_rocauc_valid_test": %.6f, '
                     '}') % (
                         opt["exp_name"], iter_counter, np.mean(
                             train_loss), opt["eval_each"], timeit.default_timer() - t_start, eval_elapsed,
                         ROCAUC_train, ROCAUC_valid, ROCAUC_test, max_ROCAUC[
                             0],  max_ROCAUC[1], max_ROCAUC_model_on_test,
                    )
                )

                # no early stopping
                former_loss = np.mean(train_loss)
                train_loss = []
                t_start = timeit.default_timer()
                if iter_counter < opt["iter"]:
                    pbar = tqdm(total=opt["eval_each"])

            if iter_counter >= opt["iter"]:
                log.info(
                    ('[%s] RESULT: {'
                     '"best_rocauc_valid": %.6f, '
                     '"best_rocauc_valid_test": %.6f, '
                     '}') % (
                        opt["exp_name"],
                        max_ROCAUC[0], max_ROCAUC_model_on_test,
                    )
                )

                print(""" save the model """)
                embeds = model.embed()

                torch.save({
                    'model': model.state_dict(),
                    'node2id': dataset.node2id,
                    'data_vectors': dataset.data_vectors,
                    'embeds_at_final_iteration': embeds,
                    'best_rocauc_model': max_ROCAUC_model,
                    'best_rocauc_valid': max_ROCAUC[0],
                    'best_rocauc_valid_embeds': max_ROCAUC_model_embed,
                    'best_rocauc_valid_test': max_ROCAUC_model_on_test,
                    'best_rocauc_valid_iteration': max_ROCAUC[1],
                    'total_iteration': iter_counter,
                }, f'{opt["save_dir"]}/{opt["exp_name"]}.pth')
                sys.exit()


def evaluation(model, neighbor_train, neighbor_valid, neighbor_test, task, log, neproc, cuda=False, verbose=False):
    t_start = timeit.default_timer()

    ips_weight = None

    embeds = model.embed()
    if model.model == "WIPS" or model.model == "MDL_WIPS":
        ips_weight = model.get_ips_weight()
        # log.info("WIPS's ips weight's ratio : pos {}, neg {}".format(
        #     np.sum(ips_weight >= 0), np.sum(ips_weight < 0)))
        log.info("WIPS's ips weight's ratio : pos {}, zero {}, neg {}".format(
            np.sum(ips_weight > 0), np.sum(ips_weight == 0), np.sum(ips_weight < 0)))

    neighbor_train = list(neighbor_train.items())
    chunk = int(len(neighbor_train) / neproc + 1)
    queue = mp.Manager().Queue()
    processes = []
    for i in range(neproc):
        p = mp.Process(
            target=eval_thread,
            args=(neighbor_train[i * chunk:(i + 1) * chunk], model,
                  embeds, ips_weight, queue, cuda, i == 0 and verbose)
        )
        p.start()
        processes.append(p)
    rocauc_scores_train = list()
    for i in range(neproc):
        rocauc_score = queue.get()
        rocauc_scores_train += rocauc_score

    rocauc_scores_valid = rocauc_scores_train.copy()
    rocauc_scores_test = rocauc_scores_train.copy()

    if neighbor_valid is not None:
        neighbor_valid = list(neighbor_valid.items())
        chunk = int(len(neighbor_valid) / neproc + 1)
        queue = mp.Manager().Queue()
        processes = []
        for i in range(neproc):
            p = mp.Process(
                target=eval_thread,
                args=(neighbor_valid[i * chunk:(i + 1) * chunk], model,
                      embeds, ips_weight, queue, cuda, i == 0 and verbose)
            )
            p.start()
            processes.append(p)
        rocauc_scores_valid = list()
        for i in range(neproc):
            rocauc_score = queue.get()
            rocauc_scores_valid += rocauc_score

    if neighbor_test is not None:
        neighbor_test = list(neighbor_test.items())
        chunk = int(len(neighbor_test) / neproc + 1)
        queue = mp.Manager().Queue()
        processes = []
        for i in range(neproc):
            p = mp.Process(
                target=eval_thread,
                args=(neighbor_test[i * chunk:(i + 1) * chunk], model,
                      embeds, ips_weight, queue, cuda, i == 0 and verbose)
            )
            p.start()
            processes.append(p)
        rocauc_scores_test = list()
        for i in range(neproc):
            rocauc_score = queue.get()
            rocauc_scores_test += rocauc_score

    return np.mean(rocauc_scores_train), np.mean(rocauc_scores_valid), np.mean(rocauc_scores_test), timeit.default_timer() - t_start


def eval_thread(neighbor_thread, model, embeds, ips_weight, queue, cuda, verbose):
    embeds = [torch.from_numpy(i) for i in embeds]
    embeddings = []
    with torch.no_grad():
        for i in range(len(embeds)):
            embeddings.append(Variable(embeds[i]))
        if ips_weight is not None:
            ips_weight = Variable(torch.from_numpy(ips_weight))
    rocauc_scores = []
    if verbose:
        bar = tqdm(desc='Eval', total=len(neighbor_thread), mininterval=1,
                   bar_format='{desc}: {percentage:3.0f}% ({remaining} left)')
    for _s, s_neighbor in neighbor_thread:
        if verbose:
            bar.update()
        s = torch.tensor(_s)
        target_embeddings = []
        with torch.no_grad():
            for i in range(len(embeds)):
                target_embeddings.append(
                    Variable(embeds[i][s].expand_as(embeddings[i])))
        if cuda:
            input_embeddings = target_embeddings + embeddings
            if ips_weight is not None:
                _dists = model.distfn(
                    input_embeddings, w=ips_weight).data.cpu().numpy().flatten()
            else:
                _dists = model.distfn(
                    input_embeddings).data.cpu().numpy().flatten()
            node_num = model.total_node_num
        else:
            input_embeddings = target_embeddings + embeddings
            if ips_weight is not None:
                _dists = model.distfn(
                    input_embeddings, w=ips_weight).data.numpy().flatten()
            else:
                _dists = model.distfn(input_embeddings).data.numpy().flatten()
            node_num = model.total_node_num
        _dists[s] = 1e+12
        _labels = np.zeros(node_num)
        for o in s_neighbor:
            o = torch.tensor(o)
            _labels[o] = 1
        _rocauc_scores = roc_auc_score(_labels, -_dists)
        rocauc_scores.append(_rocauc_scores)
    if verbose:
        bar.close()
    queue.put(rocauc_scores)

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
from copy import deepcopy
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


os.environ["OMP_NUM_THREADS"] = "4"


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
            "alpha": torch.tensor(np.ones(n_dim_e) * 0.001).to(device)
        }
        super().__init__(params, defaults=defaults)

    def optimize_alpha(
        self,
        alpha_max=100000
    ):
        group = self.param_groups[0]
        param_l = group["params"][0]
        param_A = group["params"][2]
        A_tilde = torch.cat(
            (param_l.reshape((-1, 1)), param_A), dim=1)

        # first order approximation
        for i in range(group["n_dim_e"]):
            # print(torch.norm(A_tilde[i, :]))
            # group["alpha"][i] = 4 * (group["n_dim_m"] + 1) / (4 * torch.norm(A_tilde[i, :]) + 2) + \
            #     torch.exp(- group["log_h_prime"] + 2 * np.log(
            #         (group["n_dim_m"] + 1)) - 2 * torch.log(4 * torch.norm(A_tilde[i, :]) + 2))
            group["alpha"][i] = min((group["n_dim_m"] + 1) /
                                    (torch.norm(A_tilde[i, :]) + 0.0001), alpha_max)

        # print(group["alpha"])

    def upper_bound_on_PC(
        self
    ):
        group = self.param_groups[0]

        ret = 0
        # ret += 0.5 * torch.sum(group["alpha"])
        ret -= (group["n_dim_m"] + 1) * \
            torch.sum(torch.log(group["alpha"] + 0.00001))

        param_l = group["params"][0]
        param_A = group["params"][2]
        A_tilde = torch.cat(
            (param_l.reshape((-1, 1)), param_A), dim=1)

        ret += torch.sum(group["alpha"] *
                         torch.norm(A_tilde, dim=1)).cpu().item()

        return ret

    def regularization_term(
        self
    ):
        group = self.param_groups[0]

        ret = 0

        param_l = group["params"][0]
        param_A = group["params"][2]
        A_tilde = torch.cat(
            (param_l.reshape((-1, 1)), param_A), dim=1)

        ret += torch.sum(group["alpha"] *
                         torch.norm(A_tilde, dim=1)).cpu().item()

        return ret

    def step(
        self
    ):
        group = self.param_groups[0]
        param_l = group["params"][0]
        param_B = group["params"][1]
        param_A = group["params"][2]

        # normalize grad not to diverge the parameters
        def normalized_grad(grad, threshold):
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


def trainer_for_ablation(
    model,
    dataset,
    lossfn,
    optimizer,
    opt,
    log,
    cuda,
    required_sparsity=-1,
    node_clf=False,
    labels=None,
):
    # dataloader
    loader = DataLoader(
        dataset,
        batch_size=opt["batchsize"],
        collate_fn=dataset.collate,
        sampler=dataset.sampler
    )

    # (maximum ROCAUC, the iteration at which the maximum one was achieved)
    max_ROCAUC = (-1, -1)
    max_ROCAUC_model = None
    max_ROCAUC_model_on_test = -1
    max_ROCAUC_model_embed = None

    iter_counter = 0
    former_loss = np.Inf

    t_start = timeit.default_timer()

    assert opt["iter"] % opt["eval_each"] == 0
    pbar = tqdm(total=opt["eval_each"])

    while True:
        train_loss = []
        loss = None
        # uLNML_list = []
        lik_list = []

        for inputs, targets in loader:
            pbar.update(1)

            # if cuda:
            inputs = inputs.cuda(cuda)
            targets = targets.cuda(cuda)

            # update parameters until reaching to the predefined number of
            # iterations
            optimizer.zero_grad()
            preds = model(inputs)
            pos_loss, neg_loss = lossfn(preds, targets)
            loss = pos_loss + neg_loss

            # loss = lossfn(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            iter_counter += 1
            # optimize alpha for every x iteration
            lik_list.append(
                pos_loss.item() * opt["lik_pos_ratio"] + neg_loss.item() * opt["lik_neg_ratio"] + optimizer.regularization_term())
            # uLNML_list.append(pos_loss.item() * opt["lik_pos_ratio"] + neg_loss.item() *
            #                   opt["lik_neg_ratio"] + optimizer.upper_bound_on_PC().item())
            # uLNML.append(optimizer.upper_bound_on_PC().item())

            if iter_counter % 10000 == 0:
                optimizer.optimize_alpha()

            # evaluate validation and test for each "eval_each" iterations.
            if iter_counter % opt["eval_each"] == 0:
                # print(np.mean(lik_list))
                # print(np.mean(uLNML_list))
                # uLNML_list = []
                lik_list = []

                pbar.close()
                model.eval()
                # ROCAUC_train, ROCAUC_valid, ROCAUC_test, eval_elapsed = evaluation(
                # model, dataset.neighbor_train, dataset.neighbor_valid,
                # dataset.neighbor_test, dataset.task, log, opt["neproc"],
                # cuda, True)
                if opt["task"]=="nodeclf":
                    ROCAUC_train, ROCAUC_valid, ROCAUC_test, eval_elapsed = evaluation_classification(
                        model,
                        labels,
                        dataset.train_ids,
                        dataset.valid_ids,
                        dataset.test_ids,
                        log
                    )
                else:
                    ROCAUC_train, ROCAUC_valid, ROCAUC_test, eval_elapsed = evaluation(
                        model,
                        dataset.neighbor_train,
                        dataset.neighbor_valid,
                        dataset.neighbor_test,
                        dataset.task,
                        log,
                        opt["neproc"],
                        True,
                        True
                    )
                model.train()

                # update maximum performance
                # if ROCAUC_valid > max_ROCAUC[0]:
                #     max_ROCAUC = (ROCAUC_valid, iter_counter)
                #     max_ROCAUC_model = model.state_dict()
                #     embeds = model.embed()
                #     max_ROCAUC_model_embed = embeds
                #     max_ROCAUC_model_on_test = ROCAUC_test

                ips_weight = model.get_ips_weight()
                sparsity = np.sum(ips_weight == 0) / len(ips_weight)

                if sparsity > required_sparsity and ROCAUC_valid > max_ROCAUC[0]:
                    max_ROCAUC = (ROCAUC_valid, iter_counter)
                    max_ROCAUC_model = deepcopy(model.state_dict())
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
                         opt["exp_name"],
                         iter_counter,
                         np.mean(train_loss),
                         opt["eval_each"],
                         timeit.default_timer() - t_start,
                         eval_elapsed,
                         ROCAUC_train,
                         ROCAUC_valid,
                         ROCAUC_test,
                         max_ROCAUC[0],
                         max_ROCAUC[1],
                         max_ROCAUC_model_on_test,
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
                        max_ROCAUC[0],
                        max_ROCAUC_model_on_test,
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
                    'opt': opt
                }, f'{opt["save_dir"]}/{opt["exp_name"]}.pth')
                sys.exit()


def trainer(
    model,
    dataset,
    lossfn,
    optimizer,
    opt,
    log,
    cuda
):
    # print(dataset)

    # dataloader
    loader = DataLoader(
        dataset,
        batch_size=opt["batchsize"],
        collate_fn=dataset.collate,
        sampler=dataset.sampler
    )

    # (maximum ROCAUC, the iteration at which the maximum one was achieved)
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
                    max_ROCAUC_model = deepcopy(model.state_dict())
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


def evaluation_classification(
    model,
    labels,
    train_ids,
    valid_ids,
    test_ids,
    log
):
    t_start = timeit.default_timer()

    ips_weight = None

    embeds = model.embed()
    if model.model == "WIPS" or model.model == "MDL_WIPS":
        ips_weight = model.get_ips_weight()
        # log.info("WIPS's ips weight's ratio : pos {}, neg {}".format(
        #     np.sum(ips_weight >= 0), np.sum(ips_weight < 0)))
        log.info("WIPS's ips weight's ratio : pos {}, zero {}, neg {}".format(
            np.sum(ips_weight > 0), np.sum(ips_weight == 0), np.sum(ips_weight < 0)))


    embeds = model.embed()[0]

    train_embeds = embeds[train_ids, :]
    valid_embeds = embeds[valid_ids, :]
    test_embeds = embeds[test_ids, :]

    train_labels = labels[train_ids]
    valid_labels = labels[valid_ids]
    test_labels = labels[test_ids]

    # lr = LogisticRegression(max_iter=10000, C=1e5)
    lr = LogisticRegressionCV(cv=5, max_iter=10000)
    # lr = lr.fit(embeds, labels)
    lr.fit(train_embeds, train_labels)
    # print(lr.predict(train_embeds))

    f1_micro_train = f1_score(
        y_true=train_labels, y_pred=lr.predict(train_embeds), average="micro")
    f1_micro_valid = f1_score(
        y_true=valid_labels, y_pred=lr.predict(valid_embeds), average="micro")
    f1_micro_test = f1_score(
        y_true=test_labels, y_pred=lr.predict(test_embeds), average="micro")

    f1_macro_train = f1_score(
        y_true=train_labels, y_pred=lr.predict(train_embeds), average="macro")
    f1_macro_valid = f1_score(
        y_true=valid_labels, y_pred=lr.predict(valid_embeds), average="macro")
    f1_macro_test = f1_score(
        y_true=test_labels, y_pred=lr.predict(test_embeds), average="macro")

    # r_f1_micro_train = f1_score(y_true=train_labels, y_pred=np.random.randint(
    #     5, size=len(train_embeds)), average="micro")
    # r_f1_micro_valid = f1_score(y_true=valid_labels, y_pred=np.random.randint(
    #     5, size=len(valid_embeds)), average="micro")
    # r_f1_micro_test = f1_score(y_true=test_labels, y_pred=np.random.randint(
    #     5, size=len(test_embeds)), average="micro")

    # r_f1_macro_train = f1_score(y_true=train_labels, y_pred=np.random.randint(
    #     5, size=len(train_embeds)), average="macro")
    # r_f1_macro_valid = f1_score(y_true=valid_labels, y_pred=np.random.randint(
    #     5, size=len(valid_embeds)), average="macro")
    # r_f1_macro_test = f1_score(y_true=test_labels, y_pred=np.random.randint(
    #     5, size=len(test_embeds)), average="macro")

    # valid_pred = lr.predict(valid_embeds)
    # valid_pred = lr.predict(valid_embeds)

    # print('accuracy = ', accuracy_score(y_true=valid_labels, y_pred=valid_pred))

    print("inductive setting")
    print(f1_micro_train)
    print(f1_micro_valid)
    print(f1_micro_test)

    print(f1_macro_train)
    print(f1_macro_valid)
    print(f1_macro_test)

    # print(r_f1_micro_train)
    # print(r_f1_micro_valid)
    # print(r_f1_micro_test)

    # print(r_f1_macro_train)
    # print(r_f1_macro_valid)
    # print(r_f1_macro_test)

    # lr = LogisticRegression(max_iter=10000, C=1e5)
    # lr = LogisticRegressionCV(cv=5, max_iter=10000)

    # # lr = lr.fit(embeds, labels)
    # lr.fit(train_embeds[:int(len(train_embeds) * 0.8)],
    #        train_labels[:int(len(train_embeds) * 0.8)])
    # f1_micro_train = f1_score(
    #     y_true=train_labels[:int(len(train_embeds) * 0.8)], y_pred=lr.predict(train_embeds[:int(len(train_embeds) * 0.8)]), average="micro")
    # f1_macro_train = f1_score(
    #     y_true=train_labels[:int(len(train_embeds) * 0.8)], y_pred=lr.predict(train_embeds[:int(len(train_embeds) * 0.8)]), average="macro")

    # f1_micro_test = f1_score(
    #     y_true=train_labels[int(len(train_embeds) * 0.8):], y_pred=lr.predict(train_embeds[int(len(train_embeds) * 0.8):]), average="micro")
    # f1_macro_test = f1_score(
    #     y_true=train_labels[int(len(train_embeds) * 0.8):], y_pred=lr.predict(train_embeds[int(len(train_embeds) * 0.8):]), average="macro")
    # print("transductive")
    # print(f1_micro_train)
    # print(f1_micro_test)
    # print(f1_macro_train)
    # print(f1_macro_test)

    return f1_micro_train, f1_micro_valid, f1_micro_test, timeit.default_timer() - t_start


def evaluation(
    model,
    neighbor_train,
    neighbor_valid,
    neighbor_test,
    task,
    log,
    neproc,
    cuda=False,
    verbose=False
):
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

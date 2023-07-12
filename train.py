import datetime
import gc
import json
import logging
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model.CountingNet import EdgeMean
from motif_processor import QueryPreProcessing, Queryset
from utils.graph_operator import load_graph, create_batch, \
    k_hop_induced_subgraph_edge, random_walk_on_subgraph_edge
from utils.utils import wasserstein_loss

warnings.filterwarnings("ignore")
INF = float("inf")

train_config = {
    "max_npv": 4,  # max_number_pattern_vertices: 8, 16, 32
    "max_npe": 4,  # max_number_pattern_edges: 8, 16, 32
    "max_npel": 4,  # max_number_pattern_edge_labels: 8, 16, 32

    "max_ngv": 64,  # max_number_graph_vertices: 64, 512,4096
    "max_nge": 256,  # max_number_graph_edges: 256, 2048, 16384
    "max_ngel": 8,  # max_number_graph_edge_labels: 16, 64, 256

    "base": 2,
    "cuda": True,
    "gpu_id": 0,
    "num_workers": 12,

    "epochs": 200,
    "batch_size": 16,
    "update_every": 1,  # actual batch_sizer = batch_size * update_every
    "print_every": 10,
    "init_emb": True,  # None, Normal
    "share_emb": False,  # sharing embedding requires the same vector length
    "share_arch": False,  # sharing architectures
    "dropout": 0.4,
    "dropatt": 0.2,
    "cv": False,

    "reg_loss": "HUBER",  # MAE, MSEl
    "bp_loss": "HUBER",  # MAE, MSE
    "bp_loss_slp": "anneal_cosine$1.0$0.01",  # 0, 0.01, logistic$1.0$0.01, linear$1.0$0.01, cosine$1.0$0.01,
    # cyclical_logistic$1.0$0.01, cyclical_linear$1.0$0.01, cyclical_cosine$1.0$0.01
    # anneal_logistic$1.0$0.01, anneal_linear$1.0$0.01, anneal_cosine$1.0$0.01
    "lr": 0.0004,
    "weight_decay": 0.0005,
    "weight_decay_var": 0.1,
    "weight_decay_film": 0.0001,
    "decay_factor": 0.7,
    "decay_patience": 20,
    "max_grad_norm": 8,

    "model": "EDGEMEAN",  # CNN, RNN, TXL, RGCN, RGIN, RSIN
    "motif_net": "NNGINConcat",
    "graph_net": "GIN",
    "emb_dim": 128,
    "activation_function": "relu",  # sigmoid, softmax, tanh, relu, leaky_relu, prelu, gelu

    "predict_net": "FilmSumPredictNet",  # MeanPredictNet, SumPredictNet, MaxPredictNet,
    # MeanAttnPredictNet, SumAttnPredictNet, MaxAttnPredictNet,
    # MeanMemAttnPredictNet, SumMemAttnPredictNet, MaxMemAttnPredictNet,
    # DIAMNet
    "predict_net_add_enc": False,
    "predict_net_add_degree": False,
    "predict_net_hidden_dim": 128,
    "predict_net_num_heads": 4,
    "mem_len": 1,
    "predict_net_mem_init": "mean",
    # mean, sum, max, attn, circular_mean, circular_sum, circular_max, circular_attn, lstm
    "predict_net_recurrent_steps": 3,

    "rgin_num_bases": 8,
    "rgin_regularizer": "bdd",  # basis, bdd
    "rgin_graph_num_layers": 3,
    "rgin_pattern_num_layers": 3,
    "rgin_hidden_dim": 128,

    "ppn_num_bases": 8,
    "ppn_graph_num_layers": 3,
    "ppn_pattern_num_layers": 3,
    "ppn_hidden_dim": 24,

    "edgemean_num_bases": 8,
    "edgemean_graph_num_layers": 3,
    "edgemean_pattern_num_layers": 3,
    "edgemean_hidden_dim": 64,

    "num_g_hid": 128,
    "num_e_hid": 128,
    "out_g_ch": 128,

    "graph_num_layers": 3,

    "queryset_dir": "queryset",
    "true_card_dir": "label",
    "dataset": "intel",
    "data_dir": "dataset",
    "dataset_name": "intel.txt",
    "save_res_dir": "result",
    "save_model_dir": "saved_model",
    'init_g_dim': 1,

    "test_only": False,
    "GSL": True

}


def data_graph_transform(data_dir, dataset, dataset_name, emb=None):
    graph = load_graph(os.path.join(data_dir, dataset, dataset_name),
                       emb=emb)
    candidate_sets = {}
    # for node in range(graph.number_of_nodes()):
    #     subgraph = k_hop_induced_subgraph(graph, node)
    #     candidate_sets[node] = random_walk_on_subgraph(subgraph, node)
    cnt = 0
    for edge in graph.edges(data=True):
        subgraph = k_hop_induced_subgraph_edge(graph, edge)
        if train_config['GSL']:
            candidate_sets[cnt] = subgraph
        else:
            candidate_sets[cnt] = random_walk_on_subgraph_edge(subgraph, edge)
        cnt += 1
    batch = create_batch(graph, candidate_sets, emb=emb, edge_base=True)
    return batch


def train(model, optimizer, scheduler, data_type, data_loader, device, config, epoch, graph, logger=None, writer=None,
          bottleneck=False):
    global bp_crit, reg_crit
    epoch_step = len(data_loader)
    total_step = config["epochs"] * epoch_step
    total_var_loss = 0
    total_reg_loss = 0
    total_bp_loss = 0
    total_cnt = 1e-6
    if config["reg_loss"] == "MAE":
        reg_crit = lambda pred, target: F.l1_loss(F.relu(pred), target)
    elif config["reg_loss"] == "MSE":
        reg_crit = lambda pred, target: F.mse_loss(F.relu(pred), target)
    elif config["reg_loss"] == "SMSE":
        reg_crit = lambda pred, target: F.smooth_l1_loss(pred, target)
    elif config['reg_loss'] == "HUBER":
        reg_crit = lambda pred, target: F.huber_loss(pred, target, delta=0.1)
    if config["bp_loss"] == "MAE":
        bp_crit = lambda pred, target: F.l1_loss(F.leaky_relu(pred), target)
    elif config["bp_loss"] == "MSE":
        bp_crit = lambda pred, target: F.mse_loss(pred, target)
    elif config["bp_loss"] == "SMSE":
        bp_crit = lambda pred, target: F.smooth_l1_loss(pred, target)
    elif config['bp_loss'] == "HUBER":
        bp_crit = lambda pred, target: F.huber_loss(pred, target, delta=0.1)
    # data preparation
    # config['init_pe_dim'] = graph.edge_attr.size(1)
    if bottleneck:
        model.load_state_dict(
            torch.load(os.path.join(save_model_dir, 'best_epoch_{:s}_{:s}.pt'.format(train_config['predict_net'],
                                                                                     train_config['graph_net']))))
    model.to(device)
    model.train()
    total_time = 0
    for i, batch in enumerate(data_loader):
        motif_x, motif_edge_index, motif_edge_attr, card, var = batch
        # if config['cuda']:
        #     motif_x, motif_edge_index, motif_edge_attr = \
        #         _to_cuda(motif_x), _to_cuda(motif_edge_index), _to_cuda(motif_edge_attr)
        #     card, var = card.cuda(), var.cuda()
        s = time.time()
        # motif_x = motif_x.cuda(0)
        # motif_edge_index = motif_edge_index.cuda(0)
        # motif_edge_attr = motif_edge_attr.cuda(0)
        # y = val_to_distribution(card, var)
        card = card.cuda()
        var = var.cuda()
        # y = y.cuda(0)
        # graph = graph.cuda(0)
        # if config['cuda']:
        #     motif_x, motif_edge_index, motif_edge_attr = \
        #         _to_cuda(motif_x), _to_cuda(motif_edge_index), _to_cuda(motif_edge_attr)
        #     card, var = card.cuda(), var.cuda()
        if config['predict_net'].startswith("Film"):
            # pred, pred_var, filmreg = model(motif_x, motif_edge_index, motif_edge_attr, graph)
            distribution, filmreg = model(motif_x, motif_edge_index, motif_edge_attr, graph)
            # pred_var += 1e-10
            # y_pred = val_to_distribution(pred.detach().cpu(), pred_var.detach().cpu()).cuda(0)
            # y_pred = torch.tensor(y_pred.view(-1, 1), requires_grad=True)
            # bp_loss = (1 - config['weight_decay_var']) * bp_crit(pred, card) + config[
            #     'weight_decay_var'] * bp_crit(
            #     pred_var, var) + train_config[
            #               "weight_decay_film"] * filmreg
            bp_loss = wasserstein_loss(torch.distributions.Normal(loc=card, scale=torch.sqrt(var)), distribution) \
                      + train_config["weight_decay_film"] * filmreg
        else:
            distribution = model(motif_x, motif_edge_index, motif_edge_attr, graph)
            # pred, pred_var = model(motif_x, motif_edge_index, motif_edge_attr, graph)
            # y_pred = val_to_distribution(pred.detach().cpu(), pred_var.detach().cpu()).cuda(0)
            # y_pred = torch.tensor(y_pred.view(-1, 1), requires_grad=True)
            # bp_loss = (1 - config['weight_decay_var']) * bp_crit(pred, card) + config['weight_decay_var'] * bp_crit(
            #     pred_var, var)
            # bp_loss = -distribution.log_prob(card).mean()
            bp_loss = wasserstein_loss(torch.distributions.Normal(loc=card, scale=torch.sqrt(var)), distribution)
        reg_loss = bp_loss

        # if isinstance(config["bp_loss_slp"], (int, float)):
        #     neg_slp = float(config["bp_loss_slp"])
        # else:
        #     bp_loss_slp, l0, l1 = config["bp_loss_slp"].rsplit("$", 3)
        #     neg_slp = anneal_fn(bp_loss_slp, i + epoch * epoch_step, T=total_step // 4, lambda0=float(l0),
        #                         lambda1=float(l1))
        # bp_loss = (1 - config['weight_decay_var']) * bp_crit(pred, card) + config[
        #     'weight_decay_var'] * bp_crit(
        #     pred_var, var) + train_config[
        #               "weight_decay_film"] * filmreg
        # filmloss=(torch.sum(alpha ** 2)) ** 0.5 + (torch.sum(beta ** 2)) ** 0.5
        # bp_loss+=0.00001*filmloss
        # float
        bp_loss.backward()
        # var_loss_item = bp_crit(pred_var, var).item()
        reg_loss_item = reg_loss.item()
        bp_loss_item = bp_loss.item()
        # total_var_loss += var_loss_item
        total_reg_loss += reg_loss_item
        total_bp_loss += bp_loss_item

        if writer:
            writer.add_scalar("%s/REG-%s" % (data_type, config["reg_loss"]), reg_loss_item,
                              epoch * epoch_step + i)
            writer.add_scalar("%s/BP-%s" % (data_type, config["bp_loss"]), bp_loss_item, epoch * epoch_step + i)

        if logger and ((i / config['batch_size']) % config["print_every"] == 0 or i == epoch_step - 1):
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\treg loss: {:.5f}\tbp loss: {:.5f}\t"
                # "ground: {:.3f}\tpredict: {:.3f}"
                .format(
                    int(epoch), int(config["epochs"]), data_type, int(i / config['batch_size']),
                    int(epoch_step / config['batch_size']),
                    float(reg_loss_item), float(bp_loss_item)))
            # float(var), float(pred_var[0].item())))

        if (i + 1) % config["batch_size"] == 0 or i == epoch_step - 1:
            if config["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            optimizer.step()
            optimizer.zero_grad()
        e = time.time()
        total_time += e - s
        total_cnt += 1
    # mean_var_loss = total_var_loss / total_cnt
    # mean_reg_loss = total_reg_loss / total_cnt
    # mean_bp_loss = total_bp_loss / total_cnt
    mean_bp_loss = total_bp_loss / total_cnt
    mean_reg_loss = total_reg_loss / total_cnt
    # mean_var_loss = total_var_loss / total_cnt
    if writer:
        writer.add_scalar("%s/REG-%s-epoch" % (data_type, config["reg_loss"]), mean_reg_loss, epoch)
        writer.add_scalar("%s/BP-%s-epoch" % (data_type, config["bp_loss"]), mean_bp_loss, epoch)
        # writer.add_scalar("%s/Var-%s-epoch" % (data_type, config['bp_loss']), mean_var_loss, epoch)
    if logger:
        logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\treg loss: {:.4f}\tbp loss: {:.4f}".format(
            epoch, config["epochs"], data_type, mean_reg_loss, mean_bp_loss))

    gc.collect()
    return mean_reg_loss, mean_bp_loss, total_time


def evaluate(model, data_type, data_loader, config, graph, logger=None, writer=None):
    epoch_step = len(data_loader)
    total_step = config["epochs"] * epoch_step
    total_var_loss = 0
    total_reg_loss = 0
    total_bp_loss = 0
    total_cnt = 1e-6

    evaluate_results = {"mean": {"count": list(), "pred_mean": list()},
                        "var": {"var_t": list(), "pred_var": list()},
                        "error": {"mae": 0.0, "mse": 0.0},
                        "time": {"avg": list(), "total": 0.0}}

    if config["reg_loss"] == "MAE":
        reg_crit = lambda pred, target: F.l1_loss(F.relu(pred), target)
    elif config["reg_loss"] == "MSE":
        reg_crit = lambda pred, target: F.mse_loss(pred, target)
    elif config["reg_loss"] == "SMSE":
        reg_crit = lambda pred, target: F.smooth_l1_loss(pred, target)
    elif config["reg_loss"] == "MAEMSE":
        reg_crit = lambda pred, target: F.mse_loss(F.relu(pred), target) + \
                                        F.l1_loss(F.relu(pred), target)
    elif config['reg_loss'] == "HUBER":
        reg_crit = lambda pred, target: F.huber_loss(F.relu(pred), target)
    else:
        raise NotImplementedError

    if config["bp_loss"] == "MAE":
        bp_crit = lambda pred, target: F.l1_loss(F.leaky_relu(pred), target)
    elif config["bp_loss"] == "MSE":
        bp_crit = lambda pred, target: F.mse_loss(pred, target)
    elif config["bp_loss"] == "SMSE":
        bp_crit = lambda pred, target: F.smooth_l1_loss(pred, target)
    elif config["bp_loss"] == "MAEMSE":
        bp_crit = lambda pred, target: F.mse_loss(F.leaky_relu(pred), target) \
                                       + F.l1_loss(F.leaky_relu(pred), target)
    elif config['bp_loss'] == "HUBER":
        bp_crit = lambda pred, target: F.huber_loss(pred, target)
    else:
        raise NotImplementedError
    # reg_crit_mean = lambda pred, target: F.l1_loss(F.leaky_relu(pred), target)
    # reg_crit_var = lambda pred, target: F.mse_loss(F.leaky_relu(pred), target)
    # bp_crit_mean = lambda pred, target: F.l1_loss(F.leaky_relu(pred), target)
    # bp_crit_var = lambda pred, target: F.mse_loss(F.leaky_relu(pred), target)
    model.eval()
    total_time = 0
    with torch.no_grad():
        for batch_id, batch in enumerate(data_loader):
            motif_x, motif_edge_index, motif_edge_attr, card, var = batch
            # y = val_to_distribution(card, var)
            card = card.cpu()
            var = var.cpu()
            evaluate_results["mean"]["count"].extend(card.view(-1).tolist())
            evaluate_results["var"]["var_t"].extend(var.view(-1).tolist())
            if config['predict_net'].startswith("Film"):
                st = time.time()
                # pred, pred_var, filmreg = model(motif_x, motif_edge_index, motif_edge_attr, graph)
                distribution, filmreg = model(motif_x, motif_edge_index, motif_edge_attr, graph)
                distribution_cpu = torch.distributions.Normal(distribution.mean.cpu(), distribution.stddev.cpu())
                # pred = pred.cpu()
                # pred_var = pred_var.cpu()
                # pred += 1e-10
                # pred_var += 1e-10
                # y_pred = val_to_distribution(pred, pred_var).cpu()
                # y_pred = torch.tensor(y_pred.view(-1, 1), requires_grad=True)
                # bp_loss = bp_crit(pred, card) \
                #           + config['weight_decay_var'] * bp_crit(pred_var, var) \
                #           + train_config["weight_decay_film"] * filmreg
                gt_distribution = torch.distributions.Normal(card, torch.sqrt(var))
                bp_loss = wasserstein_loss(gt_distribution, distribution_cpu) + train_config[
                    "weight_decay_film"] * filmreg.cpu()  # -distribution_cpu.log_prob(card).mean()
            else:
                st = time.time()
                distribution = model(motif_x, motif_edge_index, motif_edge_attr, graph)
                distribution_cpu = torch.distributions.Normal(distribution.mean.cpu(), distribution.stddev.cpu())
                # y_pred = val_to_distribution(pred, pred_var).cpu()
                # bp_loss = (1 - config['weight_decay_var']) * bp_crit(pred, card) + config['weight_decay_var'] * bp_crit(
                #     pred_var, var)
                #
                bp_loss = wasserstein_loss(gt_distribution, distribution_cpu)
            et = time.time()
            # pred,alpha,beta = model(pattern, pattern_len, pattern_e_len, graph, graph_len, graph_e_len)
            evaluate_results["time"]["total"] += (et - st)
            avg_t = (et - st)

            evaluate_results["time"]["avg"].extend([avg_t])
            pred = distribution.mean
            pred_var = distribution.variance
            evaluate_results["mean"]["pred_mean"].extend(pred.view(-1).tolist())
            evaluate_results['var']['pred_var'].extend(pred_var.view(-1).tolist())
            reg_loss = bp_loss
            # reg_loss = (1 - config['weight_decay_var']) * reg_crit(pred, card) \
            #            + config['weight_decay_var'] * reg_crit(pred_var, var)
            reg_loss_item = reg_loss.mean().item()
            # var_loss_item = bp_crit(pred_var, var).item()
            bp_loss_item = bp_loss.mean().item()
            total_reg_loss += reg_loss_item
            total_bp_loss += bp_loss_item
            # total_var_loss += var_loss_item
            # evaluate_results["error"]["mae"] += F.l1_loss(F.relu(pred), card).sum().item()
            # evaluate_results["error"]["mse"] += F.mse_loss(F.relu(pred), card).sum().item()
            et = time.time()
            total_time += et - st
            total_cnt += 1
        mean_bp_loss = total_bp_loss / total_cnt
        mean_reg_loss = total_reg_loss / total_cnt
        mean_var_loss = total_var_loss / total_cnt
        if logger and batch_id == epoch_step - 1 and config['test_only'] is False:
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:d}/{:d}\treg loss: {:.4f}\tbp loss: {:.4f}\t".format(
                    int(epoch), int(config["epochs"]), (data_type), int(batch_id / config['batch_size']),
                    int(epoch_step / config['batch_size']),
                    float(reg_loss_item), float(bp_loss_item)))
            # float(var), float(pred_var[0].item())))
            # "ground: {:.4f}\tpredict: {:.4f}"

        if logger and config['test_only'] is False:
            logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\treg loss: {:.4f}\tbp loss: {:.4f}".format(
                epoch, config["epochs"], data_type, mean_reg_loss, mean_bp_loss))

        # evaluate_results["error"]["mae"] = evaluate_results["error"]["mae"] / total_cnt
        # evaluate_results["error"]["mse"] = evaluate_results["error"]["mse"] / total_cnt

    gc.collect()
    return mean_reg_loss, mean_bp_loss, mean_var_loss, evaluate_results, total_time


def test(save_model_dir, test_loaders, config, graph, logger, writer):
    # global loader_idx, mean_reg_loss, mean_bp_loss, mean_var_loss, evaluate_results, _time, f
    total_test_time = 0
    model.load_state_dict(torch.load(
        os.path.join(save_model_dir,
                     'best_epoch_{:s}_{:s}_edge_bce.pt'.format(config['predict_net'], config['graph_net']))))
    # print(model)
    mean_reg_loss, mean_bp_loss, mean_var_loss, evaluate_results, _time = evaluate(model=model, data_type="test",
                                                                                   data_loader=test_loaders,
                                                                                   config=config,
                                                                                   graph=graph,
                                                                                   logger=logger, writer=writer)
    total_test_time += _time
    # if mean_reg_loss <= best_reg_losses['test']:
    #     best_reg_losses['test'] = mean_reg_loss
    # best_reg_epochs['test'] =
    logger.info(
        "data_type: {:<5s}\tbest mean loss: {:.3f}".format("test", mean_reg_loss))
    with open(os.path.join(save_model_dir,
                           '%s_%s_%s_%s_edge_bce.json' % (
                                   train_config['predict_net'], train_config['graph_net'], "best_test",
                                   train_config['dataset']
                           )), "w") as f:
        json.dump(evaluate_results, f)

    return evaluate_results, total_test_time


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    for i in range(1, len(sys.argv), 2):
        arg = sys.argv[i]
        value = sys.argv[i + 1]

        if arg.startswith("--"):
            arg = arg[2:]
        if arg not in train_config:
            print("Warning: %s is not surported now." % (arg))
            continue
        train_config[arg] = value
        try:
            value = eval(value)
            if isinstance(value, (int, float)):
                train_config[arg] = value
        except:
            pass

    ts = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model_name = "%s_%s" % (train_config["predict_net"], ts)
    save_model_dir = train_config["save_model_dir"]
    os.makedirs(save_model_dir, exist_ok=True)

    # save config
    with open(os.path.join(save_model_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f)

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt="[ %(asctime)s ] %(message)s",
                            datefmt="%a %b %d %H:%M:%S %Y")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    logfile = logging.FileHandler(os.path.join(save_model_dir, "train_log.txt"), 'w')
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    # set device
    device = torch.device("cuda:%d" % train_config["gpu_id"] if train_config["gpu_id"] != -1 else "cpu")
    if train_config["gpu_id"] != -1:
        torch.cuda.set_device(device)

        # load data
        # os.makedirs(train_config["save_data_dir"], exist_ok=True)
    QD = QueryPreProcessing(queryset_dir=train_config["queryset_dir"], true_card_dir=train_config["true_card_dir"],
                            dataset=train_config["dataset"])
    # decompose the query
    QD.decomose_queries()
    all_subsets = QD.all_queries

    QS = Queryset(dataset_name=train_config['dataset_name'], data_dir=train_config["data_dir"],
                  dataset=train_config["dataset"], all_queries=all_subsets)

    train_sets, val_sets, test_sets = QS.train_sets, QS.val_sets, QS.test_sets
    # train_datasets = _to_datasets(train_sets)
    # val_datasets, test_datasets, = _to_datasets(val_sets), _to_datasets(test_sets)
    train_loaders, val_loaders, test_loaders = QS.train_loaders, QS.val_loaders, QS.test_loaders
    # train_loaders, val_loaders, test_loaders = _to_dataloaders(train_sets), _to_dataloaders(val_sets), _to_dataloaders(
    #     test_sets)
    graph = data_graph_transform(train_config['data_dir'], train_config['dataset'],
                                 train_config['dataset_name'])  # ./dataset/krogan/graph_batch.pt"
    # config['init_g_dim'] = graph.x.size(1)
    # train_config.update({'init_g_dim': graph.x.size(1)})
    # construct the model

    if train_config["model"] == "EDGEMEAN":
        model = EdgeMean(train_config)
    # elif train_config["model"] == "EDGESUM":
    #     model = ESUM(train_config)
    # elif train_config["model"] == "EDGESUMS":
    #     model = ESUMS(train_config)
    else:
        raise NotImplementedError("Currently, the %s model is not supported" % (train_config["model"]))
    # model = torch.compile(model)
    model = model.to(device)
    logger.info(model)
    logger.info("num of parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # optimizer and losses
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"],
                                 eps=1e-6)
    optimizer.zero_grad()
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=train_config['decay_factor'])
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.001)
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    # len(train_loaders), train_config["epochs"]*len(train_loaders), min_percent=0.0001)
    best_bp_losses = INF
    best_reg_losses = INF
    best_reg_epochs = {"train": -1, "val": -1, "test": -1}
    best_bp_epochs = {"train": -1, "val": -1, "test": -1}
    torch.backends.cudnn.benchmark = True
    total_train_time = 0
    total_dev_time = 0
    total_test_time = 0
    cur_reg_loss = {}
    if train_config['test_only']:
        evaluate_results, total_test_time = test(save_model_dir, test_loaders, train_config, graph, logger, writer)
        exit(0)
    tolerance_cnt = 0
    for epoch in range(train_config["epochs"]):
        # if train_config['cv'] == True:
        #     cross_validate(model=model, query_set=QS, device=device, config=train_config, graph=graph, logger=logger,
        #                    writer=writer)
        # else:
        mean_reg_loss, mean_bp_loss, _time = train(model, optimizer=optimizer, scheduler=scheduler,
                                                   data_type="train", data_loader=train_loaders,
                                                   device=device, config=train_config,
                                                   epoch=epoch, graph=graph,
                                                   logger=logger, writer=writer
                                                   , bottleneck=False)
        total_train_time += _time
        if scheduler and (epoch + 1) % train_config['decay_patience'] == 0:
            scheduler.step()
            # torch.save(model.state_dict(), os.path.join(save_model_dir, 'epoch%d.pt' % (epoch)))
        mean_reg_loss, mean_bp_loss, mean_var_loss, evaluate_results, total_time = evaluate(model=model,
                                                                                            data_type="val",
                                                                                            data_loader=val_loaders,
                                                                                            config=train_config,
                                                                                            graph=graph,
                                                                                            logger=logger,
                                                                                            writer=writer)
        if writer:
            writer.add_scalar("%s/REG-%s-epoch" % ("val", train_config["reg_loss"]), mean_reg_loss, epoch)
            writer.add_scalar("%s/BP-%s-epoch" % ("val", train_config["bp_loss"]), mean_bp_loss, epoch)
            writer.add_scalar("%s/Var-%s-epoch" % ("val", train_config['bp_loss']), mean_var_loss, epoch)
            total_dev_time += total_time
            # cur_reg_loss[loader_idx] = mean_reg_loss
            # flag = True
            # for key1, key2 in zip(cur_reg_loss.keys(), best_reg_losses.keys()):
            #     if cur_reg_loss[key1] > best_reg_losses[key2]:
            #         flag = False
            # if flag:
            #     for key1, key2 in zip(cur_reg_loss.keys(), best_reg_losses.keys()):
            #         best_reg_losses[key2] = cur_reg_loss[key1]
            #     best_reg_epochs['val'] = epoch
        if mean_reg_loss <= best_reg_losses:
            tolerance_cnt = 0
            best_reg_losses = mean_reg_loss
            best_reg_epochs['val'] = epoch
            logger.info(
                "data_type: {:<5s}\t\tbest mean loss: {:.3f} (epoch: {:0>3d})".format("val",
                                                                                      mean_reg_loss,
                                                                                      epoch))
            torch.save(model.state_dict(),
                       os.path.join(save_model_dir,
                                    'best_epoch_{:s}_{:s}_edge_bce.pt'.format(train_config['predict_net'],
                                                                              train_config['graph_net'])))
            with open(os.path.join(save_model_dir, '%s_%d.json' % ("val", epoch)), "w") as f:
                json.dump(evaluate_results, f)
                # for data_type in data_loaders.keys():
                #     logger.info(
                #         "data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type,
                #                                                                             best_reg_losses[data_type],
                #                                                                             best_reg_epochs[data_type]))
        tolerance_cnt += 1
        if tolerance_cnt >= 20:
            break
    print("data finish")
    evaluate_results, total_test_time = test(save_model_dir, test_loaders, train_config, graph, logger, writer)
    logger.info("train time: {:.3f}, train time per epoch :{:.3f}, test time: {:.3f}, all time: {:.3f}"
                .format(total_train_time, total_train_time / train_config["epochs"], total_test_time,
                        total_train_time + total_dev_time + total_test_time))

# model.load_state_dict(torch.load(os.path.join(save_model_dir, 'best_epoch.pt')))
#    # print(model)
#    for loader_idx, data_loader in enumerate(test_loaders):
#        mean_reg_loss, mean_bp_loss, evaluate_results, _time = evaluate(model=model, data_type="test",
#                                                                        data_loader=data_loader,
#                                                                        config=train_config,
#                                                                        graph=graph,
#                                                                        logger=logger, writer=writer)
#        total_test_time += _time
#        if mean_reg_loss <= best_reg_losses['test']:
#            best_reg_losses['test'] = mean_reg_loss
#            # best_reg_epochs['test'] =
#            logger.info(
#                "data_type: {:<5s}\tbest mean loss: {:.3f}".format("val", mean_reg_loss))
#            with open(os.path.join(save_model_dir, '%s_%s_%d.json' % (train_config['motif_net'], "best_test", loader_idx)), "w") as f:
#                json.dump(evaluate_results, f)
#
#        logger.info(
#            "data_type: {:<5s}\tbest mean loss: {:.3f} ".format("test", mean_reg_loss))
#    logger.info("train time: {:.3f}, train time per epoch :{:.3f}, test time: {:.3f}, all time: {:.3f}"
#                .format(total_train_time, total_train_time / train_config["epochs"], total_test_time,
#                        total_train_time + total_dev_time + total_test_time))

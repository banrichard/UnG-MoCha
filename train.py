import torch
import os
import numpy as np
import torch_geometric
import logging
import datetime
import math
import sys
import gc
import json
import time
import torch.nn.functional as F
import warnings
from functools import partial
from collections import OrderedDict

from torch.utils.tensorboard import SummaryWriter

from model.CountingNet import EdgeMean
from utils.dataloader import DataLoader
from utils.motif_processor import QueryPreProcessing, Queryset
from utils.utils import anneal_fn, _to_cuda

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

    "gpu_id": -1,
    "num_workers": 12,

    "epochs": 200,
    "batch_size": 16,
    "update_every": 1,  # actual batch_sizer = batch_size * update_every
    "print_every": 100,
    "init_emb": True,  # None, Normal
    "share_emb": False,  # sharing embedding requires the same vector length
    "share_arch": False,  # sharing architectures
    "dropout": 0.2,
    "dropatt": 0.2,

    "reg_loss": "MSE",  # MAE, MSEl
    "bp_loss": "MSE",  # MAE, MSE
    "bp_loss_slp": "anneal_cosine$1.0$0.01",  # 0, 0.01, logistic$1.0$0.01, linear$1.0$0.01, cosine$1.0$0.01,
    # cyclical_logistic$1.0$0.01, cyclical_linear$1.0$0.01, cyclical_cosine$1.0$0.01
    # anneal_logistic$1.0$0.01, anneal_linear$1.0$0.01, anneal_cosine$1.0$0.01
    "lr": 0.001,
    "weight_decay": 0.00001,
    "weight_decay_film": 0.00001,
    "max_grad_norm": 8,

    "model": "EdgeMean",  # CNN, RNN, TXL, RGCN, RGIN, RSIN

    "emb_dim": 128,
    "activation_function": "relu",  # sigmoid, softmax, tanh, relu, leaky_relu, prelu, gelu

    "predict_net": "SumPredictNet",  # MeanPredictNet, SumPredictNet, MaxPredictNet,
    # MeanAttnPredictNet, SumAttnPredictNet, MaxAttnPredictNet,
    # MeanMemAttnPredictNet, SumMemAttnPredictNet, MaxMemAttnPredictNet,
    # DIAMNet
    "predict_net_add_enc": True,
    "predict_net_add_degree": True,
    "predict_net_hidden_dim": 128,
    "predict_net_num_heads": 4,
    "predict_net_mem_len": 4,
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

    "queryset_dir": "/dataset/krogan/queryset",
    "true_card_dir": "/dataset/krogan/label",
    "dataset": "krogan",
    "data_dir": "/dataset",
    "dataset_name": "krogan_core",
    "save_res_dir": "/result"
}


def train(model, optimizer, scheduler, data_type, data_loader, device, config, epoch, logger=None, writer=None):
    epoch_step = len(data_loader)
    total_step = config["epochs"] * epoch_step
    total_reg_loss = 0
    total_bp_loss = 0
    total_cnt = 1e-6
    if config["reg_loss"] == "MAE":
        reg_crit = lambda pred, target: F.l1_loss(F.relu(pred), target)
    elif config["reg_loss"] == "MSE":
        reg_crit = lambda pred, target: F.mse_loss(F.relu(pred), target)
    elif config["reg_loss"] == "SMSE":
        reg_crit = lambda pred, target: F.smooth_l1_loss(F.relu(pred), target)
    if config["bp_loss"] == "MAE":
        bp_crit = lambda pred, target, neg_slp: F.l1_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "MSE":
        bp_crit = lambda pred, target, neg_slp: F.mse_loss(F.leaky_relu(pred, neg_slp), target)
    elif config["bp_loss"] == "SMSE":
        bp_crit = lambda pred, target, neg_slp: F.smooth_l1_loss(F.leaky_relu(pred, neg_slp), target)
    # data preparation
    graph = torch.load(config['data_dir'] + config['data_set'] + "graph_batch.pt")  # ./dataset/krogan/graph_batch.pt"
    model.train()
    total_time = 0
    for i, (motif_x, motif_edge_index, motif_edge_attr, card, var) in enumerate(data_loader):
        if config['cuda']:
            motif_x, motif_edge_index, motif_edge_attr = \
                _to_cuda(motif_x), _to_cuda(motif_edge_index), _to_cuda(motif_edge_attr)
            card, var = card.cuda(), var.cuda()
        s = time.time()
        pred, filmreg = model(motif_x, motif_edge_index, motif_edge_attr, graph)
        reg_loss = reg_crit(pred, card)

        if isinstance(config["bp_loss_slp"], (int, float)):
            neg_slp = float(config["bp_loss_slp"])
        else:
            bp_loss_slp, l0, l1 = config["bp_loss_slp"].rsplit("$", 3)
            neg_slp = anneal_fn(bp_loss_slp, i + epoch * epoch_step, T=total_step // 4, lambda0=float(l0),
                                lambda1=float(l1))
        bp_loss = bp_crit(pred, card, neg_slp) + train_config["weight_decay_film"] * filmreg
        # filmloss=(torch.sum(alpha ** 2)) ** 0.5 + (torch.sum(beta ** 2)) ** 0.5
        # bp_loss+=0.00001*filmloss
        # float
        reg_loss_item = reg_loss.item()
        bp_loss_item = bp_loss.item()
        total_reg_loss += reg_loss_item
        total_bp_loss += bp_loss_item

        if writer:
            writer.add_scalar("%s/REG-%s" % (data_type, config["reg_loss"]), reg_loss_item,
                              epoch * epoch_step + i)
            writer.add_scalar("%s/BP-%s" % (data_type, config["bp_loss"]), bp_loss_item, epoch * epoch_step + i)

        if logger and (i % config["print_every"] == 0 or i == epoch_step - 1):
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\treg loss: {:0>10.3f}\tbp loss: {:0>16.3f}\tground: {:.3f}\tpredict: {:.3f}".format(
                    epoch, config["epochs"], data_type, i, epoch_step,
                    reg_loss_item, bp_loss_item,
                    card, pred[0].item()))
        bp_loss.backward()

        if (config["update_every"] < 2 or i % config["update_every"] == 0 or i == epoch_step - 1):
            if config["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            if scheduler is not None:
                scheduler.step(epoch * epoch_step + i)
            optimizer.step()
            optimizer.zero_grad()
        e = time.time()
        total_time += e - s
    mean_reg_loss = total_reg_loss / total_cnt
    mean_bp_loss = total_bp_loss / total_cnt
    if writer:
        writer.add_scalar("%s/REG-%s-epoch" % (data_type, config["reg_loss"]), mean_reg_loss, epoch)
        writer.add_scalar("%s/BP-%s-epoch" % (data_type, config["bp_loss"]), mean_bp_loss, epoch)
    if logger:
        logger.info("epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\treg loss: {:0>10.3f}\tbp loss: {:0>16.3f}".format(
            epoch, config["epochs"], data_type, mean_reg_loss, mean_bp_loss))

    gc.collect()
    return mean_reg_loss, mean_bp_loss, total_time


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
    model_name = "%s_%s_%s" % (train_config["model"], train_config["predict_net"], ts)
    save_model_dir = train_config["save_model_dir"]
    os.makedirs(save_model_dir, exist_ok=True)

    # save config
    with open(os.path.join(save_model_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f)

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%Y/%m/%d %H:%M:%S')
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

    # reset the pattern parameters
    if train_config["share_emb"]:
        train_config["max_npv"], train_config["max_npvl"], train_config["max_npe"], train_config["max_npel"] = \
            train_config["max_ngv"], train_config["max_ngvl"], train_config["max_nge"], train_config["max_ngel"]

    # construct the model

    if train_config["model"] == "EDGEMEAN":
        model = EdgeMean(train_config)
        print(model)
    elif train_config["model"] == "EDGESUM":
        model = ESUM(train_config)
    elif train_config["model"] == "EDGESUMS":
        model = ESUMS(train_config)
    else:
        raise NotImplementedError("Currently, the %s model is not supported" % (train_config["model"]))
    model = model.to(device)
    logger.info(model)
    logger.info("num of parameters: %d" % (sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # load data
    os.makedirs(train_config["save_data_dir"], exist_ok=True)
    data_loaders = OrderedDict({"train": None, "dev": None})
    if all([os.path.exists(os.path.join(train_config["save_data_dir"],
                                        "%s_%s_dataset.pt" % (
                                                data_type, "dgl" if train_config["model"] in ["RGCN", "RGIN", "PPN",
                                                                                              "EDGEMEAN", "EDGESUM",
                                                                                              "EGATS",
                                                                                              "EDGESUMS"] else "edgeseq")))
            for data_type in data_loaders]):

        logger.info("loading data from pt...")
        for data_type in data_loaders:
            if train_config["model"] in ["RGCN", "RGIN", "PPN", "EDGEMEAN", "EDGESUM", "EGATS", "EDGESUMS"]:
                dataset = GraphAdjDataset(list())
                dataset.load(os.path.join(train_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=train_config["batch_size"],
                                  shuffle=data_type == "train", drop_last=False)
                #      sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=train_config["batch_size"],shuffle=False, drop_last=False)
                data_loader = DataLoader(dataset,
                                         batch_sampler=sampler,
                                         collate_fn=GraphAdjDataset.batchify,
                                         pin_memory=data_type == "train")
            data_loaders[data_type] = data_loader
            logger.info("data (data_type: {:<5s}, len: {}) generated".format(data_type, len(dataset.data)))
            logger.info(
                "data_loader (data_type: {:<5s}, len: {}, batch_size: {}) generated".format(data_type, len(data_loader),
                                                                                            train_config["batch_size"]))
    else:
        data = load_data(train_config["graph_dir"], train_config["pattern_dir"], train_config["metadata_dir"],
                         num_workers=train_config["num_workers"])
        logger.info("{}/{}/{} data loaded".format(len(data["train"]), len(data["dev"]), len(data["test"])))
        for data_type, x in data.items():
            if train_config["model"] in ["RGCN", "RGIN", "PPN", "EDGEMEAN", "EDGESUM", "EGATS", "EDGESUMS"]:
                if os.path.exists(os.path.join(train_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type))):
                    dataset = GraphAdjDataset(list())
                    dataset.load(os.path.join(train_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                else:
                    dataset = GraphAdjDataset(x)
                    dataset.save(os.path.join(train_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=train_config["batch_size"],
                                  shuffle=data_type == "train", drop_last=False)
                data_loader = DataLoader(dataset,
                                         batch_sampler=sampler,
                                         collate_fn=GraphAdjDataset.batchify,
                                         pin_memory=data_type == "train")
            data_loaders[data_type] = data_loader
            logger.info("data (data_type: {:<5s}, len: {}) generated".format(data_type, len(dataset.data)))
            logger.info(
                "data_loader (data_type: {:<5s}, len: {}, batch_size: {}) generated".format(data_type, len(data_loader),
                                                                                            train_config["batch_size"]))

    # optimizer and losses
    writer = SummaryWriter(save_model_dir)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"],
                                  amsgrad=True)
    optimizer.zero_grad()
    scheduler = None
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #     len(data_loaders["train"]), train_config["epochs"]*len(data_loaders["train"]), min_percent=0.0001)
    best_reg_losses = {"train": INF, "dev": INF, "test": INF}
    best_reg_epochs = {"train": -1, "dev": -1, "test": -1}
    torch.backends.cudnn.benchmark = True
    total_train_time = 0
    total_dev_time = 0
    total_test_time = 0
    for epoch in range(train_config["epochs"]):
        for data_type, data_loader in data_loaders.items():

            if data_type == "train":
                mean_reg_loss, mean_bp_loss, _time = train(model, optimizer, scheduler, data_type, data_loader, device,
                                                           train_config, epoch, logger=logger, writer=writer)
                total_train_time += _time
                torch.save(model.state_dict(), os.path.join(save_model_dir, 'epoch%d.pt' % (epoch)))
            if mean_reg_loss <= best_reg_losses[data_type]:
                best_reg_losses[data_type] = mean_reg_loss
                best_reg_epochs[data_type] = epoch
                logger.info(
                    "data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type, mean_reg_loss,
                                                                                        epoch))
    for data_type in data_loaders.keys():
        logger.info(
            "data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type, best_reg_losses[data_type],
                                                                                best_reg_epochs[data_type]))
    # _________________________________________________________________________________________________________________
    best_epoch = best_reg_epochs["dev"]

    data_loaders = OrderedDict({"test": None})
    if all([os.path.exists(os.path.join(train_config["save_data_dir"],
                                        "%s_%s_dataset.pt" % (
                                                data_type, "dgl" if train_config["model"] in ["RGCN", "RGIN", "PPN",
                                                                                              "EDGEMEAN", "EDGESUM",
                                                                                              "EGATS",
                                                                                              "EDGESUMS"] else "edgeseq")))
            for data_type in data_loaders]):

        logger.info("loading data from pt...")
        for data_type in data_loaders:
            if train_config["model"] in ["RGCN", "RGIN", "PPN", "EDGEMEAN", "EDGESUM", "EGATS", "EDGESUMS"]:
                dataset = GraphAdjDataset(list())
                dataset.load(os.path.join(train_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=train_config["batch_size"],
                                  shuffle=data_type == "train", drop_last=False)
                data_loader = DataLoader(dataset,
                                         batch_sampler=sampler,
                                         collate_fn=GraphAdjDataset.batchify,
                                         pin_memory=data_type == "train")
            data_loaders[data_type] = data_loader
            logger.info("data (data_type: {:<5s}, len: {}) generated".format(data_type, len(dataset.data)))
            logger.info(
                "data_loader (data_type: {:<5s}, len: {}, batch_size: {}) generated".format(data_type, len(data_loader),
                                                                                            train_config["batch_size"]))
    else:
        data = load_data(train_config["graph_dir"], train_config["pattern_dir"], train_config["metadata_dir"],
                         num_workers=train_config["num_workers"])
        logger.info("{}/{}/{} data loaded".format(len(data["train"]), len(data["dev"]), len(data["test"])))
        for data_type, x in data.items():
            if train_config["model"] in ["RGCN", "RGIN", "PPN", "EDGEMEAN", "EDGESUM", "EGATS", "EDGESUMS"]:
                if os.path.exists(os.path.join(train_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type))):
                    dataset = GraphAdjDataset(list())
                    dataset.load(os.path.join(train_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                else:
                    dataset = GraphAdjDataset(x)
                    dataset.save(os.path.join(train_config["save_data_dir"], "%s_dgl_dataset.pt" % (data_type)))
                sampler = Sampler(dataset, group_by=["graph", "pattern"], batch_size=train_config["batch_size"],
                                  shuffle=data_type == "train", drop_last=False)
                data_loader = DataLoader(dataset,
                                         batch_sampler=sampler,
                                         collate_fn=GraphAdjDataset.batchify,
                                         pin_memory=data_type == "train")
            data_loaders[data_type] = data_loader
            logger.info("data (data_type: {:<5s}, len: {}) generated".format(data_type, len(dataset.data)))
            logger.info(
                "data_loader (data_type: {:<5s}, len: {}, batch_size: {}) generated".format(data_type, len(data_loader),
                                                                                            train_config["batch_size"]))
    print("data finish")

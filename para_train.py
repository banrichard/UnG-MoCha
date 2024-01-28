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
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from model.CountingNet import Mocha
from motif_processor import QueryPreProcessing, Queryset
from utils.graph_operator import data_graph_transform

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
    "gpu_id": [0, 1, 2, 3],
    "num_workers": 16,
    "epochs": 200,
    "batch_size": 1,
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
    "bp_loss_slp": "anneal_cosine$1.0$0.01",
    "lr": 0.0006,
    "weight_decay": 0.0005,
    "weight_decay_var": 0.1,
    "weight_decay_film": 0.0001,
    "decay_factor": 0.7,
    "decay_patience": 20,
    "max_grad_norm": 8,
    "model": "Mocha",
    "motif_net": "NNGINConcat",
    "graph_net": "GIN",
    "emb_dim": 64,
    # sigmoid, softmax, tanh, relu, leaky_relu, prelu, gelu
    "activation_function": "relu",
    "motif_hidden_dim": 64,
    "motif_num_layers": 3,
    "predict_net": "CCANet",
    # MeanAttnPredictNet, SumAttnPredictNet, MaxAttnPredictNet,
    # MeanMemAttnPredictNet, SumMemAttnPredictNet, MaxMemAttnPredictNet,
    # DIAMNet, CCANet
    "predict_net_add_enc": False,
    "predict_net_add_degree": False,
    "predict_net_hidden_dim": 64,
    "predict_net_num_heads": 4,
    "mem_len": 1,
    "predict_net_mem_init": "mean",
    # mean, sum, max, attn, circular_mean, circular_sum, circular_max, circular_attn, lstm
    "predict_net_recurrent_steps": 3,
    "edgemean_num_bases": 8,
    "edgemean_graph_num_layers": 3,
    "edgemean_pattern_num_layers": 3,
    "edgemean_hidden_dim": 64,
    "num_g_hid": 64,
    "num_e_hid": 64,
    "out_g_ch": 64,
    "graph_num_layers": 3,
    "queryset_dir": "queryset",
    "true_card_dir": "label",
    "dataset": "krogan",
    "data_dir": "dataset",
    "dataset_name": "krogan_core.txt",
    "save_res_dir": "result",
    "save_model_dir": "saved_model",
    "init_g_dim": 1,
    "test_only": False,
    "GSL": True,
    "sub_ext": "edge",
    "hop": 1,
    "parallel": True,
}

def train(
        rank,
        model,
        train_set,
        data_type,
        config,
        epoch,
        graph,
        optimizer,
        logger=None,
        writer=None,
        bottleneck=False,
):
    model = DistributedDataParallel(model.cuda(), device_ids=[rank], find_unused_parameters=True)
    global bp_crit, reg_crit

    total_var_loss = 0
    total_reg_loss = 0
    total_bp_loss = 0
    total_cnt = 1e-6
    if config["reg_loss"] == "MAE":

        def reg_crit(pred, target):
            return F.l1_loss(F.relu(pred), target)

    elif config["reg_loss"] == "MSE":

        def reg_crit(pred, target):
            return F.mse_loss(F.relu(pred), target)

    elif config["reg_loss"] == "SMSE":

        def reg_crit(pred, target):
            return F.smooth_l1_loss(pred, target)

    elif config["reg_loss"] == "HUBER":

        def reg_crit(pred, target):
            return F.huber_loss(pred, target, delta=0.1)

    if config["bp_loss"] == "MAE":

        def bp_crit(pred, target):
            return F.l1_loss(F.leaky_relu(pred), target)

    elif config["bp_loss"] == "MSE":

        def bp_crit(pred, target):
            return F.mse_loss(pred, target)

    elif config["bp_loss"] == "SMSE":

        def bp_crit(pred, target):
            return F.smooth_l1_loss(pred, target)

    elif config["bp_loss"] == "HUBER":

        def bp_crit(pred, target):
            return F.huber_loss(pred, target, delta=0.1)

    # data preparation
    # config['init_pe_dim'] = graph.edge_attr.size(1)
    if bottleneck:
        model.load_state_dict(
            torch.load(
                os.path.join(
                    save_model_dir,
                    "best_epoch_{:s}_{:s}.pt".format(
                        train_config["predict_net"], train_config["graph_net"]
                    ),
                )
            )
        )
    sampler = DistributedSampler(train_set)
    sampler.set_epoch(epoch=epoch)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], sampler=sampler)
    epoch_step = len(train_loader)
    total_step = config["epochs"] * epoch_step
    total_time = 0
    model.train()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        motif, card, var = batch
        s = time.time()
        card = torch.log2_(card)
        var = torch.log2_(var)
        card = card.cuda()
        var = var.cuda()
        if config["predict_net"].startswith("Film") or config["predict_net"].startswith("CCA"):
            pred, pred_var, filmreg = model(
                motif.x, motif.edge_index, motif.edge_attr, graph
            )
            bp_loss = (
                    (1 - config["weight_decay_var"]) * bp_crit(pred, card)
                    + config["weight_decay_var"] * bp_crit(pred_var, var)
                    + train_config["weight_decay_film"] * filmreg
            )
        else:
            pred, pred_var = model(
                motif.x, motif.edge_index, motif.edge_attr, graph)
            bp_loss = (1 - config["weight_decay_var"]) * bp_crit(pred, card) + config[
                "weight_decay_var"
            ] * bp_crit(pred_var, var)
        reg_loss = (
            bp_loss
            if not config["predict_net"].startswith("Film")
               or config["predict_net"].startswith("CCA")
            else bp_loss - train_config["weight_decay_film"] * filmreg
        )

        bp_loss.backward()
        optimizer.step()
        var_loss_item = bp_crit(pred_var, var).item()
        reg_loss_item = reg_loss.item()
        bp_loss_item = bp_loss.item()
        total_var_loss += var_loss_item
        total_reg_loss += reg_loss_item
        total_bp_loss += bp_loss_item
        dist.all_reduce(bp_loss / len(batch))
        if logger and (
                i % config["print_every"] == 0
                or i == epoch_step - 1
        ) and local_rank == 0:
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:0>5d}/{:0>5d}\treg loss: {:.5f}\tbp loss: {"
                ":.5f}\tgt:{:.5f}\tpred:{:.5f}".format(
                    int(epoch),
                    int(config["epochs"]),
                    data_type,
                    int(i),
                    int(epoch_step),
                    float(reg_loss_item),
                    float(bp_loss_item),
                    float(card[0].item()),
                    float(pred[0].item())
                )
            )

        # if (i + 1) % 2 == 0 or i == epoch_step - 1:
        #     if config["max_grad_norm"] > 0:
        #         torch.nn.utils.clip_grad_norm_(
        #             model.parameters(), config["max_grad_norm"]
        #         )
        e = time.time()
        total_time += e - s
        total_cnt += 1
    mean_bp_loss = total_bp_loss / total_cnt
    mean_reg_loss = total_reg_loss / total_cnt
    mean_var_loss = total_var_loss / total_cnt
    if logger and local_rank == 0:
        logger.info(
            "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\treg loss: {:.4f}\tbp loss: {:.4f}".format(
                epoch, config["epochs"], data_type, mean_reg_loss, mean_bp_loss
            )
        )

    gc.collect()
    return mean_reg_loss, mean_bp_loss, total_time


def evaluate(model, data_type, data_loader, config, graph, logger=None, writer=None):
    epoch_step = len(data_loader)
    total_step = config["epochs"] * epoch_step
    total_var_loss = 0
    total_reg_loss = 0
    total_bp_loss = 0
    total_cnt = 1e-6

    evaluate_results = {
        "mean": {"count": list(), "pred_mean": list()},
        "var": {"var_t": list(), "pred_var": list()},
        "error": {"mae": 0.0, "mse": 0.0},
        "time": {"avg": list(), "total": 0.0},
    }

    if config["reg_loss"] == "MAE":

        def reg_crit(pred, target):
            return F.l1_loss(F.relu(pred), target)

    elif config["reg_loss"] == "MSE":

        def reg_crit(pred, target):
            return F.mse_loss(pred, target)

    elif config["reg_loss"] == "SMSE":

        def reg_crit(pred, target):
            return F.smooth_l1_loss(pred, target)

    elif config["reg_loss"] == "MAEMSE":

        def reg_crit(pred, target):
            return F.mse_loss(F.relu(pred), target) + F.l1_loss(F.relu(pred), target)

    elif config["reg_loss"] == "HUBER":

        def reg_crit(pred, target):
            return F.huber_loss(F.relu(pred), target)

    else:
        raise NotImplementedError

    if config["bp_loss"] == "MAE":

        def bp_crit(pred, target):
            return F.l1_loss(F.leaky_relu(pred), target)

    elif config["bp_loss"] == "MSE":

        def bp_crit(pred, target):
            return F.mse_loss(pred, target)

    elif config["bp_loss"] == "SMSE":

        def bp_crit(pred, target):
            return F.smooth_l1_loss(pred, target)

    elif config["bp_loss"] == "MAEMSE":

        def bp_crit(pred, target):
            return F.mse_loss(F.leaky_relu(pred), target) + F.l1_loss(
                F.leaky_relu(pred), target
            )

    elif config["bp_loss"] == "HUBER":

        def bp_crit(pred, target):
            return F.huber_loss(pred, target)

    else:
        raise NotImplementedError
    model.eval()
    total_time = 0
    with torch.no_grad():
        for batch_id, batch in enumerate(data_loader):
            motif, card, var = batch
            # y = val_to_distribution(card, var)
            motif = motif.cuda()
            card = card.cuda()
            var = var.cuda()
            evaluate_results["mean"]["count"].extend(card.view(-1).tolist())
            evaluate_results["var"]["var_t"].extend(var.view(-1).tolist())
            if config["predict_net"].startswith("Film") or config[
                "predict_net"
            ].startswith("CCA"):
                st = time.time()
                pred, pred_var, filmreg = model(
                    motif.x, motif.edge_index, motif.edge_attr, graph
                )
                pred = pred.cuda()
                pred_var = pred_var.cuda()
                bp_loss = (
                        bp_crit(pred, card)
                        + config["weight_decay_var"] * bp_crit(pred_var, var)
                        + train_config["weight_decay_film"] * filmreg
                )
            else:
                st = time.time()
                pred, pred_var = model(
                    motif.x, motif.edge_index, motif.edge_attr, graph
                )
                pred = pred.cuda()
                pred_var = pred_var.cuda()
                bp_loss = (1 - config["weight_decay_var"]) * bp_crit(
                    pred, card
                ) + config["weight_decay_var"] * bp_crit(pred_var, var)
            et = time.time()
            evaluate_results["time"]["total"] += et - st
            avg_t = et - st

            evaluate_results["time"]["avg"].extend([avg_t])
            evaluate_results["mean"]["pred_mean"].extend(
                pred.view(-1).tolist())
            evaluate_results["var"]["pred_var"].extend(
                pred_var.view(-1).tolist())
            reg_loss = (1 - config["weight_decay_var"]) * reg_crit(pred, card) + config[
                "weight_decay_var"
            ] * reg_crit(pred_var, var)
            reg_loss_item = reg_loss.mean().item()
            var_loss_item = bp_crit(pred_var, var).item()
            bp_loss_item = bp_loss.mean().item()
            total_reg_loss += reg_loss_item
            total_bp_loss += bp_loss_item
            total_var_loss += var_loss_item
            evaluate_results["error"]["mae"] += (
                F.l1_loss(F.relu(pred), card).sum().item()
            )
            evaluate_results["error"]["mse"] += (
                F.mse_loss(F.relu(pred), card).sum().item()
            )
            et = time.time()
            total_time += et - st
            total_cnt += 1
        mean_bp_loss = total_bp_loss / total_cnt
        mean_reg_loss = total_reg_loss / total_cnt
        mean_var_loss = total_var_loss / total_cnt
        if logger and batch_id == epoch_step - 1 and config["test_only"] is False:
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\tbatch: {:d}/{:d}\treg loss: {:.4f}\tbp loss: {:.4f}\tgt:{:.4f}\tpred:{:.4f}".format(
                    int(epoch),
                    int(config["epochs"]),
                    (data_type),
                    int(batch_id),
                    int(epoch_step),
                    float(reg_loss_item),
                    float(bp_loss_item),
                    float(card[0].item()),
                    float(pred[0].item())
                )
            )

        if logger and config["test_only"] is False:
            logger.info(
                "epoch: {:0>3d}/{:0>3d}\tdata_type: {:<5s}\treg loss: {:.4f}\tbp loss: {:.4f}".format(
                    epoch, config["epochs"], data_type, mean_reg_loss, mean_bp_loss
                )
            )

    gc.collect()
    return mean_reg_loss, mean_bp_loss, mean_var_loss, evaluate_results, total_time


def test(model, save_model_dir, test_loaders, config, graph, logger):
    total_test_time = 0
    model.load_state_dict(
        torch.load(
            os.path.join(
                save_model_dir,
                "best_epoch_{:s}_{:s}_{:s}_edge_gsl.pt".format(
                    config["predict_net"], config["graph_net"], config["motif_net"]
                ),
            )
        )
    )
    print("model loaded!")
    # print(model)
    mean_reg_loss, mean_bp_loss, mean_var_loss, evaluate_results, _time = evaluate(
        model=model,
        data_type="test",
        data_loader=test_loaders,
        config=config,
        graph=graph,
        logger=logger,
    )
    total_test_time += _time
    logger.info(
        "data_type: {:<5s}\tbest mean loss: {:.3f}".format(
            "test", mean_reg_loss)
    )
    with open(
            os.path.join(
                save_model_dir,
                "%s_%s_%s_%s_%s_edge_gsl_parallel.json"
                % (
                        train_config["predict_net"],
                        train_config["graph_net"],
                        train_config["motif_net"],
                        "best_test",
                        train_config["dataset"],
                ),
            ),
            "w",
    ) as f:
        json.dump(evaluate_results, f)

    return evaluate_results, total_test_time


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
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

    ts = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    model_name = "%s_%s" % (train_config["predict_net"], ts)
    save_model_dir = train_config["save_model_dir"]
    os.makedirs(save_model_dir, exist_ok=True)

    # save config
    with open(os.path.join(save_model_dir, "train_config.json"), "w") as f:
        json.dump(train_config, f)

    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="[ %(asctime)s ] %(message)s", datefmt="%a %b %d %H:%M:%S %Y"
    )
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    local_time = time.strftime("%Y-%m-%d_%H%M%S", time.localtime())
    logfile = logging.FileHandler(
        os.path.join(save_model_dir,
                     "train_log_{:s}.txt".format(local_time)), "w"
    )
    logfile.setFormatter(fmt)
    logger.addHandler(logfile)

    # load data
    QD = QueryPreProcessing(
        queryset_dir=train_config["queryset_dir"],
        true_card_dir=train_config["true_card_dir"],
        dataset=train_config["dataset"],
    )
    QD.decomose_queries()
    all_subsets = QD.all_queries

    QS = Queryset(
        dataset_name=train_config["dataset_name"],
        data_dir=train_config["data_dir"],
        dataset=train_config["dataset"],
        all_queries=all_subsets,
    )
    train_sets, val_sets, test_sets = QS.train_sets, QS.val_sets, QS.test_sets
    val_loaders, test_loaders = QS.val_loaders, QS.test_loaders
    # load graph
    graph = data_graph_transform(
        train_config["data_dir"],
        train_config["dataset"],
        train_config["dataset_name"],
        gsl=train_config["GSL"],
        extraction=train_config["sub_ext"],
        hop=train_config["hop"]
    )
    model = Mocha(train_config)
    if local_rank == 0:
        logger.info(model)
        # debug the non-gradient layer
        for name, param in model.named_parameters():
            if param.requires_grad is False:
                logger.info(name)
        logger.info(
            "num of parameters: %d"
            % (sum(p.numel() for p in model.parameters() if p.requires_grad))
        )

    # optimizer and losses
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
        eps=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.001)
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
    if train_config["test_only"]:
        evaluate_results, total_test_time = test(
            model, save_model_dir, test_loaders, train_config, graph, logger
        )
        exit(0)
    tolerance_cnt = 0

    for epoch in range(train_config["epochs"]):
        mean_reg_loss, mean_bp_loss, _time = train(
            rank=local_rank,
            model=model,
            train_set=train_sets,
            data_type="train",
            config=train_config,
            epoch=epoch,
            graph=graph,
            optimizer=optimizer,
            logger=logger,
            writer=None,
            bottleneck=False,
        )
        total_train_time += _time
        if scheduler and (epoch + 1) % train_config["decay_patience"] == 0:
            scheduler.step()
            # torch.save(model.state_dict(), os.path.join(save_model_dir, 'epoch%d.pt' % (epoch)))
        if local_rank == 0:
            (
                mean_reg_loss,
                mean_bp_loss,
                mean_var_loss,
                evaluate_results,
                total_time,
            ) = evaluate(
                model=model,
                data_type="val",
                data_loader=val_loaders,
                config=train_config,
                graph=graph,
                logger=logger,
                writer=None,
            )
            total_dev_time += total_time
            err = best_reg_losses - mean_reg_loss
            if err > 1e-4:
                tolerance_cnt = 0
                best_reg_losses = mean_reg_loss
                best_reg_epochs["val"] = epoch
                logger.info(
                    "data_type: {:<5s}\t\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(
                        "val", mean_reg_loss, epoch
                    )
                )
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        save_model_dir,
                        "best_epoch_{:s}_{:s}_{:s}_edge_gsl.pt".format(
                            train_config["predict_net"],
                            train_config["graph_net"],
                            train_config["motif_net"],
                        ),
                    ),
                )
                with open(
                        os.path.join(save_model_dir, "%s_%d.json" %
                                                     ("val", epoch)), "w"
                ) as f:
                    json.dump(evaluate_results, f)
                # for data_type in data_loaders.keys():
                #     logger.info(
                #         "data_type: {:<5s}\tbest mean loss: {:.3f} (epoch: {:0>3d})".format(data_type,
                #                                                                             best_reg_losses[data_type],
                #                                                                             best_reg_epochs[data_type]))
            tolerance_cnt += 1
            if tolerance_cnt >= 20:
                break
        dist.barrier()
    print("data finish")
    if local_rank == 0:
        evaluate_results, total_test_time = test(
            model, save_model_dir, test_loaders, train_config, graph, logger
        )
        logger.info(
            "train time: {:.3f}, train time per epoch :{:.3f}, test time: {:.3f}, all time: {:.3f}".format(
                total_train_time,
                total_train_time / train_config["epochs"],
                total_test_time,
                total_train_time + total_dev_time + total_test_time,
            )
        )
    dist.destroy_process_group()

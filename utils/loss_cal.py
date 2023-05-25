import re
import subprocess

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support


def compute_mae(predict, count):
    error = np.absolute(predict - count)
    return error.mean()


def compute_abmae(predict, count):
    error = np.absolute(predict - count) / (count + 10)
    return error.mean()


def compute_rmse(predict, count):
    error = np.power(predict - count, 2)
    return np.power(error.mean(), 0.5)


def compute_p_r_f1(predict, count):
    p, r, f1, _ = precision_recall_fscore_support(predict, count, average="binary")
    return p, r, f1


def compute_tp(predict, count):
    true_count = count == 1
    true_pred = predict == 1
    true_pred_count = true_count * true_pred
    return np.count_nonzero(true_pred_count) / np.count_nonzero(true_count)


def get_best_epochs(log_file):
    regex = re.compile(r"data_type:\s+(\w+)\s+best\s+([\s\w\-]+).*?\(epoch:\s+(\d+)\)")
    best_epochs = dict()
    # get the best epoch
    try:
        lines = subprocess.check_output(["tail", log_file, "-n3"]).decode("utf-8").split("\n")[0:-1]
        print(lines)
    except:
        with open(log_file, "r") as f:
            lines = f.readlines()

    for line in lines[-3:]:
        matched_results = regex.findall(line)
        for matched_result in matched_results:
            if "loss" in matched_result[1]:
                best_epochs[matched_result[0]] = int(matched_result[2])
    if len(best_epochs) != 3:
        for line in lines:
            matched_results = regex.findall(line)
            for matched_result in matched_results:
                if "loss" in matched_result[1]:
                    best_epochs[matched_result[0]] = int(matched_result[2])
    return best_epochs

def bp_compute_abmae(predict, count):
    error = torch.absolute(predict - count) / (count + 1)
    return error.mean()


def bp_compute_large10_abmae(predict, count):
    tcount = count >= 10
    count = count * tcount
    nonzero = torch.nonzero(count)
    predict = predict[nonzero]
    count = count[nonzero]
    error = torch.absolute(predict - count) / count
    return error.mean()


def bp_compute_large20_abmae(predict, count):
    tcount = count >= 20
    count = count * tcount
    nonzero = torch.nonzero(count)
    predict = predict[nonzero]
    count = count[nonzero]
    error = torch.absolute(predict - count) / count
    return error.mean()
import argparse
import time, os, sys
from shutil import copy
import matplotlib.pyplot as plt
import logging
from math import ceil
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx

from graph_converter import Converter
from model.HUGNN import NestedGIN
from utils.utils import create_subgraphs
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True

import pdb


def simulate(args, device):
    results = {}
    for n in args.n:
        print('n = {}'.format(n))
        for h in range(1, args.h+1):
            G = Converter()
            loader = DataLoader(G, batch_size=1)
            model = NestedGIN(args.layers, 32)
            model.to(device)
            output = run_simulation(model, loader, device)  # output shape [G.number_of_nodes(), feat_dim]
            collision_rate = compute_simulation_collisions(output, ratio=True)
            results[(n, h)] = collision_rate
            torch.cuda.empty_cache()
            print('h = {}: {}'.format(h, collision_rate))
        print('#'*30)
    return results


def run_simulation(model, loader, device):
    model.eval()
    with torch.no_grad():
        output = []
        for data in loader:
            data = data.to(device)
            output.append(model(data))
        output = torch.cat(output, 0)
    return output


def save_simulation_result(results, res_dir, pic_format='pdf'):
    n_l, h_l, r_l = [], [], []
    for (n, h), r in results.items():
        n_l.append(n)
        h_l.append(h)
        r_l.append(r)
    main = plt.scatter(n_l, h_l, c=r_l, cmap="Greys", edgecolors='k', linewidths=0.2)
    plt.colorbar(main, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    n_min, n_max = min(n_l), max(n_l)
    lbound = plt.plot([n_min, n_max], 
                      [np.log(n_min)/np.log(2)/2, np.log(n_max)/np.log(2)/2], 
                      'r--', label='0.5 log(n) / log(r-1)')
    ubound = plt.plot([n_min, n_max], 
                      [np.log(n_min)/np.log(2), np.log(n_max)/np.log(2)], 
                      'b--', label='log(n) / log(r-1)')
    plt.xscale('log')
    plt.xlabel('number of nodes (n)')
    plt.ylabel('height of rooted subgraphs (h)')
    plt.legend(loc = 'upper left')
    plt.savefig('{}/simulation_results.{}'.format(res_dir, pic_format), dpi=300)


def compute_simulation_collisions(outputs, ratio=True):
    epsilon = 1e-10
    N = outputs.size(0)
    with torch.no_grad():
        a = outputs.unsqueeze(-1)
        b = outputs.t().unsqueeze(0)
        diff = a-b
        diff = (diff**2).sum(dim=1)
        n_collision = int(((diff < epsilon).sum().item()-N)/2)
        r = n_collision / (N*(N-1)/2)
    if ratio:
        return r
    else:
        return n_collision


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description='Nested GNN Simulation Experiment')
    parser.add_argument('--N', type=int, default=100,
                    help='number of graphs in simultation')
    parser.add_argument('--h', type=int, default=6,
                    help='largest height of rooted subgraphs to simulate')
    parser.add_argument('--layers', type=int, default=1, help='# message passing layers')
    parser.add_argument('--save_appendix', default='',
                    help='what to append to save-names when saving results')
    args = parser.parse_args()
    # args.n = [int(n) for n in args.n]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.save_appendix == '':
        args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
    args.res_dir = 'results/simulation{}'.format(args.save_appendix)
    print('Results will be saved in ' + args.res_dir)
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
    # Backup python files.
    copy('run_simulation.py', args.res_dir)
    copy('utils.py', args.res_dir)
    # Save command line input.
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)
    print('Command line input: ' + cmd_input + ' is saved.')

    path = 'data/simulation'
    pre_transform = None
    if args.h is not None:
        if type(args.h) == int:
            path += '/ngnn_h' + str(args.h)
    def pre_transform(g, h):
        return create_subgraphs(g, h, node_label='no', use_rd=False,
                                subgraph_pretransform=None)

    # Plot visualization figure
    results = simulate(args, device)
    save_simulation_result(results, args.res_dir)













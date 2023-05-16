import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec

from node2vec import Node2Vec
from node2vec.edges import HadamardEmbedder
from utils.utils import load_graph


class Embedding(nn.Module):

    def __init__(self, input_dim, emb_dim):
        super(Embedding, self).__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.emb_layer = nn.Linear(input_dim, emb_dim, bias=False)
        nn.init.normal_(self.emb_layer.weight, 0.0, 1.0)

    def forward(self, x):
        out = self.emb_layer(x)
        return out

graph = load_graph("../dataset/krogan/krogan_core.txt")
node2vec = Node2Vec(graph, workers=1)
model = node2vec.fit(min_count=1,workers=1)
edges_embs = HadamardEmbedder(keyed_vectors=model.wv)
outfile="../dataset/krogan/embedding/edge_hadamard.csv"
edges_kv = edges_embs.as_keyed_vectors()
edges_kv.save_word2vec_format(outfile)

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv,GraphConv,GATConv
from dgl.nn.pytorch.glob import SumPooling
from utils import *
class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h=self.linears[0](h)
        h = F.relu(self.batch_norm(h))
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(self, numberofglycos, hidden_dim, output_dim,init_eps):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.glyco_embedding=nn.Embedding(numberofglycos, hidden_dim, padding_idx=None)
                #如果输入一直MLP的话，不同的糖会呈现线性关系，这里我们采用embedding
        # num_layers = 5
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(GNN_global_num_layers - 1):  # excluding the input layer
            mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, init_eps=init_eps,learn_eps=False)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(GNN_global_num_layers):
            self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h):
        u , v = g.edges()
        g.add_edges(v , u) # bidirect
        g = g.add_self_loop() #add self-loops #global representation也从有向无环图变成无向有环图
        h=self.glyco_embedding(h)
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))

        return score_over_layer

class GCN(nn.Module):
    def __init__(self, numberofglycos, hidden_dim, output_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.glyco_embedding=nn.Embedding(numberofglycos, hidden_dim, padding_idx=None)
                #如果输入一直MLP的话，不同的糖会呈现线性关系，这里我们采用embedding
        # num_layers = 5 #也可以调整
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(GNN_global_num_layers - 1):  # excluding the input layer
            mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GraphConv(in_feats=hidden_dim,out_feats=hidden_dim,allow_zero_in_degree=True)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(GNN_global_num_layers):
            self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h):
        u , v = g.edges()
        g.add_edges(v , u) # bidirect
        g = g.add_self_loop() #add self-loops #global representation也从有向无环图变成无向有环图
        h=self.glyco_embedding(h)
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))

        return score_over_layer

class GAT(nn.Module):
    def __init__(self, numberofglycos, hidden_dim, output_dim,num_heads):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.glyco_embedding=nn.Embedding(numberofglycos, hidden_dim, padding_idx=None)
                #如果输入一直MLP的话，不同的糖会呈现线性关系，这里我们采用embedding
        # num_layers = 5 #也可以调整
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(GNN_global_num_layers - 1):  # excluding the input layer
            mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GATConv(in_feats=hidden_dim,out_feats=hidden_dim//num_heads,num_heads=num_heads,allow_zero_in_degree=True)
            )  # set to True if learning epsilon
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.linear_prediction = nn.ModuleList()
        for layer in range(GNN_global_num_layers):
            self.linear_prediction.append(nn.Linear(hidden_dim, output_dim))
        self.drop = nn.Dropout(0.5)
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets

    def forward(self, g, h):
        u , v = g.edges()
        g.add_edges(v , u) # bidirect
        g = g.add_self_loop() #add self-loops #global representation也从有向无环图变成无向有环图
        h=self.glyco_embedding(h)
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h=h.reshape(-1,GNN_global_hidden_dim)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)
        score_over_layer = 0
        # perform graph sum pooling over all nodes in each layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linear_prediction[i](pooled_h))

        return score_over_layer
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="name of dataset",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GINConv module with a fixed epsilon")
    #有epsilon以就可以改变自身节点的权重
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load and split dataset
    dataset_train=torch.load("/remote-home/yxwang/test/zzb/DeepGlyco/model/20230127_test_model_validata")
    dataset=dataset_train['strct_graph'].values.tolist()
    import dgl
    import random #后面随机选择，包括batch内数目改一下
    batchsize=2
    sample=[i[0] for i in random.sample(dataset, batchsize)]
    train_loader = [dgl.batch(sample).to(device)]
    import ipdb
    # ipdb.set_trace()
    # create GIN model
    batched_graph = train_loader[0].to(device)
    feat = batched_graph.ndata.pop("attr")
    print("feat",feat)
    number_of_glycos=20
    print(number_of_glycos)
    out_size = 768
    hidden_size=16
    model = GIN(number_of_glycos, hidden_size, out_size,init_eps=0).to(device)
    print("batchgraph",batched_graph)
    ipdb.set_trace()
    logits = model(batched_graph, feat)
    print(logits.size())
    print("logits",logits)

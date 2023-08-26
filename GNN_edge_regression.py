import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv,GATConv,GraphConv
# from dgl.nn.pytorch.conv import GraphConv
from dgl.nn.pytorch.glob import SumPooling
from utils import *
import ipdb
#因为要做全局表示，就没有训练，可以加上COMPLEX,HYBRID之类的label，做分类任务
#numberofglycos不用特别大
#GNN_ablation:GAT GCN GIN
#从单向图改成双向图 引入对于P的影响
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
        # ipdb.set_trace()
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(self, numberofglycos, hidden_dim, output_dim,init_eps):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.glyco_embedding=nn.Embedding(numberofglycos, hidden_dim, padding_idx=None)
        #如果输入一直MLP的话，不同的糖会呈现线性关系，这里我们采用embedding
        # num_layers = 5 #层数可以调整
        print("GNN_edge_num_layers ",GNN_edge_num_layers)
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(GNN_edge_num_layers - 1):  # excluding the input layer
            mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(
                GINConv(mlp, init_eps=init_eps,learn_eps=False)
            )  # set to True if learning epsilon #学一下这里#aggregator_type
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.W=nn.Linear(2*hidden_dim,int(num_col*2/3))
        self.W2=nn.Linear(hidden_dim,int(num_col*2/3))
        self.W1=nn.Linear(2*hidden_dim,hidden_dim)
        self.W3=nn.Linear(hidden_dim,hidden_dim)
        self.predictH=nn.Parameter(torch.randn(hidden_dim,hidden_dim))
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets
    def apply_edges(self, g,targetedges):
        h_u = g.ndata['h'][targetedges[0]]
        h_v = g.ndata['h'][targetedges[1]]
        if GNN_edge_decoder_type=="linear":
            score = self.W(torch.cat([h_u, h_v], 1))
        elif GNN_edge_decoder_type=="mlp":
            score = self.W2(F.relu(self.W1(torch.cat([h_u, h_v], 1))))
        elif GNN_edge_decoder_type=="hadamardlinear":
            score = self.W2(h_u* h_v)
        elif GNN_edge_decoder_type=="hadamardmlp":
            score = self.W2(F.relu(self.W3(h_u* h_v)))
        return {'score': score}
    def forward(self, g, h,peptide_rep=None,peptide_ind=None):
        #g是batched_graph, h是feat
        u , v = g.edges()
        targetedges=[u,v]
        g.add_edges(v , u) # bidirect
        g = g.add_self_loop() #add self-loops
        h=self.glyco_embedding(h)
        if peptide_rep is not None:
            #之前有问题，hidden dim和output dim都是16，替换的不对。
            # ipdb.set_trace()
            h[peptide_ind]=peptide_rep
        # ipdb.set_trace()
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            # ipdb.set_trace()
            h = layer(g, h) #为什么有两个输入
            # ipdb.set_trace()
            h = self.batch_norms[i](h)
            # ipdb.set_trace()
            h = F.relu(h)
            # ipdb.set_trace()
            hidden_rep.append(h)
        h=sum(hidden_rep)
        g.ndata["h"]=h
        edgescore=self.apply_edges(g,targetedges)["score"]
        return edgescore
    
class GCN(nn.Module):
    def __init__(self, numberofglycos, hidden_dim, output_dim):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.glyco_embedding=nn.Embedding(numberofglycos, hidden_dim, padding_idx=None)
        #如果输入一直MLP的话，不同的糖会呈现线性关系，这里我们采用embedding
        # num_layers = 5 #层数可以调整
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        print("GNN_edge_num_layers ",GNN_edge_num_layers)
        for layer in range(GNN_edge_num_layers - 1):  # excluding the input layer
            self.ginlayers.append(
                GraphConv(in_feats=hidden_dim,out_feats=hidden_dim,allow_zero_in_degree=True)
            )  # set to True if learning epsilon #学一下这里#aggregator_type
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.W=nn.Linear(2*hidden_dim,int(num_col*2/3))
        self.W2=nn.Linear(hidden_dim,int(num_col*2/3))
        self.W1=nn.Linear(2*hidden_dim,hidden_dim)
        self.W3=nn.Linear(hidden_dim,hidden_dim)
        self.predictH=nn.Parameter(torch.randn(hidden_dim,hidden_dim))
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets
    def apply_edges(self, g,targetedges):
        h_u = g.ndata['h'][targetedges[0]]
        h_v = g.ndata['h'][targetedges[1]]
        # print("GNN_edge_decoder_type ",GNN_edge_decoder_type)
        if GNN_edge_decoder_type=="linear":
            score = self.W(torch.cat([h_u, h_v], 1))
        elif GNN_edge_decoder_type=="mlp":
            score = self.W2(F.relu(self.W1(torch.cat([h_u, h_v], 1))))
        elif GNN_edge_decoder_type=="hadamardlinear":
            score = self.W2(h_u* h_v)
        elif GNN_edge_decoder_type=="hadamardmlp":
            score = self.W2(F.relu(self.W3(h_u* h_v)))
        return {'score': score}
    def forward(self, g, h,peptide_rep=None,peptide_ind=None):
        #g是batched_graph, h是feat
        u , v = g.edges()
        targetedges=[u,v]
        g.add_edges(v , u) # bidirect
        g = g.add_self_loop() #add self-loops
        h=self.glyco_embedding(h)
        # import ipdb
        # ipdb.set_trace()
        if peptide_rep is not None:
            #之前有问题，hidden dim和output dim都是16，替换的不对。
            # ipdb.set_trace()
            h[peptide_ind]=peptide_rep
        # ipdb.set_trace()
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            # ipdb.set_trace()
            h = layer(g, h) #为什么有两个输入
            # ipdb.set_trace()
            h = self.batch_norms[i](h)
            # ipdb.set_trace()
            h = F.relu(h)
            # ipdb.set_trace()
            hidden_rep.append(h)
        h=sum(hidden_rep)
        g.ndata["h"]=h
        edgescore=self.apply_edges(g,targetedges)["score"]
        return edgescore
    
  
class GAT(nn.Module):
    def __init__(self, numberofglycos, hidden_dim, output_dim,num_heads):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.glyco_embedding=nn.Embedding(numberofglycos, hidden_dim, padding_idx=None)
        #如果输入一直MLP的话，不同的糖会呈现线性关系，这里我们采用embedding
        # num_layers = 5 #层数可以调整
        # five-layer GCN with two-layer MLP aggregator and sum-neighbor-pooling scheme
        for layer in range(GNN_edge_num_layers - 1):  # excluding the input layer
            self.ginlayers.append(
                GATConv(in_feats=hidden_dim,out_feats=hidden_dim//num_heads,num_heads=num_heads,allow_zero_in_degree=True)
            )  # set to True if learning epsilon #学一下这里#aggregator_type
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        # linear functions for graph sum poolings of output of each layer
        self.W=nn.Linear(2*hidden_dim,int(num_col*2/3))
        self.W2=nn.Linear(hidden_dim,int(num_col*2/3))
        self.W1=nn.Linear(2*hidden_dim,hidden_dim)
        self.W3=nn.Linear(hidden_dim,hidden_dim)
        self.predictH=nn.Parameter(torch.randn(hidden_dim,hidden_dim))
        self.pool = (
            SumPooling()
        )  # change to mean readout (AvgPooling) on social network datasets
    def apply_edges(self, g,targetedges):
        h_u = g.ndata['h'][targetedges[0]]
        h_v = g.ndata['h'][targetedges[1]]
        if GNN_edge_decoder_type=="linear":
            score = self.W(torch.cat([h_u, h_v], 1))
        elif GNN_edge_decoder_type=="mlp":
            score = self.W2(F.relu(self.W1(torch.cat([h_u, h_v], 1))))
        elif GNN_edge_decoder_type=="hadamardlinear":
            score = self.W2(h_u* h_v)
        elif GNN_edge_decoder_type=="hadamardmlp":
            score = self.W2(F.relu(self.W3(h_u* h_v)))
        return {'score': score}
    def forward(self, g, h,peptide_rep=None,peptide_ind=None):
        #g是batched_graph, h是feat
        u , v = g.edges()
        targetedges=[u,v]
        g.add_edges(v , u) # bidirect
        g = g.add_self_loop() #add self-loops
        h=self.glyco_embedding(h)
        if peptide_rep is not None:
            #之前有问题，hidden dim和output dim都是16，替换的不对。
            h[peptide_ind]=peptide_rep
        # list of hidden representation at each layer (including the input layer)
        hidden_rep = [h]
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h) #为什么有两个输入 【node_dim,head_num,hidden_dim]
            h=h.reshape(-1,GNN_edge_hidden_dim)
            # ipdb.set_trace()
            h = self.batch_norms[i](h)
            # ipdb.set_trace()
            h = F.relu(h)
            # ipdb.set_trace()
            hidden_rep.append(h)
        h=sum(hidden_rep)
        g.ndata["h"]=h
        # ipdb.set_trace()
        edgescore=self.apply_edges(g,targetedges)["score"]
        # ipdb.set_trace()
        return edgescore


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
    # ipdb.set_trace()
    feat = batched_graph.ndata["attr"]
    #我是对边进行embed，进行运算，还是先对节点运算，再进行操作到边
    print("feat",feat)
    number_of_glycos=20
    print(number_of_glycos)
    out_size = 768
    hidden_size=16
    # ipdb.set_trace()
    model = GIN(number_of_glycos, hidden_size, out_size,init_eps=0).to(device)
    print("batchgraph",batched_graph)
    # ipdb.set_trace()
    logits = model(batched_graph, feat)
    print(logits.size())
    ipdb.set_trace()
    print("logits",logits)
from operator import concat
from tkinter import N
from turtle import forward
import torch
import torch.nn as nn
from torch.nn import init, LSTM, GRU
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn import GATConv
# from layer import LP_Filter, HP_Filter, BP_Filter, LPLayer_single, HPLayer_single, BPLayer_single, ChannelAttention
from layer import LPLayer_single, HPLayer_single, ChannelAttention, LP_Filter
from layer import DiffAttention, HopsAttention
import numpy as np

class DiffGCN(nn.Module):
    def __init__(self,
                g,
                in_dim,
                hidden_dim,
                out_dim,
                hop_num=1,
                feat_drop=0,
                device='cuda'
                ):
        super(DiffGCN, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.hop_num = hop_num
        self.device = device
        self.feat_drop = nn.Dropout(feat_drop)
        
        self.diff_layer = nn.ModuleList()
        # self.diff_layer.append(DiffAttention(g[0], in_dim, hidden_dim, feat_drop))        
        for i in range(hop_num):#DiffAttention层堆叠：提取中心节点与多跳邻居节点的特征差异
            self.diff_layer.append(DiffAttention(g[i], in_dim, hidden_dim, feat_drop, device))
        self.out_layer = nn.Linear(hidden_dim, out_dim, bias=False)
        
        #新增邻居阶次的自适应选择
        if hop_num > 1:
            # self.neihop_select = LSTM(hidden_dim, hidden_dim)
            self.neihop_select = GRU(hidden_dim, hidden_dim)
            # self.neihop_select = HopsAttention(hidden_dim)            
        nn.init.xavier_uniform_(self.out_layer.weight, gain=1.414)
        
    def forward(self, feat):
        h_e = []
        for i in range(self.hop_num):
            h_k = self.diff_layer[i](feat)
            h_e.append(h_k)
        #新增邻居阶次的自适应选择
        if self.hop_num > 1:
            # 点积注意力
            # h_out = self.neihop_select(h_e, self.hop_num)
            
            # GRU
            h = torch.zeros(1, feat.shape[0], self.hidden_dim).to(self.device)
            for i in range(self.hop_num):
                h_out, h = self.neihop_select(h_e[i].unsqueeze(0), h)
                h_out = h_out.mean(0)
            
            # LSTM
            # h = torch.zeros(1, feat.shape[0], self.hidden_dim).to(self.device)
            # c = torch.zeros(1, feat.shape[0], self.hidden_dim).to(self.device)
            # for i in range(self.hop_num):
            #     h_out, (h, c) = self.neihop_select(h_e[i].unsqueeze(0), (h, c))
            #     h_out = h_out.mean(0)
        else:
            h_out = h_e[0]
        return self.out_layer(h_out)
        
        
'''
class MSGCN(nn.Module):
    r"""
        filterbank:str,optional
            From the perspective of spectral domain, implement different forms of filters
            ALL:Low-Pass Filter + High-Pass Filter + Band-Pass Filter
            LP+HP:Low-Pass Filter + High-Pass Filter
            LP+BP:Low-Pass Filter + Band-Pass Filter
            BP+HP:Band-Pass Filter + High-Pass Filter
            LP:Low-Pass Filter
            HP:High-Pass Filter
            BP:Band-Pass Filter
            default:LP
            调参后，确定模型架构为LP+HP并列两个单层，拼接后送入单层GAT
    """
    def __init__(self,
                graph,
                in_feats,
                hidden_feats,
                hidden_channel,
                mlp_dim,
                out_feats,
                norm=None,
                feat_drop=0,
                attr_drop=0,
                activation=F.relu,
                filterbank = 'LP',
                self_loop = False
                ):
        super(MSGCN, self).__init__()
        self.graph = graph
        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats
        self.feat_drop = feat_drop
        self.attr_drop = attr_drop
        self.activation = activation
        self.filterbank = filterbank
        self.self_loop = self_loop        
        
        # self.mlp = MLP(in_feats, mlp_dim)
        if filterbank == 'LP':
            self.filter = LP_Filter(in_feats, hidden_feats, out_feats,  feat_drop=feat_drop, self_loop=self_loop)
        elif filterbank == 'HP':
            self.filter = HP_Filter(in_feats, hidden_feats, out_feats,  feat_drop=feat_drop, self_loop=self_loop)
        elif filterbank == 'BP':
            self.filter = BP_Filter(in_feats, hidden_feats, out_feats,  feat_drop=feat_drop, self_loop=self_loop)
        elif filterbank == 'LP+HP':
            self.filter_layers = nn.ModuleList()
            # self.filter_layers.append(LPLayer_single(mlp_dim, hidden_feats, feat_drop=self.feat_drop, self_loop=self_loop))
            # self.filter_layers.append(HPLayer_single(mlp_dim, hidden_feats, feat_drop=self.feat_drop, self_loop=self_loop))
            self.filter_layers.append(LPLayer_single(in_feats, hidden_feats, feat_drop=self.feat_drop, self_loop=self_loop))
            self.filter_layers.append(HPLayer_single(in_feats, hidden_feats, feat_drop=self.feat_drop, self_loop=self_loop))
            # self.channel_attr = ChannelAttention(hidden_feats, hidden_channel)
        elif filterbank == 'LP+BP':
            self.filter_layers = nn.ModuleList()
            self.filter_layers.append(LPLayer_single(in_feats, hidden_feats, feat_drop=self.feat_drop, self_loop=self_loop))
            self.filter_layers.append(BPLayer_single(in_feats, hidden_feats, feat_drop=self.feat_drop, self_loop=self_loop))
        elif filterbank == 'HP+BP':
            self.filter_layers = nn.ModuleList()
            self.filter_layers.append(HPLayer_single(in_feats, hidden_feats, feat_drop=self.feat_drop, self_loop=self_loop))
            self.filter_layers.append(BPLayer_single(in_feats, hidden_feats, feat_drop=self.feat_drop, self_loop=self_loop))
        else:
            # self.filter = nn.ModuleList()
            # self.filter.append(LP_Filter(in_feats, hidden_feats, out_feats,  feat_drop=feat_drop, self_loop=self_loop))
            # self.filter.append(HP_Filter(in_feats, hidden_feats, out_feats,  feat_drop=feat_drop, self_loop=self_loop))
            # self.filter.append(BP_Filter(in_feats, hidden_feats, out_feats,  feat_drop=feat_drop, self_loop=self_loop))            
            self.filter_layers = nn.ModuleList()
            self.filter_layers.append(LPLayer_single(in_feats, hidden_feats, feat_drop=self.feat_drop, self_loop=self_loop))
            self.filter_layers.append(HPLayer_single(in_feats, hidden_feats, feat_drop=self.feat_drop, self_loop=self_loop))
            self.filter_layers.append(BPLayer_single(in_feats, hidden_feats, feat_drop=self.feat_drop, self_loop=self_loop))
            # self.channel_attr = ChannelAttention(out_feats, hidden_channel)
            # self.gat = GATConv(out_feats*3, out_feats, 1, feat_drop, attr_drop, activation=F.elu, allow_zero_in_degree=True, bias=False)           
        #注意力模型:先采用LPLayer_single、HPLayer_single、BPLayer_single分别提取相应频率信息并变换到hidden空间，然后再利用一层GAT
        if filterbank == 'LP+HP' or filterbank == 'LP+BP' or filterbank == 'HP+BP':
            #单独注意力层
            self.gat_layers = GATConv(hidden_feats*2, out_feats, 1, feat_drop, attr_drop, activation=F.elu, allow_zero_in_degree=True, bias=False)
            #配合ChannelAttention
            # self.gat_layers = GATConv(hidden_feats, out_feats, 1, feat_drop, attr_drop, activation=F.elu, allow_zero_in_degree=True, bias=False)
        elif filterbank == 'ALL':
            self.gat_layers = GATConv(hidden_feats*3, out_feats, 1, feat_drop, attr_drop, activation=F.elu, allow_zero_in_degree=True, bias=False)

                
    def forward(self, input, edge_weight=None):
        if self.filterbank == 'LP' or self.filterbank == 'HP' or self.filterbank == 'BP':
            # input = self.mlp(input)
            logits= self.filter(self.graph, input).flatten(1)
            return logits
        # if self.filterbank == 'ALL':
        #     channel_embeddings = []
        #     for i in range(len(self.filter)):
        #         channel_embeddings.append(self.filter[i](self.graph, input).flatten(1))
        #     # channel_embeddings = torch.stack(channel_embeddings, dim=1)
        #     # return self.gat(self.graph, self.channel_attr(channel_embeddings)).mean(1)
        else:
            channel_embeddings = []
            # input = self.mlp(input)
            for i in range(len(self.filter_layers)):
                channel_embeddings.append(self.filter_layers[i](self.graph, input).flatten(1))
            # return self.gat_layers(self.graph, self.channel_attr(torch.stack(channel_embeddings, dim=1))).mean(1) 
            return self.gat_layers(self.graph, torch.hstack(channel_embeddings)).mean(1) 
'''  

class MLP(nn.Module):
    def __init__(self, in_size, hidden, out_size, dropout=0):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(in_size, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_size)
        )     
    def forward(self,feat):
        return self.mlp(self.dropout(feat))

class MLPGCN(nn.Module):
    def __init__(self,
                graph,
                in_feats,
                hidden_feats,
                hidden_channel,
                mlp_dim,
                out_feats,
                norm=None,
                feat_drop=0,
                attr_drop=0,
                activation=F.relu,
                filterbank = 'LP',
                self_loop = False
                ):
        super(MLPGCN, self).__init__()
        self.graph = graph
        self.in_feats = in_feats
        self.hidden_feats = hidden_feats
        self.out_feats = out_feats
        self.attr_drop = attr_drop
        self.activation = activation
        self.filterbank = filterbank
        self.self_loop = self_loop
                
        self.feat_drop = nn.Dropout(feat_drop)
        #模型1说明：线性层+LP_single + or HP_single  ——> 层间attention ——> 线性层
        self.fc1 = nn.Linear(in_feats, hidden_feats)
        if filterbank == 'LP':#Test weighted F1: 93.67%   Test accuracy: 94.05%
            self.filter1 = LPLayer_single(in_feats, hidden_feats,  feat_drop=feat_drop, self_loop=self_loop)
            self.filter2 = LPLayer_single(hidden_feats, hidden_feats,  feat_drop=feat_drop, self_loop=self_loop)
        elif filterbank == 'HP':#Test weighted F1: 93.66%   Test accuracy: 94.08%
            self.filter1 = HPLayer_single(in_feats, hidden_feats,  feat_drop=feat_drop, self_loop=self_loop)
            self.filter2 = HPLayer_single(hidden_feats, hidden_feats,  feat_drop=feat_drop, self_loop=self_loop)
        # elif filterbank == 'BP':  #Test weighted F1: 93.75%   Test accuracy: 94.14%
        #     self.filter = BPLayer_single(in_feats, hidden_feats,  feat_drop=feat_drop, self_loop=self_loop)
        self.channel_attr = ChannelAttention(hidden_feats, hidden_channel)
        self.fc2 = nn.Linear(hidden_feats, out_feats)
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2.weight, gain=1.414)
        
        # 模型2说明：线性层+LP_single + or HP_single  ——> 层间attention ——> gat
        # self.fc1 = nn.Linear(in_feats, hidden_feats)
        # self.gat = GATConv(in_feats, hidden_feats, 1, feat_drop, attr_drop, activation=F.elu, allow_zero_in_degree=True, bias=False)
        # if filterbank == 'LP':
        #     self.filter = LPLayer_single(in_feats, hidden_feats,  feat_drop=feat_drop, self_loop=self_loop)
        # elif filterbank == 'HP':
        #     self.filter = HPLayer_single(in_feats, hidden_feats,  feat_drop=feat_drop, self_loop=self_loop)
        # self.channel_attr = ChannelAttention(hidden_feats, hidden_channel)
        # self.fc2 = nn.Linear(hidden_feats, out_feats)
        # nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        # nn.init.xavier_normal_(self.fc2.weight, gain=1.414)

    #模型2    
    # def forward(self, feat):
    #     h_1 = torch.tanh(self.fc1(self.feat_drop(feat)))
    #     h_2 = self.gat(self.graph, feat).mean(1)
    #     return self.fc2(self.feat_drop(self.channel_attr(torch.stack([h_1, h_2], dim=1))))
    
    #模型1    
    def forward(self, feat):
        h_1 = torch.tanh(self.fc1(self.feat_drop(feat)))
        # h_2 = self.filter2(self.graph, self.filter1(self.graph, feat))
        h_2 = self.filter1(self.graph, feat)
        return self.fc2(self.feat_drop(self.channel_attr(torch.stack([h_1,h_2], dim=1))))
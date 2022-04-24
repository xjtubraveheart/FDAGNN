from operator import concat
from tkinter import N
from turtle import forward
import torch
import torch.nn as nn
from torch.nn import init, LSTM, GRU
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn import GATConv
from layer import ChannelAttention
from layer import DiffAttention, HopsAttention
import numpy as np

class DiffGCN(nn.Module):
    def __init__(self,
                in_dim,
                hidden_dim,
                out_dim,
                hop_num=1,
                feat_drop=0,
                attn_drop=0,
                # gru_drop=0, #04-19
                device='cuda'
                ):
        super(DiffGCN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.hop_num = hop_num
        self.device = device
        self.feat_drop = nn.Dropout(feat_drop)
        #04-20 version
        # self.fc = nn.Linear(in_dim, hidden_dim, bias=False)
        # nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        
        self.diff_layer = nn.ModuleList()
        self.fusion = nn.ModuleList()
        # self.fusion.append(GRU(hidden_dim, hidden_dim, bias=False))
        # self.diff_layer.append(DiffAttention(g[0], in_dim, hidden_dim, feat_drop))        
        for i in range(hop_num):#DiffAttention层堆叠：提取中心节点与多跳邻居节点的特征差异
            self.diff_layer.append(DiffAttention(in_dim, hidden_dim, feat_drop, attn_drop, device))
            self.fusion.append(GRU(hidden_dim, hidden_dim, bias=False))
        self.out_layer = nn.Linear(hidden_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.out_layer.weight, gain=1.414)
        
        #新增邻居阶次的自适应选择
        # if hop_num > 1:
            # self.neihop_select = LSTM(hidden_dim, hidden_dim)
        # self.neihop_select = GRU(hidden_dim, hidden_dim, 1, bias=False)
            # self.neihop_select = GRU(hidden_dim, hidden_dim, dropout=gru_drop)#04-19
            # self.neihop_select = HopsAttention(hidden_dim)            
           
    # def forward(self, g):
    def forward(self, g, h_src, h_dst):
        h_e = []
        # h_self = self.fc(h_dst.float().to(self.device))#04-20 version
        for i in range(self.hop_num):
            # h_k = self.diff_layer[i](g[i].to(self.device))
            h_k = self.diff_layer[i](g[i], h_src[i], h_dst)
            h_e.append(h_k)
        #新增邻居阶次的自适应选择
        # if self.hop_num > 1:
            # 点积注意力
            # h_out = self.neihop_select(h_e, self.hop_num)
            
        h = torch.zeros(1, h_dst.shape[0], self.hidden_dim).to(self.device)
        # h_in = self.fc(self.feat_drop(h_dst.float())).unsqueeze(0)#04-23
        # h_out, h = self.fusion[0](h_in, h)#04-23
        for i in range(self.hop_num):
            h_out, h = self.fusion[i](h_e[i].unsqueeze(0), h)
            # h_out, h = self.fusion[i+1](h_e[i].unsqueeze(0), h)#04-23
        h_out = h_out.mean(0)
            
            # LSTM
            # h = torch.zeros(1, feat.shape[0], self.hidden_dim).to(self.device)
            # c = torch.zeros(1, feat.shape[0], self.hidden_dim).to(self.device)
            # for i in range(self.hop_num):
            #     h_out, (h, c) = self.neihop_select(h_e[i].unsqueeze(0), (h, c))
            #     h_out = h_out.mean(0)
        # else:
        #     h_out = h_e[0]#04-17 version
            # h_out = F.elu(h_self + h_e[0])#04-20 version            
        return self.out_layer(h_out)
        
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
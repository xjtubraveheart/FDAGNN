import torch
import torch.nn as nn
from torch.nn import GRU
import torch.nn.functional as F
from layer import DiffAttention

class DiffGCN(nn.Module):
    def __init__(self,
                in_dim,
                hidden_dim,
                out_dim,
                num_layer=1,
                feat_drop=0,
                attn_drop=0,
                use_diff=True,
                device='cuda'
                ):
        super(DiffGCN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.use_diff = use_diff
        self.device = device
        self.feat_drop = nn.Dropout(feat_drop)

        self.fc = nn.Linear(in_dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        
        self.fc = nn.ModuleList()
        self.diff_layer = nn.ModuleList()
        self.fusion = nn.ModuleList()
        self.fc.append(nn.Linear(in_dim, hidden_dim, bias=False))
        if use_diff:        
            self.diff_layer.append(DiffAttention(in_dim, hidden_dim, feat_drop, attn_drop, device))
            self.fusion.append(GRU(hidden_dim, hidden_dim, bias=False))
            self.fusion.append(GRU(hidden_dim, hidden_dim, bias=False))
        for i in range(num_layer):  
            self.fc.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            if use_diff:      
                self.diff_layer.append(DiffAttention(hidden_dim, hidden_dim, feat_drop, attn_drop, device))
                self.fusion.append(GRU(hidden_dim, hidden_dim, bias=False))
                self.fusion.append(GRU(hidden_dim, hidden_dim, bias=False))
        for i in range(num_layer):
            nn.init.xavier_uniform_(self.fc[i].weight, gain=1.414)
        self.out_layer = nn.Linear(hidden_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.out_layer.weight, gain=1.414)            
           
    def forward(self, g, feat):
        for i in range(self.num_layer):
            h_src = feat[g[i].srcdata['_ID']]
            h_dst = feat[g[i].dstdata['_ID']]
            h_d = self.diff_layer[i](g[i], h_src, h_dst)   
            h_0 = torch.zeros(1, h_dst.shape[0], self.hidden_dim).to(self.device)
            h_f = self.fc[i](self.feat_drop(h_dst.float())).unsqueeze(0)
            h_f = F.elu(h_f)
            h_out, h = self.fusion[i * self.num_layer](h_f, h_0)
            h_out, h = self.fusion[i * self.num_layer + 1](h_d.unsqueeze(0), h)
            h_out = h_out.mean(0)
            feat = h_out
        return self.out_layer(feat)
        
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
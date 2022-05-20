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
        self.act = nn.Tanh()
        '''
        # not use diff, equal two-layer mlp
        self.fc = nn.Linear(in_dim, hidden_dim, bias=False)
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        '''
        
        self.fc = nn.ModuleList()
        self.diff_layer = nn.ModuleList()
        '''
        self.fuse = nn.Linear(hidden_dim, 1, bias=False)
        self.act  = nn.Tanh()
        nn.init.xavier_uniform_(self.fuse.weight, gain=1.414)
        '''
        self.fusion = nn.ModuleList()#GRU fusion
        # self.fuse = nn.ModuleList()#weight fusion
        self.fc.append(nn.Linear(in_dim, hidden_dim, bias=False))
        if use_diff:        
            self.diff_layer.append(DiffAttention(in_dim, hidden_dim, feat_drop, attn_drop, device))
            # self.fuse.append(nn.Linear(hidden_dim, 1, bias=False))
            self.fusion.append(GRU(hidden_dim, hidden_dim, bias=False))
            self.fusion.append(GRU(hidden_dim, hidden_dim, bias=False))
        for i in range(num_layer - 1):  
            self.fc.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            if use_diff:      
                self.diff_layer.append(DiffAttention(hidden_dim, hidden_dim, feat_drop, attn_drop, device))
                # self.fuse.append(nn.Linear(hidden_dim, 1, bias=False))
                self.fusion.append(GRU(hidden_dim, hidden_dim, bias=False))
                self.fusion.append(GRU(hidden_dim, hidden_dim, bias=False))
        for i in range(num_layer):
            nn.init.xavier_uniform_(self.fc[i].weight, gain=1.414)
            # nn.init.xavier_uniform_(self.fuse[i].weight, gain=1.414)
        #特征差异和原始特征拼接
        # self.out_layer_1 = nn.Linear(num_layer * 2 * hidden_dim, hidden_dim, bias=False)
        # self.out_layer_2 = nn.Linear(hidden_dim, out_dim, bias=False)
        # nn.init.xavier_uniform_(self.out_layer_1.weight, gain=1.414)            
        # nn.init.xavier_uniform_(self.out_layer_2.weight, gain=1.414)            
        self.out_layer = nn.Linear(hidden_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.out_layer.weight, gain=1.414)            
    
    #GRU融合
    # def forward(self, g, feat):
    #     for i in range(self.num_layer):
    #         h_src = feat[g[i].srcdata['_ID']]
    #         h_dst = feat[g[i].dstdata['_ID']]
    #         h_d = self.diff_layer[i](g[i], h_src, h_dst)   
    #         h_0 = torch.zeros(1, h_dst.shape[0], self.hidden_dim).to(self.device)
    #         h_f = self.fc[i](self.feat_drop(h_dst.float())).unsqueeze(0)
    #         h_f = F.elu(h_f)
    #         h_out, h = self.fusion[i * 2](h_f, h_0)
    #         h_out, h = self.fusion[i * 2 + 1](h_d.unsqueeze(0), h)
    #         h_out = h_out.mean(0)
    #         feat = h_out
    #     return self.out_layer(feat)
    
    #不融合，直接拼接
    # def forward(self, g, feat):
    #     for i in range(self.num_layer):
    #         h_src = feat[g[i].srcdata['_ID']]
    #         h_dst = feat[g[i].dstdata['_ID']]
    #         h_d = self.diff_layer[i](g[i], h_src, h_dst)
    #         h_f = self.fc[i](self.feat_drop(h_dst.float()))   
    #         feat = torch.cat((h_f, h_d), dim=1)
    #         feat = F.elu(self.out_layer_1(self.feat_drop(feat)))
    #     return self.out_layer_2(feat)

    #不使用特征差异，相当于MLP
    def forward(self, g, feat):
        for i in range(self.num_layer):
            h_src = feat[g[i].srcdata['_ID']]
            h_dst = feat[g[i].dstdata['_ID']]
            h_f = self.fc[i](self.feat_drop(h_dst.float()))
            feat = F.elu(h_f)
        return self.out_layer(feat)
    
    '''
    #initial features + diff-features
    def forward(self, g, feat):
        for i in range(self.num_layer):
            h_src = feat[g[i].srcdata['_ID']]
            h_dst = feat[g[i].dstdata['_ID']]
            h_d = self.diff_layer[i](g[i], h_src, h_dst)
            feat = F.elu(self.fc[i](self.feat_drop(h_dst.float()))) + h_d
        return self.out_layer(feat)
    '''
    '''
    # use weight to fuse initial features and diff-features
    def forward(self, g, feat):
        w = []
        h = []
        for i in range(self.num_layer):
            h_src = feat[g[i].srcdata['_ID']]
            h_dst = feat[g[i].dstdata['_ID']]
            h_d = self.diff_layer[i](g[i], h_src, h_dst)
            h_dst = self.fc[i](self.feat_drop(h_dst.float()))
            h.append(h_dst)
            h.append(h_d)
            w.append(self.act(self.fuse[i](h_dst)))
            w.append(self.act(self.fuse[i](h_d)))
            w = torch.stack(w, 0)
            w = torch.softmax(w, dim=0)
            h = torch.stack(h, 0)
            w = w.expand(h.shape)
            feat = (w * h).sum(0)
        return self.out_layer(feat)
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
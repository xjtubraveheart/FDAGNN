import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn

def diff(graph,feat):#对中心节点和其邻居节点特征差异直接做平均
    with graph.local_scope():
        graph.ndata['h'] = feat      
        graph.update_all(fn.u_sub_v('h', 'h', 'm'), fn.mean('m', 'diff'))
        return graph.dstdata['diff']

class DiffAttention(nn.Module):
    def __init__(self, in_dim, out_dim, feat_drop=0, attn_drop=0, device='cuda'):
        super(DiffAttention, self).__init__()
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.device = device
        
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        
        self.attn_fc = nn.Linear(out_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.attn_fc.weight, gain=1.414)
    
    def edge_attention(self, edges): 
        diff = edges.dst['h'] - edges.src['h']  
        diff = self.fc(self.feat_drop(diff.float()))
        a = self.attn_fc(diff)
        return {'e': torch.tanh(a), 'diff_v_sub_u': diff} 
    
    def message_func(self, edges):
        return {'diff_v_sub_u': edges.data['diff_v_sub_u'], 'e': edges.data['e']}
        
    def reduce_func(self, nodes):
        alpha = self.attn_drop(F.softmax(nodes.mailbox['e'], dim=1))
        h_diff = torch.sum(alpha * nodes.mailbox['diff_v_sub_u'], dim=1)
        return {'h_diff': h_diff}
       
    def forward(self, g, h_src, h_dst):  
        g.srcdata['h'] = h_src
        g.dstdata['h'] = h_dst
        g.apply_edges(self.edge_attention) 
        g.update_all(self.message_func, self.reduce_func)  
        logits = g.dstdata.pop('h_diff')
        return F.elu(logits)
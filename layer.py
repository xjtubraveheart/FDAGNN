from turtle import forward
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from dgl import function as fn
import dgl.ops as op 

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
        #04-17 version
        self.fc1 = nn.Linear(in_dim, out_dim, bias=False)
        self.fc2 = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.414)
        
        #04-20 version
        # self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # nn.init.xavier_uniform_(self.fc.weight, gain=1.414)
        
        self.attn_fc = nn.Linear(out_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.attn_fc.weight, gain=1.414)
    
    def edge_attention(self, edges):
        # diff = edges.dst['feat'] - edges.src['feat']    
        diff = edges.dst['h'] - edges.src['h']  
        # diff = self.fc1(diff.float())
        diff = self.fc2(self.feat_drop(diff.float()))#04-17 version
        # diff = self.fc(self.feat_drop(diff.float()))#04-20 version  
        a = self.attn_fc(diff)
        return {'e': torch.tanh(a), 'diff_v_sub_u': diff}  #将中心节点与其邻居节点的差异保存在边上  04-20 将Leak-Relu换为tanh
    
    def message_func(self, edges):
        return {'diff_v_sub_u': edges.data['diff_v_sub_u'], 'e': edges.data['e']}
        
    def reduce_func(self, nodes):
        alpha = self.attn_drop(F.softmax(nodes.mailbox['e'], dim=1))
        h_diff = torch.sum(alpha * nodes.mailbox['diff_v_sub_u'], dim=1)
        return {'h_diff': h_diff}
       
    # def forward(self, g):    
    def forward(self, g, h_src, h_dst):    
        # h_src =  self.fc1(h_src.float())      
        # h_dst =  self.fc1(h_dst.float()) 
        g.srcdata['h'] = h_src
        g.dstdata['h'] = h_dst
        g.apply_edges(self.edge_attention) 
        g.update_all(self.message_func, self.reduce_func) 
        # logits = h_dst + g.dstdata.pop('h_diff')
        
        #04-17 version       
        # logits = self.fc1(g.dstdata['feat'].float()) + g.dstdata.pop('h_diff') 
        logits = self.fc1(h_dst.float()) + g.dstdata.pop('h_diff')      
        return F.elu(logits)
        
        #04-20 version
        # return g.dstdata.pop('h_diff')

class HopsAttention(nn.Module):
    def __init__(self, in_size, dropout=0):
        super(HopsAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Linear(in_size, 1, bias=False)
        self.activation = nn.Sigmoid()
        #注意力评分函数
        # self.project = nn.Sequential(
        #     nn.Linear(in_size, 1),
        #     nn.Sigmoid()
        # )  
        nn.init.xavier_normal_(self.attention.weight, gain=1.414)       
    def forward(self, h, hop_num):
        w = []
        for i in range(hop_num):
            w.append(self.activation(self.attention(self.dropout(h[i]))))
        w = torch.stack(w, 0)
        w = torch.softmax(w, dim=0)
        h = torch.stack(h, 0)
        w = w.expand(h.shape)
        return (w * h).sum(0)
        
class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # 公式 (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # 公式 (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        # 仿照AdaGCN增加的
        self.reg_params = list(self.fc.parameters())

    def edge_attention(self, edges):
        # 公式 (2) 所需，边上的用户定义函数
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # 公式 (3), (4)所需，传递消息用的用户定义函数
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # 公式 (3), (4)所需, 归约用的用户定义函数
        # 公式 (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # 公式 (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # 公式 (1)
        # print(h.size())
        z = self.fc(h)
        # 报错修改：https://blog.csdn.net/weixin_42815609/article/details/113481943
        self.g = self.g.to(torch.device('cuda:0'))
        self.g.ndata['z'] = z
        # 公式 (2)
        self.g.apply_edges(self.edge_attention)
        # 公式 (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h'),self.reg_params,self.g.edata.pop('e')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        # self.reg_params=[]
        # self.head_outs=[]
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        self.reg_params=[]
        self.head_outs = []
        self.attention=[]
        for attn_head in self.heads:
            embedding,reg_params,attention=attn_head(h)
            self.head_outs.append(embedding)
            self.reg_params.append(reg_params)
            self.attention.append(attention)
        # head_outs = [attn_head(h) for attn_head in self.heads]

        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            return torch.cat(self.head_outs, dim=1),self.reg_params,torch.cat(self.attention, dim=1)
        else:
            # 用求平均整合多头结果
            return torch.mean(torch.stack(self.head_outs),dim=0),self.reg_params

def neigh_compute(graph, feat):
    graph = graph.local_var()     
    with graph.local_scope():
        #抽取起点特征、终点特征，再加dropout。注意：block中终点总是被包含在起点的最前面。
        # feat_src = feat_dst = feat
        # if graph.is_block:
        #         feat_dst = feat_src[:graph.number_of_dst_nodes()]
        
        #提前记录目标节点的原始特征
        # graph.srcdata['h'] = feat_src
        
        #此为消息函数，把起点的h特征拷贝到边的m特征上。copy_src为内置函数。
        # msg_fn = fn.copy_src('h', 'm')
        
        #Handle the case of graphs without edges
        # if graph.number_of_edges() == 0:
        #     graph.dstdata['neigh'] = torch.zeros(
        #         feat_dst.shape[0], self.in_feats).to(feat_dst)
        #Message Passing,fn.mean:将邻居节点特征的平均值作为目标节点的neigh
        graph.srcdata['h'] = feat
        graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
        # for i, node_index in enumerate(node_index_edge):   
        #     graph.dstdata['neigh'][node_index] = torch.zeros_like(graph.dstdata['neigh'][node_index])
        # h_neigh = graph.dstdata['neigh']        
        return graph.dstdata['neigh']

class ChannelAttention(nn.Module):
    def __init__(self, in_size, hidden_size=32, dropout=0):
        super(ChannelAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        #注意力评分函数：点积
        # self.project = nn.Sequential(
        #     nn.Linear(in_size, 1),
        # )
        #注意力评分函数：加性模型，单线性层
        self.project = nn.Sequential(
            nn.Linear(in_size, 1),
            nn.Tanh()
        )
        #注意力评分函数：加性模型，双线性层
        # self.project = nn.Sequential(
        #     nn.Linear(in_size, hidden_size),
        #     nn.Tanh(),
        #     nn.Linear(hidden_size, 1, bias=False)
        # )  
        nn.init.xavier_normal_(self.project[0].weight, gain=1.414) 
        # nn.init.xavier_normal_(self.project[2].weight, gain=1.414)      
    def forward(self, z):
        w = self.dropout(self.project(z)).mean(0)      # (M, 1)
        beta = w
        # w = self.project(z).mean(0)                  # (M, 1)  
        beta = torch.softmax(beta, dim=0)            # (M, 1)
        # beta = w.expand((z.shape[0],) + w.shape) # (N, M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)

        return (beta * z).sum(1)                       # (N, D * K)
    



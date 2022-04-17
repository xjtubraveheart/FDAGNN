from pickle import FALSE
from statistics import mean
import time, datetime
import argparse, logging
import numpy as np
from numpy import random
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import dgl
from dgl import load_graphs
from dgl.data.utils import load_info
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from model import MLP, DiffGCN
from utils import EarlyStopping
from sklearn.metrics import auc, f1_score, classification_report, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import tqdm
# the first thought
def setup_seed(seed):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)
    dgl.random.seed(seed)
def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def evaluate(model, g, node_id, feat, labels, n_classes, device, batch_size, loss_fcn, num_workers, neigh_hops):
    model.eval()
    pred = torch.zeros(g[0].num_nodes(), n_classes).to(torch.device('cuda', 0))
    feat, labels, node_id = load_sample_data(feat, labels, node_id, device)
    with torch.no_grad():
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
        dataloaders = []
        for i in range(args.neigh_hop):
            g_t = g[i].to(device)
            dataloader = dgl.dataloading.DataLoader(
                g_t,
                node_id,
                sampler,
                device=device,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers)
            dataloaders.append(dataloader)
        for (neigbor_nodes_1, target_nodes_1, blocks_1), (neigbor_nodes_2, target_nodes_2, blocks_2) in zip(dataloaders[0], dataloaders[-1]): 
            batch_srcnodes_1, batch_dstnodes, _ = load_subtensor(feat, labels, target_nodes_1, neigbor_nodes_1, device) 
            batch_srcnodes_2, _, _ = load_subtensor(feat, labels, target_nodes_2, neigbor_nodes_2, device)
            blocks = [blocks_1[0].int().to(device), blocks_2[0].int().to(device)]
            batch_srcnodes = [batch_srcnodes_1, batch_srcnodes_2]
            batch_pred = model(blocks, batch_srcnodes, batch_dstnodes)
            pred[target_nodes_1] = batch_pred
        labels = labels.to(torch.device('cuda', 0)) 
        # model.train()
        return accuracy(pred, labels), loss_fcn(pred, labels), pred

def gen_mask_category(g, train_per_class_num, val_num, test_num):
    labels = g.ndata['label']
    g.ndata['label'] = labels.long()
    labels = np.array(labels)
    rng = np.random.default_rng(42)
    train_idx_split = []
    n_nodes = len(labels)
    all_idx = np.arange(n_nodes)
    for i in range(max(labels) + 1):
        train_idx_split.append(rng.choice(all_idx[labels == i], train_per_class_num, replace=False))
    train_idx = np.concatenate(train_idx_split)
    exclude_train_idx = np.array([i for i in all_idx if i not in train_idx])
    val_idx = rng.choice(exclude_train_idx, val_num, replace=False)
    ex_train_val_idx = np.array([i for i in exclude_train_idx if i not in val_idx])
    test_idx = rng.choice(ex_train_val_idx, test_num, replace=False)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    return g, train_idx

def gen_mask(g, train_rate, val_rate):
    # train_per_class_num=int(g.num_nodes()*train_rate/num_classes)
    # val_num=int(g.num_nodes()*val_rate)
    # test_num=int(g.num_nodes()*(1-train_rate-val_rate))
    labels = g.ndata['label']
    g.ndata['label'] = labels.long()
    labels = np.array(labels)
    n_nodes = len(labels)
    index=list(range(n_nodes))
    train_idx, val_test_idx, _, y_validate_test = train_test_split(index, labels, stratify=labels, train_size=train_rate,test_size=1-train_rate,
                                                 random_state=2, shuffle=True)
    val_idx, test_idx, _, _ = train_test_split(val_test_idx,y_validate_test, train_size=val_rate/(1-train_rate), test_size=1-val_rate/(1-train_rate),
                                                     random_state=2, shuffle=True)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    return g

def inductive_split(g):
    """Split the graph into training graph, validation graph, and test graph by training
    and validation masks.  Suitable for inductive models."""
    train_g = g.subgraph(g.ndata['train_mask'])
    val_g = g.subgraph(g.ndata['val_mask'])
    test_g = g.subgraph(g.ndata['test_mask'])
    return train_g, val_g, test_g

# 修改日志记录为北京时间
def beijing(sec,what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()

#剔除掉度为1节点的入边
def del_edgenode_indegree(graph):
    graph = dgl.remove_self_loop(graph)
    node_degree = np.array(graph.in_degrees())
    node_index_edge = np.argwhere(node_degree == 1)
    node_index_edge = torch.tensor(node_index_edge.reshape(len(node_index_edge)))
    graph.remove_edges(graph.in_edges(node_index_edge,'eid'))
    return dgl.add_self_loop(graph)

#目前实现二跳邻居图
def compute_adj_K(graph, K):
    if K == 1:
        return graph
    adj_coo = graph.adj_sparse('coo')
    i=adj_coo[0].numpy()
    j=adj_coo[1].numpy()
    data=np.ones_like(i)
    adj_coo_torch=torch.sparse_coo_tensor(torch.tensor([i,j]),torch.tensor(data), [graph.num_nodes(), graph.num_nodes()]) 
    adj_coo_torch=adj_coo_torch.float()
    adj_coo_K = adj_coo_torch
    for i in range(K-1):
        adj_coo_K = torch.sparse.mm(adj_coo_K, adj_coo_torch)#稀疏型：A^2，'COO'
    indices = adj_coo_K._indices().detach().numpy()
    row = indices[0]
    col = indices[1]
    values = adj_coo_K._values().detach().numpy()
    not_K_neigh = np.argwhere(values > 1)
    values = np.delete(values, not_K_neigh)
    row = np.delete(row, not_K_neigh)
    col = np.delete(col, not_K_neigh)
    coo = sp.coo_matrix((values, (row, col)), shape=(graph.num_nodes(), graph.num_nodes()))
    graph_K_hop = dgl.from_scipy(coo)
    graph_K_hop = dgl.remove_self_loop(graph_K_hop)
    graph_K_hop = dgl.add_self_loop(graph_K_hop)
    # graph_2_hop = del_edgenode_indegree(graph_2_hop)
    # adj_2_g=torch.sparse_coo_tensor(adj_coo_2._indices(),torch.tensor(values), [graph.num_nodes(), graph.num_nodes()])
    graph_K_hop.ndata['feat'] = graph.ndata['feat']
    graph_K_hop.ndata['label'] = graph.ndata['label']
    graph_K_hop.ndata['train_mask'] = graph.ndata['train_mask']
    graph_K_hop.ndata['val_mask'] = graph.ndata['val_mask']
    graph_K_hop.ndata['test_mask'] = graph.ndata['test_mask']
    return graph_K_hop

def load_subtensor(nfeat, labels, target_nodes, neigbor_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    if nfeat.device == device and labels.device == device:
        batch_srcnodes = nfeat[neigbor_nodes]
        batch_dstnodes = nfeat[target_nodes]
        batch_labels = labels[target_nodes]
    else:
        batch_srcnodes = nfeat[neigbor_nodes].to(device)
        batch_dstnodes = nfeat[target_nodes].to(device)
        batch_labels = labels[target_nodes].to(device)
    return batch_srcnodes, batch_dstnodes, batch_labels

def load_sample_data(nfeat, labels, nid, device):
    nfeat = nfeat.to(device)
    labels = labels.to(device)
    nid = nid.to(device)
    return nfeat, labels, nid

def main(args):
    # logging基础配置
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)
    logging.Formatter.converter = beijing
    log_name=(datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d')
    logging.basicConfig(
        format='%(asctime)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=20,
        filename=args.log_name + log_name + '.log',
        filemode='a'
        )
    
    #load dataset and preprocess
    if args.dataset == 'citeseer':
        dataset = CiteseerGraphDataset()
        n_classes = dataset.num_classes
        g = dataset[0]
        g = gen_mask_category(dataset[0],20,500,1000)
        label_names=['0','1','2','3','4','5']
    elif args.dataset == 'citeseer_sg':
        dataset, _ = load_graphs('/code/DiffGCN/data/citeseer_max_sub.bin')
        n_classes = 6
        # citeseer train:val:test:20,500,1000
        g = gen_mask_category(dataset[0],20,500,1000)
        label_names=['0','1','2','3','4','5']
    elif args.dataset == 'BUPT':
    # dataset, _ = load_graphs("./data/Sichuan_tele.bin")  # glist will be [g1]
        dataset, _ = load_graphs("/code/DiffGCN/data/BUPT_tele.bin")  # glist will be [g1]
        n_classes = load_info("/code/DiffGCN/data/BUPT_tele.pkl")['num_classes']
        # {0: 99861, 1: 8448, 2: 8074}
        #g = gen_mask(dataset[0], 6000,4000, 5000)
        # g, train_idx = gen_mask_same_num(dataset[0], 6000, 4000, 5000)
        g = gen_mask(dataset[0], args.train_ratio, args.val_ratio)
        label_names=['0','1','2']
        
    g_hops = []
    train_g, val_g, test_g = [], [], []
    for i in range(args.neigh_hop):
        g_hop = compute_adj_K(g, i+1)
        if args.edgenode:
            g_hop = del_edgenode_indegree(g_hop)
        g_hops.append(g_hop)
    for i in range(args.neigh_hop):
        train_g_s, val_g_s, test_g_s = inductive_split(g_hops[i])
        #下一步在GPU上进行采样，GPU采样只支持csc格式
        train_g_s = train_g_s.formats(['csc'])
        val_g_s = val_g_s.formats(['csc'])
        val_g_s = val_g_s.formats(['csc'])
        train_g.append(train_g_s)
        val_g.append(val_g_s)
        test_g.append(test_g_s)
        
    train_nfeat = train_g[0].ndata.pop('feat')
    val_nfeat = val_g[0].ndata.pop('feat')
    test_nfeat = test_g[0].ndata.pop('feat')
    train_labels = train_g[0].ndata.pop('label')
    val_labels = val_g[0].ndata.pop('label')
    test_labels = test_g[0].ndata.pop('label')
    num_feats = train_nfeat.shape[1]
    train_nid = torch.nonzero(train_g[0].ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(val_g[0].ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(test_g[0].ndata['test_mask'], as_tuple=True)[0]  
    
    train_nfeat, train_labels, train_nid = load_sample_data(train_nfeat, train_labels, train_nid, device)     
    # Create PyTorch DataLoader for constructing blocks
    if args.neigh_hop == 1:
        train_size = len(train_nid)
        val_size = len(val_nid)
        test_size = len(test_nid)
    else:
        train_size = len(train_nid) if args.hidden < 200 and args.train_ratio < 0.25 else args.batch_size
        val_size = len(val_nid) if args.hidden < 200 and args.val_ratio < 0.25 else args.batch_size
        train_size = len(train_nid) if args.hidden < 200 and args.train_ratio > 0.55 else args.batch_size 
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    train_dataloader = []
    for i in range(args.neigh_hop):
        g_t = train_g[i].to(device)
        dataloader = dgl.dataloading.DataLoader(
            g_t,
            train_nid,
            sampler,
            device=device,
            batch_size=train_size,
            shuffle=False,
            drop_last=False,
            num_workers=0)
        train_dataloader.append(dataloader)
    logging.log(23,f"---------------------dataset: {args.dataset}-------------------------------------------------------------")
    logging.log(23,f"train: {args.train_ratio * 100:.1f}% val: {args.val_ratio * 100:.1f}% hidden: {args.hidden} batch_size:{args.batch_size} max_hop:{args.neigh_hop} seed: {args.seed} lr: {args.lr} weight_decay: {args.weight_decay} epochs: {args.epochs} feat_drop: {args.feat_drop} attr_drop:{args.attr_drop}")
    model = DiffGCN(num_feats, args.hidden, n_classes, args.neigh_hop, args.feat_drop, device)
    if args.early_stop:
        stopper = EarlyStopping(args.patience)
    model.to(device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    start_time = time.time()
    last_time = start_time
    for epoch in range(args.epochs):
        model.train()
        train_acc = []
        batch_num_train = train_g[0].number_of_nodes() // train_size
        for step, ((neigbor_nodes_1, target_nodes_1, blocks_1), (neigbor_nodes_2, target_nodes_2, blocks_2)) in enumerate(zip(train_dataloader[0], train_dataloader[-1])): 
            model.train()
            batch_srcnodes_1, batch_dstnodes, batch_labels = load_subtensor(train_nfeat, train_labels, target_nodes_1, neigbor_nodes_1, device) 
            batch_srcnodes_2, _, _ = load_subtensor(train_nfeat, train_labels, target_nodes_2, neigbor_nodes_2, device)
            blocks = [blocks_1[0].int().to(device), blocks_2[0].int().to(device)]
            batch_srcnodes = [batch_srcnodes_1, batch_srcnodes_2]
            batch_pred = model(blocks, batch_srcnodes, batch_dstnodes)
            train_acc.append(accuracy(batch_pred, batch_labels))
            loss = loss_fcn(batch_pred, batch_labels)
            if step == batch_num_train or train_size == len(train_nid):
                val_acc, val_loss, _ = evaluate(model, val_g, val_nid, val_nfeat, val_labels, n_classes, device, val_size, loss_fcn, 0, args.neigh_hop) 
                torch.cuda.empty_cache()
            optimizer.zero_grad()
            loss.backward()      
            optimizer.step()                                                    
            # iter_tput.append(time.time() - tic_step)
            # # if step % args.log_every == 0:
            # # if step == 0:
            # acc = accuracy(batch_pred, batch_labels)
            # gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
            # print('Epoch {:04d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
            #     epoch, loss.item(), acc, np.mean(iter_tput[3:]), gpu_mem_alloc))
            # tic_step = time.time() 
                
        if epoch % args.print_interval == 0 and epoch != 0:
            duration = time.time() - last_time  # each interval including training and early-stopping
            last_time = time.time()
            logging.info(f"Epoch {epoch}: "
                        f"Train_loss = {loss:.6f},"
                        f"Train_acc = {mean(train_acc) * 100:.4f},"
                        f"Val_loss = {val_loss:.6f}, "
                        f"Val_acc = {val_acc * 100:.4f} "
                        f"({duration:.3f} sec)")
        if args.early_stop and epoch != 0:
            if stopper.step(val_acc, model, epoch, val_loss):
                break   

    #test
    runtime = time.time() - start_time
    if args.early_stop:
        logging.log(21, f"best epoch: {stopper.best_epoch}, best val acc:{stopper.best_score * 100:.4f}， val_loss:{stopper.best_epoch_val_loss:.6f}, ({runtime:.3f} sec)")
    if args.early_stop:
        model.load_state_dict(torch.load('/code/DiffGCN/es_checkpoint.pt'))
    if args.train_ratio < 0.5: 
        device = torch.device('cpu')
    test_acc, _, test_logits = evaluate(model, test_g, test_nid, test_nfeat, test_labels, n_classes, device, test_size, loss_fcn, 0, args.neigh_hop)
    test_h = torch.argmax(test_logits, dim=1)
    test_f1 = f1_score(test_labels.cpu(), test_h.cpu(), average='weighted')
    report = classification_report(test_labels.cpu().detach().numpy(), test_h.cpu().detach().numpy(), target_names=label_names, digits=4)
    auc = roc_auc_score(test_labels.cpu(), torch.softmax(test_logits, dim=1).cpu(), average='weighted', multi_class='ovo')
    logging.log(23,f"Test:  Weighted_F1: {test_f1 * 100:.4f}%   Accuracy: {test_acc * 100:.4f}%    Weighted_AUC: {auc * 100:.4f}%")
    logging.log(23,f'Report:\n {report}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DiffGCN')
    parser.add_argument('--dataset', type=str, default='BUPT', help='cora, citeseer, citeseer_sg, pubmed, BUPT, BUPT_SG')
    parser.add_argument("--feat_drop", type=float, default=0, help="Input dropout probability")
    parser.add_argument("--attr_drop", type=float, default=0, help="attention dropout probability")
    parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')   
    parser.add_argument('--neigh_hop', type=int, default=1, help='K-hop neighbors.')   
    parser.add_argument('--lr', type=float, default=4e-3, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--early_stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--epochs', type=int, default=4000, help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=2000, help='Patience in early stopping')
    parser.add_argument('--train_ratio', type=float, default=0.2, help='Ratio of training set')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of valing set')
    parser.add_argument('--seed', type=int, default=42, help="seed for our system")
    parser.add_argument('--print_interval', type=int, default=100, help="the interval of printing in training")
    parser.add_argument('--log_name', type=str, default='DiffGCN', help="Name for logging")
    parser.add_argument('--edgenode', type=bool, default=True, help="edge node not converge other nodes")
    parser.add_argument('--batch_size', type=int, default=4096)
    # parser.add_argument('--val_batch_size', type=int, default=1024)
    # parser.add_argument('--test_batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    # parser.add_argument('--sample-gpu', action='store_true',
    #                        help="Perform the sampling process on the GPU. Must have 0 workers.")
    args = parser.parse_args()
       
    main(args)
from pickle import FALSE
import time, datetime
import argparse, logging
import numpy as np
from numpy import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import dgl
from dgl import load_graphs
from dgl.data.utils import load_info
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from model import MLP, MLPGCN, DiffGCN
# from model import MSGCN, MLP, MLPGCN
from utils import EarlyStopping
from sklearn.metrics import f1_score, classification_report, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# the first thought
def setup_seed(seed):
    random.seed(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def evaluate(model, features, labels, mask, loss_fcn):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        loss = loss_fcn(logits, labels) 
        return accuracy(logits, labels), loss, logits

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
    return g,train_idx

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

def main(args):
    # logging基础配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        g, train_idx = gen_mask_category(dataset[0],20,500,1000)
        label_names=['0','1','2','3','4','5']
    elif args.dataset == 'citeseer_sg':
        dataset, _ = load_graphs('/code/DiffGCN/data/citeseer_max_sub.bin')
        n_classes = 6
        # citeseer train:val:test:20,500,1000
        g, train_idx = gen_mask_category(dataset[0],20,500,1000)
        label_names=['0','1','2','3','4','5']
    elif args.dataset == 'BUPT':
    # dataset, _ = load_graphs("./data/Sichuan_tele.bin")  # glist will be [g1]
        dataset, _ = load_graphs("/code/DiffGCN/data/BUPT_tele.bin")  # glist will be [g1]
        n_classes = load_info("/code/DiffGCN/data/BUPT_tele.pkl")['num_classes']
        graph = dataset[0]
        # {0: 99861, 1: 8448, 2: 8074}
        #g = gen_mask(dataset[0], 6000,4000, 5000)
        # g, train_idx = gen_mask_same_num(dataset[0], 6000, 4000, 5000)
        g, train_idx = gen_mask(dataset[0], 0.2, 0.2) 
        label_names=['0','1','2']     
    if args.edgenode:
        g = del_edgenode_indegree(g)
    g = g.to(device)
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)    
    features = g.ndata['feat'].float()
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    num_edges = g.num_edges()
    logging.log(23,f"---------------------dataset: {args.dataset}-------------------------------------------------------------")
    logging.log(23,f"filterbanks: {args.filterbanks} hidden: {args.hidden} seed: {args.seed} lr: {args.lr} weight_decay: {args.weight_decay} feat_drop: {args.feat_drop} attr_drop:{args.attr_drop} self_loop:{args.self_loop} hidden_channel:{args.hidden_channel}")
    # model = MSGCN(g,
    # model = MLPGCN(g,
    #               num_feats,
    #               args.hidden,
    #               args.hidden_channel,
    #               args.mlp_dim,
    #               n_classes,
    #               None,
    #               args.feat_drop,
    #               args.attr_drop,
    #               filterbank = args.filterbanks,
    #               self_loop = args.self_loop
    #             )
    model = MLP(num_feats, args.hidden, n_classes, args.feat_drop)
    # model = DiffGCN(g, num_feats, args.hidden, n_classes, 1, args.feat_drop)
    if args.early_stop:
        stopper = EarlyStopping(args.patience)
    model.to(device)
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    start_time = time.time()
    last_time = start_time
    
    for epoch in range(args.epochs):
        # if hasattr(torch.cuda, 'empty_cache'):
        #     torch.cuda.empty_cache()
        model.train()
        logits = model(features)
        loss = loss_fcn(logits[train_mask],labels[train_mask])
        train_acc = accuracy(logits[train_mask], labels[train_mask])
        train_loss = loss.item() * 1.0        
        val_acc, val_loss, _ = evaluate(model, features, labels, val_mask, loss_fcn) 
                
        if epoch % args.print_interval == 0:
            duration = time.time() - last_time  # each interval including training and early-stopping
            last_time = time.time()
            logging.info(f"Epoch {epoch}: "
                        f"Train loss = {train_loss:.6f}, "
                        f"Train acc = {train_acc * 100:.2f}, "
                        f"Validation loss = {val_loss:.6f}, "
                        f"Validation acc = {val_acc * 100:.2f} "
                        f"({duration:.3f} sec)")
        if args.early_stop:
            if stopper.step(val_acc, model, epoch, train_loss, val_loss):
                break
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  

    #test
    runtime = time.time() - start_time
    if args.early_stop:
        logging.log(21, f"best epoch: {stopper.best_epoch}, best val acc:{stopper.best_score * 100:.4f}， ({runtime:.3f} sec)")
        logging.log(21, f"train_loss in this epoch :{stopper.best_epoch_train_loss:.6f}, val_loss in this epoch :{stopper.best_epoch_val_loss:.6f}")
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))
    with torch.no_grad():    
        test_acc, test_loss, test_logits = evaluate(model, features, labels, test_mask, loss_fcn)
    test_h = torch.argmax(test_logits, dim=1)
    test_f1 = f1_score(labels[test_mask].cpu(), test_h.cpu(), average='weighted')
    report = classification_report(labels[test_mask].cpu().detach().numpy(), test_h.cpu().detach().numpy(), target_names=label_names, digits=4)
    logging.log(23,f"Test weighted F1: {test_f1 * 100:.2f}%   Test accuracy: {test_acc * 100:.2f}%")
    logging.log(23,f'Report:\n {report}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DiffGCN')
    parser.add_argument('--dataset', type=str, default='BUPT', help='cora, citeseer, citeseer_sg, pubmed, BUPT, BUPT_SG')
    parser.add_argument('--filterbanks', type=str, default='LP', help='LP, HP, BP, LP+HP, LP+BP, HP+BP, ALL')
    parser.add_argument("--feat_drop", type=float, default=0, help="Input dropout probability")
    parser.add_argument("--attr_drop", type=float, default=0, help="attention dropout probability")
    parser.add_argument('--hidden', type=int, default=1024, help='Number of hidden units.')
    parser.add_argument('--mlp_dim', type=int, default=4096, help='MLP output dim.')    
    parser.add_argument('--hidden_channel', type=int, default=8, help='Number of hidden units in ChannelAttention.')
    parser.add_argument('--lr', type=float, default=5e-3, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--early_stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--epochs', type=int, default=4000, help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=2000, help='Patience in early stopping')
    parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training set')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of valing set')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Ratio of testing set')
    parser.add_argument('--seed', type=int, default=42, help="seed for our system")
    parser.add_argument('--print_interval', type=int, default=100, help="the interval of printing in training")
    parser.add_argument('--log_name', type=str, default='DiffGCN', help="Name for logging")
    parser.add_argument('--self_loop', type=bool, default=False, help="add self-loop")
    parser.add_argument('--edgenode', type=bool, default=True, help="edge node not converge other nodes")
    args = parser.parse_args()
       
    main(args)
import time, datetime
import argparse, logging
import numpy as np
from numpy import random
import scipy.sparse as sp
import torch
import dgl
from dgl import load_graphs
from dgl.data.utils import load_info
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from model import MLP, DiffGCN
from utils import EarlyStopping, FocalLoss
from sklearn.metrics import auc, f1_score, classification_report, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

def setup_seed(seed):
    random.seed(seed)
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

def gen_mask(g, train_rate, val_rate):
    labels = g.ndata['label']
    labels = labels.reshape(len(labels))
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

def beijing(sec,what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()
 
def del_edgenode_indegree(graph):
    graph = dgl.remove_self_loop(graph)
    node_degree = np.array(graph.in_degrees())
    node_index_edge = np.argwhere(node_degree == 1)
    node_index_edge = torch.tensor(node_index_edge.reshape(len(node_index_edge)))
    graph.remove_edges(graph.in_edges(node_index_edge,'eid'))
    return dgl.add_self_loop(graph)

def loader(g, node_id, size, device):
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.num_layer)
    dataloader = dgl.dataloading.DataLoader(
            g,
            node_id,
            sampler,
            device=device,
            batch_size=size,
            shuffle=False,
            drop_last=False,
            num_workers=0)
    return dataloader

def inductive_split(g):
    train_id = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_id = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_id = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
    train_dataloader = loader(g, train_id, len(train_id), torch.device('cpu'))
    val_dataloader = loader(g, val_id, len(val_id), torch.device('cpu'))
    test_dataloader = loader(g, test_id, len(test_id), torch.device('cpu'))
    for neighbor_train, _, _ in train_dataloader:
            train_g = g.subgraph(neighbor_train)
    for neighbor_val, _, _ in val_dataloader:
            val_g = g.subgraph(neighbor_val)
    for neighbor_test, _, _ in test_dataloader:
            test_g = g.subgraph(neighbor_test)
    return train_g, val_g, test_g

def evaluate(model, blocks, num_node_id, batch_size, batch_feat, labels, n_classes, loss_fcn, device):
    model.eval()
    pred = torch.zeros(num_node_id, n_classes).to(device)
    with torch.no_grad():
        for i, block in enumerate(blocks):
            batch_pred = model(block, batch_feat[i])
            if i != len(blocks)-1:
                pred[i * batch_size : (i+1) * batch_size, : ] = batch_pred
            else:
                pred[i * batch_size : , : ] = batch_pred
        labels = labels.to(device)
        loss = loss_fcn(pred, labels)
        return accuracy(pred, labels), loss, pred


def main(args):
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
    
    if args.dataset == 'BUPT':
        dataset, _ = load_graphs("/code/DiffGCN/data/BUPT_tele.bin") 
        n_classes = load_info("/code/DiffGCN/data/BUPT_tele.pkl")['num_classes']
        g = gen_mask(dataset[0], args.train_ratio, args.val_ratio)
        label_names=['0','1','2']
    elif args.dataset == 'SC':
        dataset, _ = load_graphs("/code/DiffGCN/data/Sichuan_tele.bin")
        n_classes = load_info("/code/DiffGCN/data/Sichuan_tele.pkl")['num_classes']
        g = gen_mask(dataset[0], args.train_ratio, args.val_ratio)
        label_names=['0','1']
        
    if args.edgenode:
        g = del_edgenode_indegree(g)        
        
    train_g, val_g, test_g = inductive_split(g)
    train_g = train_g.formats(['csc'])
    val_g = val_g.formats(['csc'])
    test_g = test_g.formats(['csc'])
    
    train_nid = torch.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(test_g.ndata['test_mask'], as_tuple=True)[0]
    train_g_nfeat = train_g.ndata.pop('feat')
    val_g_nfeat = val_g.ndata.pop('feat')
    test_g_nfeat = test_g.ndata.pop('feat')
    train_labels = train_g.ndata.pop('label')
    val_labels = val_g.ndata.pop('label')
    test_labels = test_g.ndata.pop('label')
    train_labels = train_labels[train_nid]
    val_labels = val_labels[val_nid]
    test_labels = test_labels[test_nid]
    num_feats = train_g_nfeat.shape[1]

    if args.is_batch:
        train_size = args.train_batch_size
        val_size = args.val_batch_size
        test_size = args.test_batch_size
    else:
        train_size = len(train_nid)
        val_size = len(val_nid)
        test_size = len(test_nid) 
    logging.log(23,f"---------------------dataset: {args.dataset}-------------------------------------------------------------")
    logging.log(23,f"train: {args.train_ratio * 100:.1f}% val: {args.val_ratio * 100:.1f}% hidden: {args.hidden} train_batch_size:{train_size} val_batch_size:{val_size} test_batch_size:{test_size}")
    logging.log(23,f"layers: {args.num_layer} gamma: {args.gamma} seed: {args.seed} lr: {args.lr} weight_decay: {args.weight_decay} epochs: {args.epochs} feat_drop: {args.feat_drop} attr_drop:{args.attn_drop}")
    model = DiffGCN(num_feats, args.hidden, n_classes, args.num_layer, args.feat_drop, args.attn_drop, args.use_diff, device)
    if args.early_stop:
        stopper = EarlyStopping(args.patience)
    model.to(device)
    # loss_fcn = torch.nn.CrossEntropyLoss()
    loss_fcn = FocalLoss(args.gamma) if args.use_focalloss else torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_dataloader_sub = loader(train_g, train_nid, train_size, torch.device('cpu'))
    val_dataloader_sub = loader(val_g, val_nid, val_size, torch.device('cpu'))
    test_dataloader_sub = loader(test_g, test_nid, test_size, torch.device('cpu'))
    train_batch, val_batch, test_batch = [], [], []
    train_batch_nfeats, val_batch_nfeats, test_batch_nfeats = [], [], []
    for src, _, train_blocks_batch in train_dataloader_sub:
        train_batch.append([train_blocks_batch[i].to(device) for i in range(args.num_layer)])
        train_batch_nfeats.append(train_g_nfeat[src].to(device))       
    for src, _, val_blocks_batch in val_dataloader_sub:  
        val_batch.append([val_blocks_batch[i].to(device) for i in range(args.num_layer)])
        val_batch_nfeats.append(val_g_nfeat[src].to(device))

    start_time = time.time()
    last_time = start_time
    for epoch in range(args.epochs):
        model.train()
        train_acc = []
        pred = torch.zeros(len(train_nid), n_classes).to(device)
        for i, train_block in enumerate(train_batch):
            batch_pred = model(train_block, train_batch_nfeats[i])
            if i != len(train_batch)-1:
                pred[i * train_size : (i+1) * train_size, : ] = batch_pred
            else:
                pred[i * train_size : , : ] = batch_pred
        train_acc = accuracy(pred, train_labels.to(device))
        loss = loss_fcn(pred, train_labels.to(device))
        val_acc, val_loss, _ = evaluate(model, val_batch, len(val_nid), val_size, val_batch_nfeats, val_labels, n_classes, loss_fcn, device) 
        optimizer.zero_grad()
        loss.backward()      
        optimizer.step()                                                                 
        if epoch % args.print_interval == 0 and epoch != 0:
            duration = time.time() - last_time
            last_time = time.time()
            logging.info(f"Epoch {epoch}: "
                        f"Train_loss = {loss:.6f},"
                        f"Train_acc = {train_acc * 100:.4f},"
                        f"Val_loss = {val_loss:.6f}, "
                        f"Val_acc = {val_acc * 100:.4f} "
                        f"({duration:.3f} sec)")
        if args.early_stop and epoch != 0:
            if stopper.step(val_acc, model, epoch, val_loss):
                break   

    runtime = time.time() - start_time
    del train_batch, val_batch, train_batch_nfeats, val_batch_nfeats
    torch.cuda.empty_cache()
    if args.early_stop:
        logging.log(21, f"best epoch: {stopper.best_epoch}, best val acc:{stopper.best_score * 100:.4f}, val_loss:{stopper.best_epoch_val_loss:.6f}, ({runtime:.3f} sec)")
    if args.early_stop:
        model.load_state_dict(torch.load('/code/DiffGCN/es_checkpoint.pt')) 
    for src, _, test_blocks_batch in test_dataloader_sub: 
        test_batch.append([test_blocks_batch[i].to(device) for i in range(args.num_layer)])
        test_batch_nfeats.append(test_g_nfeat[src].to(device))
    test_acc, _, test_logits = evaluate(model, test_batch, len(test_nid), test_size, test_batch_nfeats, test_labels, n_classes, loss_fcn, device)
    test_h = torch.argmax(test_logits, dim=1)
    test_f1 = f1_score(test_labels.cpu(), test_h.cpu(), average='macro')
    report = classification_report(test_labels.cpu().detach().numpy(), test_h.cpu().detach().numpy(), target_names=label_names, digits=4)
    if n_classes > 2:
        auc = roc_auc_score(test_labels.cpu(), torch.softmax(test_logits, dim=1).cpu(), average='macro', multi_class='ovo')
    else:
        auc = roc_auc_score(test_labels.cpu(), torch.softmax(test_logits, dim=1)[:,1].cpu())
    logging.log(23,f"Test:  Macro_F1: {test_f1 * 100:.4f}%   Accuracy: {test_acc * 100:.4f}%    Macro_AUC: {auc * 100:.4f}%")
    logging.log(23,f'Report:\n {report}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DiffGCN')
    parser.add_argument('--dataset', type=str, default='SC', help='BUPT, SC')
    parser.add_argument("--feat_drop", type=float, default=0.3, help="Input dropout probability")
    parser.add_argument("--attn_drop", type=float, default=0.7, help="attention dropout probability")
    parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')   
    parser.add_argument('--num_layer', type=int, default=1, help='Number of Conv layers.')   
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=6e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
    parser.add_argument('--patience', type=int, default=500, help='Patience in early stopping')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma in FocalLoss')
    parser.add_argument('--train_ratio', type=float, default=0.2, help='Ratio of training set')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of valing set')
    parser.add_argument('--seed', type=int, default=42, help="seed for our system")
    parser.add_argument('--print_interval', type=int, default=100, help="the interval of printing in training")
    parser.add_argument('--log_name', type=str, default='Ablation-', help="Name for logging")
    parser.add_argument('--edgenode', type=bool, default=True, help="edge node not converge other nodes")
    parser.add_argument('--is_batch', type=bool, default=False, help="batch or not")
    parser.add_argument('--use_diff', type=bool, default=False, help="use feature differences or not")
    parser.add_argument('--use_focalloss', type=bool, default=True, help="use focalloss or not")
    parser.add_argument('--train_batch_size', type=int, default=2048)
    parser.add_argument('--val_batch_size', type=int, default=2048)
    parser.add_argument('--test_batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=4,
                           help="Number of sampling processes. Use 0 for no extra process.")
    parser.add_argument('--early_stop', action='store_true', default=True,
                        help="indicates whether to use early stop or not")
    args = parser.parse_args()
       
    main(args)
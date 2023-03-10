from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = None
        self.best_epoch_val_loss = 0
        
    def step(self, acc, model, epoch, val_loss):
        score = acc
        # score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_epoch_val_loss = val_loss
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.best_epoch_val_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), '/code/DiffGCN/es_checkpoint.pt')

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, size_average = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.elipson = 1e-8

    def forward(self, inputs, labels):
        # alpha = [1, 2, 1]
        # # for i in range(3):
        # #     alpha.append(1 - torch.sum(labels == i) / len(labels))
        # # co = [3 , 1 , 1]
        # # alpha = torch.tensor(co) * torch.tensor(alpha)
        # alpha = torch.tensor(alpha).to(torch.device('cuda'))
        # alpha = torch.gather(alpha, 0, labels)     
        prob = F.softmax(inputs, dim=1) + self.elipson
        prob_c = -F.nll_loss(prob, labels, reduction='none')
        loss = -torch.pow((1-prob_c), self.gamma) * torch.log(prob_c) 
        # loss = -1 * alpha * torch.pow((1-prob_c), self.gamma) * torch.log(prob_c) 
        loss = loss.mean() if self.size_average else loss.sum()
        return loss


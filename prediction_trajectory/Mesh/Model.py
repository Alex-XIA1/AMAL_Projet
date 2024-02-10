import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import torch
import torch.nn as nn
import torch.nn.functional
import torch.utils.data as data
import numpy as np
import sys
import time
import pickle
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
import seaborn as sns
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler

class Model_traj(nn.Module):
    def __init__(self,d1,d2,d3,d4,d5,d6,nb_hop):
        super(Model_traj,self).__init__()
        self.nb_hop = nb_hop
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # useful functions
        L_relu = nn.LeakyReLU()
        sig = nn.Sigmoid()
        relu = nn.ReLU(inplace=False)
        tanh = nn.Tanh()
        softmax = nn.Softmax(dim=0)
        dropout = nn.Dropout(p=0.3)
        
        # Simplices of dimension 1.
        self.g1_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d1, d2), tanh,dropout,
                nn.Linear(d2, d2), tanh,dropout,
                nn.Linear(d2, d2), tanh,dropout,
                nn.Linear(d2, d3), tanh
            ) for _ in range(3)
        ])

        self.D = nn.Sequential(nn.Linear(nb_hop*d5,d5),tanh,dropout,nn.Linear(d5,d5),tanh,dropout, nn.Linear(d5,d5),tanh,dropout, nn.Linear(d5,d6),softmax)
        
        

    def forward(self, x1_0, x1_1, x1_2, B1, Z_):
        xi_ins = [x1_0, x1_1, x1_2]
        xi_in_nodes=[] # in node space

        for i in range(self.nb_hop):
            x=xi_ins[i]
            xi_in = self.g1_layers[i](x) @ B1.T
            xi_in_nodes.append(xi_in)

        if self.nb_hop == 1:
            xi_out = xi_in_nodes[0] * Z_.to(self.device)
            final_out = self.D(xi_out.to(self.device))
        else:
            xi_out = torch.cat([xi_in_node * Z_.to(self.device) for xi_in_node in xi_in_nodes], 1)
            final_out = self.D(xi_out.to(self.device))

        return final_out
        

# Training Function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_samples, total_acc = 0, 0, 0
    B1 = dataloader.dataset.B1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start_time = time.time()
    for x1_0, x1_1, x1_2, z, y in dataloader:
        outputs = model(x1_0.to(device), x1_1.to(device), x1_2.to(device), torch.Tensor(B1).to(device), z.to(device))
        loss = criterion(outputs, y.to(device))
        optimizer.zero_grad()
        outputs = model(x1_0.to(device), x1_1.to(device), x1_2.to(device), torch.Tensor(B1).to(device), z.to(device))
        loss = criterion(outputs.squeeze(), y.long().to(device))
        total_loss += loss.item() * x1_0.size(0)
        total_acc += evaluate(outputs, y) * x1_0.size(0)
        total_samples += x1_0.size(0)
        loss.backward()
        optimizer.step()
    epoch_time = time.time() - start_time
    return total_loss / total_samples, total_acc / total_samples, epoch_time

def validate(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss, total_samples, total_acc = 0, 0, 0
    B1 = dataloader.dataset.B1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for x1_0, x1_1, x1_2, z, y in dataloader:
        outputs = model(x1_0.to(device), x1_1.to(device), x1_2.to(device), torch.Tensor(B1).to(device), z.to(device))
        loss = criterion(outputs, y.to(device))
        outputs = model(x1_0.to(device), x1_1.to(device), x1_2.to(device), torch.Tensor(B1).to(device), z.to(device))
        loss = criterion(outputs.squeeze(), y.long().to(device))
        total_loss += loss.item() * x1_0.size(0)
        total_acc += evaluate(outputs, y) * x1_0.size(0)
        total_samples += x1_0.size(0)
    return total_loss / total_samples, total_acc / total_samples


# Evaluation Function
def evaluate(logits, labels):
    # Move labels to the same device as logits
    labels = labels.to(logits.device)
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).float().mean().item()
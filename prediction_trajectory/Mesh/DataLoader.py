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

class TrajectoryDataset(Dataset):
    def __init__(self, X, Y, Z, B1, B2, mask):
        self.X = X[mask != 0]
        self.Y = Y[mask != 0]
        self.Z = Z[mask != 0]
        #self.X = X
        #self.Y = Y
        #self.Z = Z
        self.B1 = B1
        self.B2 = B2

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x1_0 = np.squeeze(self.X[idx])
        x1_1 = x1_0 @ (self.B2 @ self.B2.T) + x1_0 @ (self.B1.T @ self.B1)
        x1_2 = x1_1 @ (self.B2 @ self.B2.T) + x1_1 @ (self.B1.T @ self.B1)
        y = int(self.Y[idx]-1)
        z = self.Z[idx]
        z=torch.tensor([z], dtype=torch.float)
        #z=z.unsqueeze(1)
        #print(y)
        return torch.tensor(x1_0, dtype=torch.float), torch.tensor(x1_1, dtype=torch.float), torch.tensor(x1_2, dtype=torch.float), z, torch.tensor(y, dtype=torch.long)

        
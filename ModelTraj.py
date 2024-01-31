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
import networkx as nx
from torch.utils.data import Dataset, DataLoader


###################################################### Data Loading and Preprocessing ######################################################
class TrajectoryDataset(Dataset):
    def __init__(self, X, Y, Z, B1, B2, mask):
        self.X = X[mask != 0]
        self.Y = Y[mask != 0]
        self.Z = Z[mask != 0]
        self.B1 = B1 # matrix of incidence of edges and nodes
        self.B2 = B2 # matrix of incidence of edges and edges

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x1_0 = np.squeeze(self.X[idx]) # 0-simplices
        x1_1 = x1_0@(self.B2@self.B2.T) + x1_0@(self.B1.T@self.B1) # 1-simplices
        x1_2 = x1_1@(self.B2@self.B2.T) + x1_1@(self.B1.T@self.B1) # 2-simplices
        y = self.Y[idx]
        z = self.Z[idx]

        class_number = self.binary_list_to_class_number(y)
        
        # Convert y and z to 1-dimensional tensors
        #y_tensor = torch.tensor([class_number]) 
        z_tensor = torch.tensor([z]) if np.isscalar(z) else torch.tensor(z)

        class_number = self.binary_list_to_class_number(y)
    
        # Convert class_number to a 1D tensor with a single value
        y_tensor = torch.tensor(class_number, dtype=torch.long)  # Ensure it's a long tensor

        return torch.Tensor(x1_0), torch.Tensor(x1_1), torch.Tensor(x1_2), z_tensor, y_tensor
    @staticmethod
    def binary_list_to_class_number(binary_list):
        binary_list = binary_list.flatten()
        if np.array_equal(binary_list, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]): return 0
        elif np.array_equal(binary_list, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]): return 1
        elif np.array_equal(binary_list, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]): return 2
        elif np.array_equal(binary_list, [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]): return 3
        elif np.array_equal(binary_list, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]): return 4
        elif np.array_equal(binary_list, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]): return 5
        else: return -1  # Or any other default/error value



###################################################### Model Definition ######################################################
class Model_traj(nn.Module):
    def __init__(self,d1,d2,d3,d4,d5,d6):
        super(Model_traj,self).__init__()

        # useful functions
        L_relu = nn.LeakyReLU()
        sig = nn.Sigmoid()
        relu = nn.ReLU(inplace=False)
        tanh = nn.Tanh()
        softmax = nn.Softmax(dim=0)
        
        # Simplices of dimension 1.
        self.g1_0 = nn.Sequential(nn.Linear(d1,d2),tanh,nn.Linear(d2,d2),tanh, nn.Linear(d2,d2), tanh, nn.Linear(d2,d3),tanh)
        self.g1_1 = nn.Sequential(nn.Linear(d1,d2),tanh,nn.Linear(d2,d2),tanh, nn.Linear(d2,d2), tanh, nn.Linear(d2,d3),tanh)
        self.g1_2 = nn.Sequential(nn.Linear(d1,d2),tanh,nn.Linear(d2,d2),tanh, nn.Linear(d2,d2), tanh, nn.Linear(d2,d3),tanh)

        self.D = nn.Sequential(nn.Linear(2*d5,d5),tanh,nn.Linear(d5,d5),tanh, nn.Linear(d5,d5),tanh, nn.Linear(d5,d6),softmax)
        
        

    def forward(self, x1_0, x1_1, x1_2, B1, Z_,device):

        out1_1 = self.g1_0(x1_0) 
        out1_2 = self.g1_1(x1_1) 
        out1_3 = self.g1_2(x1_2)
        
        #map the embeddings from the vector space of edge embeddings to the vector space of node embeddings
        xi_in0 = out1_1@B1.T 
        xi_in1 = out1_2@B1.T 
        #xi_in2 = out1_3@B1.T
        
        
        
        xi_out0 = xi_in0
        xi_out1 = xi_in1

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        xi_out = torch.cat((xi_out0*Z_.to(device),xi_out1*Z_.to(device)),1)
        final_out = self.D(xi_out.to(device))     				       
        return final_out

###################################################### Training and Evaluation ######################################################
# Training Function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_samples, total_acc = 0, 0, 0
    B1 = dataloader.dataset.B1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    return total_loss / total_samples, total_acc / total_samples

# Evaluation Function
def evaluate(logits, labels):
    # Move labels to the same device as logits
    labels = labels.to(logits.device)
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).float().mean()

# Utility Functions
def plot_loss_acc(list_loss,list_acc):
  plt.figure()
  plt.plot(list_loss)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.savefig('loss.png')
  plt.figure()
  plt.plot(list_acc)
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.savefig('acc.png') 

# Main Function
def main():
    # Data Loading and Preprocessing
    # Replace the following lines with your data loading logic
    datapath = './tp/ocean/'
    X = np.load(datapath + 'flows_in.npy')
    Y = np.load(datapath + 'targets.npy')
    Z_ = np.load(datapath + 'last_nodes.npy')  # Update as per your data processing
    B1 = np.load(datapath + 'B1.npy')
    B2 = np.load(datapath+'B2.npy')
    train_mask = np.load(datapath + 'train_mask.npy')

    # Create Dataset and DataLoader
    dataset = TrajectoryDataset(X, Y, Z_, B1, B2, train_mask)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True)

    # Model Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model_traj(d1=X.shape[1], d2=X.shape[1], d3=X.shape[1], d4=B1.shape[0], d5=B1.shape[0], d6=6).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

    # Training Loop
    for epoch in range(700):
        loss, acc = train(model, dataloader, criterion, optimizer, device)
        
        print(f"Epoch: {epoch}, Loss: {loss}, Accuracy: {acc}")

    # Plotting Loss and Accuracy
    # plot_loss_acc function...

if __name__ == '__main__':
    main()

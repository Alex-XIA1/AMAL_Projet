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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler


class TrajectoryDataset(Dataset):
    def __init__(self, X, Y, Z, B1, B2, mask):
        self.X = X[mask != 0]
        self.Y = Y[mask != 0]
        self.Z = Z[mask != 0]
        self.B1 = B1
        self.B2 = B2

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        x1_0 = np.squeeze(self.X[idx])
        x1_1 = x1_0@(self.B2@self.B2.T) + x1_0@(self.B1.T@self.B1)
        x1_2 = x1_1@(self.B2@self.B2.T) + x1_1@(self.B1.T@self.B1)
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
        #return torch.Tensor(x1_0), torch.Tensor(x1_1), torch.Tensor(x1_2), z_tensor, y_tensor
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
        
        

    def forward(self, x1_0, x1_1, x1_2, B1, Z_):

        out1_1 = self.g1_0(x1_0) 
        out1_2 = self.g1_1(x1_1) 
        out1_3 = self.g1_2(x1_2)
        
        #map the embeddings from the vector space of edge embeddings to the vector space of node embeddings
        xi_in0 = out1_1@B1.T  
        xi_in1 = out1_2@B1.T
        xi_in2 = out1_3@B1.T
        
        #xi_out0 = self.xi0(xi_in0)  # original code, but self.xi0 is not defined
        #xi_out1 = self.xi1(xi_in1)  # original code, but self.xi1 is not defined
        #xi_out2 = self.xi1(xi_in2)  # original code, but self.xi1 is not defined 
        
        
        xi_out0 = xi_in0
        xi_out1 = xi_in1
        #xi_out2 = xi_in2
        #print("xi_out0",xi_out0.shape)
        #print("Z_",Z_.shape)
        #xi_out = torch.cat((xi_out0*Z_.to(device),xi_out1*Z_.to(device),xi_out2*Z_.to(device)),1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        xi_out = torch.cat((xi_out0*Z_.to(device),xi_out1*Z_.to(device)),1)
        final_out = self.D(xi_out.to(device))     				       
        return final_out

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

def validate(model, dataloader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    with torch.no_grad():  # No gradients needed for validation
        for x1_0, x1_1, x1_2, z, y in dataloader:
            x1_0, x1_1, x1_2, z, y = x1_0.to(device), x1_1.to(device), x1_2.to(device), z.to(device), y.to(device)
            outputs = model(x1_0, x1_1, x1_2, torch.Tensor(dataloader.dataset.B1).to(device), z)
            loss = criterion(outputs, y)
            total_loss += loss.item() * x1_0.size(0)
            total_correct += (outputs.argmax(1) == y).sum().item()
    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = total_correct / len(dataloader.dataset)
    return avg_loss, avg_acc


# Evaluation Function
def evaluate(logits, labels):
    # Move labels to the same device as logits
    labels = labels.to(logits.device)
    pred = torch.argmax(logits, dim=1)
    return (pred == labels).float().mean().item()

# Utility Functions
def plot_loss_acc(list_loss, list_acc, plot_type):
    """
    Plots loss and accuracy graphs.

    Parameters:
    - list_loss: NumPy array of loss values.
    - list_acc: NumPy array of accuracy values.
    - plot_type: String indicating the type of plot (e.g., 'Train', 'Validation').
    """
    print("list_loss",list_loss)
    print("list_acc",list_acc)
    print("list_loss",type(list_loss))
    print("list_acc",type(list_acc))
    print("list_loss",list_loss.shape)
    print("list_acc",list_acc.shape)
    # Plotting Loss
    plt.figure(figsize=(8, 6))
    plt.plot(list_loss, label=f'{plot_type} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.ylim(bottom=0)
    plt.title(f'{plot_type} Loss')
    plt.legend()
    plt.savefig(f'{plot_type}_loss.png')
    
    # Plotting Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(list_acc, label=f'{plot_type} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(bottom=0)
    plt.title(f'{plot_type} Accuracy')
    plt.legend()
    plt.savefig(f'{plot_type}_accuracy.png')


# for matrix of confusion
def get_all_preds(model, loader, device):
    all_preds = []
    true_labels = []
    with torch.no_grad():
        for x1_0, x1_1, x1_2, z, y in loader:
            x1_0, x1_1, x1_2, z = x1_0.to(device), x1_1.to(device), x1_2.to(device), z.to(device)
            preds = model(x1_0, x1_1, x1_2, torch.Tensor(loader.dataset.B1).to(device), z)
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            true_labels.extend(y.numpy())
    return np.array(all_preds), np.array(true_labels)



def plot_confusion_matrix(true_labels, pred_labels, classes):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')

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
    dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

    # Model Initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model_traj(d1=X.shape[1], d2=X.shape[1], d3=X.shape[1], d4=B1.shape[0], d5=B1.shape[0], d6=6).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

    num_folds = 5 # 5 folder cross validation
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    # Convert targets to a tensor and get total size
    targets = torch.tensor(dataset.Y)
    dataset_size = len(dataset)
    
    # Track loss and accuracy for plotting
    fold_performance = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(dataset_size))):
        print(f"Starting fold {fold+1}/{num_folds}")
        list_train_loss, list_train_acc = [], []
        list_val_loss, list_val_acc = [], []
        
        # Create subset samplers for training and validation datasets
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(dataset, batch_size=20, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=20, sampler=val_subsampler)
        
        # Initialize your model, criterion, and optimizer here
        model = Model_traj(d1=X.shape[1], d2=X.shape[1], d3=X.shape[1], d4=B1.shape[0], d5=B1.shape[0], d6=6).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
        
        for epoch in range(5):  # Adjust the number of epochs if necessary
            train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print("type(train_loss)", type(train_loss))
            print("type(train_acc)", type(train_acc))
            
            #fold_performance["train_loss"].append(train_loss)
            #fold_performance["train_acc"].append(train_acc)
            #fold_performance["val_loss"].append(val_loss)
            #fold_performance["val_acc"].append(val_acc)
            list_train_loss.append(train_loss)
            list_train_acc.append(train_acc)
            list_val_loss.append(val_loss)
            list_val_acc.append(val_acc)
            
            print(f"Epoch {epoch}: Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}")

        fold_performance["train_loss"].append(list_train_loss)
        fold_performance["train_acc"].append(list_train_acc)
        fold_performance["val_loss"].append(list_val_loss)
        fold_performance["val_acc"].append(list_val_acc)

    # Plotting Loss and Accuracy
    # plot_loss_acc function...
    # Assuming fold_performance is a dictionary that holds lists of metrics for each fold
    # Convert lists to NumPy arrays for each metric
    train_loss_np = np.array(fold_performance['train_loss'])  # Shape: (num_folds, num_epochs)
    train_acc_np = np.array(fold_performance['train_acc'])
    val_loss_np = np.array(fold_performance['val_loss'])
    val_acc_np = np.array(fold_performance['val_acc'])

    # Now, compute the mean across folds (axis 0), preserving the epoch dimension (axis 1)
    avg_train_loss = np.mean(train_loss_np, axis=0)
    avg_train_acc = np.mean(train_acc_np, axis=0)
    avg_val_loss = np.mean(val_loss_np, axis=0)
    avg_val_acc = np.mean(val_acc_np, axis=0)

    plot_loss_acc(avg_train_loss, avg_train_acc, "Train")
    plot_loss_acc(avg_val_loss, avg_val_acc, "Validation")



    # Testing
    test_mask = np.load(datapath + 'test_mask.npy')
    test_dataset = TrajectoryDataset(X, Y, Z_, B1, B2, test_mask)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    # Compute predictions
    pred_labels, true_labels = get_all_preds(model, test_loader, device)

    # Define your classes as a list of strings
    classes = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6']

    # Plot confusion matrix
    plot_confusion_matrix(true_labels, pred_labels, classes)

if __name__ == '__main__':
    main()
    
    

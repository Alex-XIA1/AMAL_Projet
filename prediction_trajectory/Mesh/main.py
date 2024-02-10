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
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, accuracy_score
import seaborn as sns
import networkx as nx
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler



from DataLoader import TrajectoryDataset
from Model import *
from utils import *

# Main Function
def main():
    # Data Loading and Preprocessing
    # Replace the following lines with your data loading logic
    datapath = '../tp/mesh/'
    X = load_variable(datapath + 'edge_attributes_mesh')
    Y = load_variable(datapath + 'targets_mesh')
    Y = np.array(Y).astype(float)
    Z_ = load_variable(datapath + 'last_nodes_mesh')  # Update as per your data processing
    Z_ = np.array(Z_).astype(float)
    B1 = load_variable(datapath + 'B1_mesh')
    B2 = load_variable(datapath+'B2_mesh')

    nb_epoch = 150
    nb_hop = 1 # between 1 and 3
    early_stopping_patience = 50

    print("X",X.shape)
    #print("Y",Y.shape)
    print("Z_",Z_.shape)
    base_factor = 0.1  # dimunue the size of the dataset for testing, set to 1 for the full dataset
    # Create train mask
    train_mask =  np.zeros((X.shape[0],))
    n_ = int(0.8*X.shape[0]*base_factor)
    for i in range(n_): train_mask[random.randint(0,(X.shape[0])-1)]=1

    # create test mask
    test_mask =  np.zeros((X.shape[0],))
    #test_mask[list(np.squeeze(np.where(train_mask==0)))]=1
    for i in range(len(train_mask)):
        if train_mask[i]==0: test_mask[i]=1


    # Create Dataset and DataLoader
    dataset = TrajectoryDataset(X, Y, Z_, B1, B2, train_mask)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Model Initialization
   
    criterion = torch.nn.CrossEntropyLoss()

    nb_folds = 5 # 5 folder cross validation
    kfold = KFold(n_splits=nb_folds, shuffle=True, random_state=42)
    
    # Convert targets to a tensor and get total size
    targets = torch.tensor(dataset.Y)
    dataset_size = len(dataset)
    
    # Track loss and accuracy for plotting
    fold_performance = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "test_acc": []}
    
    for fold, (train_ids, val_ids) in enumerate(kfold.split(range(dataset_size))):
        print(f"Starting fold {fold+1}/{nb_folds}")

        # Reset the early stopping and best val accuracy for each fold
        best_val_acc = 0.0
        epochs_no_improve = 0
        early_stop = False

        np_train_loss=np.zeros(nb_epoch)
        np_train_acc=np.zeros(nb_epoch)
        np_val_loss=np.zeros(nb_epoch)
        np_val_acc=np.zeros(nb_epoch)

        epoch_times = np.zeros(nb_epoch)

        nb_test_acc = np.zeros(nb_folds)
        
        # Create subset samplers for training and validation datasets
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        
        train_loader = DataLoader(dataset, batch_size=8, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=8, sampler=val_subsampler)
        
        # Initialize your model, criterion, and optimizer here
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = Model_traj(d1=X.shape[1], d2=X.shape[1], d3=X.shape[1], d4=B1.shape[0], d5=B1.shape[0], d6=7,nb_hop=nb_hop).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
        
        for epoch in range(nb_epoch):  # Adjust the number of epochs if necessary
            train_loss, train_acc, epoch_time = train(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            
            # Store the loss and accuracy for plotting
            np_train_loss[epoch]=train_loss
            np_train_acc[epoch]=train_acc
            np_val_loss[epoch]=val_loss
            np_val_acc[epoch]=val_acc
            epoch_times[epoch]=epoch_time
            
            print(f"Epoch {epoch}: Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}")

            # Early stopping logic
            if val_acc < best_val_acc:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break


        # print the minimum validation loss 

        print("end of fold")
        print("Minimum validation loss: ", np.min(np_val_loss))

        #print the maximum validation accuracy
        print("Maximum validation accuracy: ", np.max(np_val_acc))

        # print the minimum training loss
        print("Minimum training loss: ", np.min(np_train_loss))

        # print the maximum training accuracy
        print("Maximum training accuracy: ", np.max(np_train_acc))

        # print the average time per epoch
        print("Average time per epoch: ", np.mean(epoch_times))

        #plot_loss_acc(avg_train_loss, avg_train_acc, "Train")
        #plot_loss_acc(avg_val_loss, avg_val_acc, "Validation")

        # test
        # Testing
        print(f"Testing for fold {fold+1}/{nb_folds}")
        test_dataset = TrajectoryDataset(X, Y, Z_, B1, B2, test_mask)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # Compute predictions
        pred_labels, true_labels, all_probs = get_all_preds(model, test_loader, device)

        # Compute accuracy
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f'Accuracy Test: {accuracy * 100:.2f}%')
        nb_test_acc[fold] = accuracy

        # Define your classes as a list of strings
        #classes = ['Class1', 'Class2', 'Class3', 'Class4', 'Class5', 'Class6']

        # Plot confusion matrix
        #plot_confusion_matrix(true_labels, pred_labels, classes)

        # plot the combined precision-recall curve
        #file_name_pr = 'combined_precision_recall_curve.png'
        #plot_combined_precision_recall_curve(true_labels, all_probs, len(classes), file_name_pr)
        
    
    print("End of all folds")
    print(f"Average test accuracy: {np.mean(nb_test_acc) * 100:.2f}%")
    print(f"Standard deviation of test accuracy: {np.std(nb_test_acc) * 100:.2f}%")
    print(f"Minimum test accuracy: {np.min(nb_test_acc) * 100:.2f}%")
    print(f"Maximum test accuracy: {np.max(nb_test_acc) * 100:.2f}%")
    
if __name__ == '__main__':
    main()
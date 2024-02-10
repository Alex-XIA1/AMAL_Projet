import numpy as np
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import torch
import torch.nn as nn
import torch.nn.functional
import torch.utils.data as data
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
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SubsetRandomSampler

# Utility Functions
def plot_loss_acc(list_loss, list_acc, plot_type):
    """
    Plots loss and accuracy graphs.

    Parameters:
    - list_loss: NumPy array of loss values.
    - list_acc: NumPy array of accuracy values.
    - plot_type: String indicating the type of plot (e.g., 'Train', 'Validation').
    """
    
    # Plotting Loss
    plt.figure(figsize=(8, 6))
    plt.plot(list_loss, label=f'{plot_type} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.ylim(bottom=0)
    plt.title(f' Loss')
    plt.legend()
    plt.savefig(f'{plot_type}_loss.png')
    
    # Plotting Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(list_acc, label=f'{plot_type} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(bottom=0)
    plt.title(f'Accuracy')
    plt.legend()
    plt.savefig(f'{plot_type}_accuracy.png')


# for matrix of confusion
def get_all_preds(model, loader, device):
    all_preds = []
    true_labels = []
    all_probs = []  # Store probabilities for each class
    with torch.no_grad():
        for x1_0, x1_1, x1_2, z, y in loader:
            x1_0, x1_1, x1_2, z = x1_0.to(device), x1_1.to(device), x1_2.to(device), z.to(device)

            preds = model(x1_0, x1_1, x1_2, torch.Tensor(loader.dataset.B1).to(device), z)
            # Get the raw output from the model
            outputs = model(x1_0, x1_1, x1_2, torch.Tensor(loader.dataset.B1).to(device), z)
            # Apply softmax to convert to probabilities
            probs = torch.nn.functional.softmax(outputs, dim=1)
            # Store probabilities
            all_probs.extend(probs.cpu().numpy())


            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            true_labels.extend(y.numpy())
    return np.array(all_preds), np.array(true_labels), np.array(all_probs)



def plot_confusion_matrix(true_labels, pred_labels, classes):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')

def plot_combined_precision_recall_curve(true_labels, pred_probs, num_classes, output_file):
    """
    Saves a combined Precision-Recall curve for multiple classes to a file.

    Parameters:
    - true_labels: True labels of the data (1D array).
    - pred_probs: Predicted probabilities for each class (2D array).
    - num_classes: Number of classes.
    - output_file: Filename where the combined plot will be saved.
    """
    plt.figure(figsize=(10, 8))
    
    # Convert true labels to one hot encoding
    true_one_hot = np.eye(num_classes)[true_labels]

    for i in range(num_classes):
        # Compute precision and recall for class i
        precision, recall, _ = precision_recall_curve(true_one_hot[:, i], pred_probs[:, i])
        ap_score = average_precision_score(true_one_hot[:, i], pred_probs[:, i])
        plt.plot(recall, precision, lw=2, label=f'Class {i} (AP={ap_score:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Combined Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    
    # Saving the combined plot
    plt.savefig(output_file)
    plt.close()  # Close the plot to free memory
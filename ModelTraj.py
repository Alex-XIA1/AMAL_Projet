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
def save_variable(variable,filename):
  pickle.dump(variable,open(filename, "wb"))
def load_variable(filename):
  return pickle.load(open(filename,'rb')) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

datapath = './tp/ocean/'
# trajectoire x nombre aretes x valeur (1,0 ou -1)
X = np.load('./tp/ocean/flows_in.npy')
print(X.shape)
# train_mask = np.load(datapath+'train_mask.npy')
# print(train_mask)
# test_mask = np.load(datapath+'test_mask.npy')
# X_tr = X[train_mask!=0]
# X_test = X[test_mask!=0]
# last_nodes = np.load(datapath+'last_nodes.npy')
# target_nodes = np.load(datapath+'target_nodes.npy')
Y = np.load(datapath+'targets.npy')
#y_tr = [list(np.squeeze(Y[np.where(train_mask!=0)])[i]) for i in range(len(Y[np.where(train_mask!=0)]))]
#y_tr_ = len(y_tr)*[1] # 160 for buoy
print(Y.shape)
# #print('y_tr',np.squeeze(y_tr[0]))
# for i in range(len(y_tr)): 
#   if y_tr[i] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]: y_tr_[i] = [0]
#   elif y_tr[i] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]: y_tr_[i] = [1]
#   elif y_tr[i] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]: y_tr_[i] = [2]
#   elif y_tr[i] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]: y_tr_[i] = [3]
#   elif y_tr[i] == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]: y_tr_[i] = [4]
#   elif y_tr[i] == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]: y_tr_[i] = [5]
  
# #print(y_tr_[0])  
# y_tr = torch.squeeze(torch.Tensor(np.array([[int(y_tr_[i][0])] for i in range(len(y_tr_))])))
# y_test = [list(np.squeeze(Y[test_mask!=0])[i]) for i in range(len(Y[test_mask!=0]))]
# y_test_ = len(y_test)*[1] 
# for i in range(len(y_test)): 
#   if y_test[i] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]: y_test_[i] = [0] #[1.0, 0.0, 0.0, 0.0, 0.0, 0.0] for buoy
#   elif y_test[i] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]: y_test_[i] = [1]
#   elif y_test[i] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]: y_test_[i] = [2]
#   elif y_test[i] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]: y_test_[i] = [3]
#   elif y_test[i] == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]: y_test_[i] = [4]
#   elif y_test[i] == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]: y_test_[i] = [5]
  
# y_test = torch.squeeze(torch.Tensor(np.array([[y_test_[i][0]] for i in range(len(y_test_))])))
# B1 = np.load(datapath+'B1.npy')
# B2 = np.load(datapath+'B2.npy')
# G = load_variable(datapath+'G_undir.pkl')
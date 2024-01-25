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
def save_variable(variable,filename):
  pickle.dump(variable,open(filename, "wb"))
def load_variable(filename):
  return pickle.load(open(filename,'rb')) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  

datapath = './tp/ocean/'
# trajectoire x nombre aretes x valeur (1,0 ou -1)
# (200,320,1) means 200 trajectories, 320 edges, value 1 or 0 or -1. 1 
# means the edge is in the trajectory, 0 means the edge is not in the trajectory, -1 means the edge is in the trajectory but in the opposite direction
X = np.load('./tp/ocean/flows_in.npy')
#print(X.shape)                             

### prepare the data, 80% for training, 20% for testing ##
train_mask = np.load(datapath+'train_mask.npy') ## 1 means training, 0 means testing

# print(train_mask)
test_mask = np.load(datapath+'test_mask.npy')
X_tr = X[train_mask!=0]
print("X_tr",X_tr.shape)
X_test = X[test_mask!=0]
# last_nodes = np.load(datapath+'last_nodes.npy')
# target_nodes = np.load(datapath+'target_nodes.npy')
Y = np.load(datapath+'targets.npy')
last_nodes = np.load(datapath+'last_nodes.npy')
y_tr = [list(np.squeeze(Y[np.where(train_mask!=0)])[i]) for i in range(len(Y[np.where(train_mask!=0)]))]
y_tr_ = len(y_tr)*[1] # 6 for ocean 
print('y_tr',len(y_tr))
print("Y",Y.shape) # (200,6,1) means 200 trajectories, 6 classes, 1 or 0
print('y_tr',np.squeeze(y_tr[0]))

# convert binary list to class number for training, 6 classes for ocean
for i in range(len(y_tr)): 
   if y_tr[i] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]: y_tr_[i] = [0]
   elif y_tr[i] == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]: y_tr_[i] = [1]
   elif y_tr[i] == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]: y_tr_[i] = [2]
   elif y_tr[i] == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]: y_tr_[i] = [3]
   elif y_tr[i] == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]: y_tr_[i] = [4]
   elif y_tr[i] == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]: y_tr_[i] = [5]
  
#print('y_tr_',y_tr_)   # convert binary list to class number for training, so 
# #print(y_tr_[0])  
y_tr = torch.squeeze(torch.Tensor(np.array([[int(y_tr_[i][0])] for i in range(len(y_tr_))])))

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

## B1 is the incidence matrix of the graph
B1 = np.load(datapath+'B1.npy') 
B2 = np.load(datapath+'B2.npy')

#### G is the graph with 133 nodes and 320 edges
G = load_variable(datapath+'G_undir.pkl')
#print('G',G)
options = {
    'node_color': 'black',
    'node_size': 20,
    'width': 1,
}
plt.figure(figsize=(10,10))  # Set the size of the plot as needed
nx.draw(G, **options)
plt.savefig("graph_plot.png")  # Save the plot to a file

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

        self.D = nn.Sequential(nn.Linear(3*d5,d5),tanh,nn.Linear(d5,d5),tanh, nn.Linear(d5,d5),tanh, nn.Linear(d5,d6),softmax)
        
        

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
        xi_out2 = xi_in2
        xi_out = torch.cat((xi_out0*Z_.to(device),xi_out1*Z_.to(device),xi_out2*Z_.to(device)),1)
        final_out = self.D(xi_out.to(device))     				       
        return final_out

    


def evaluate(logits,labels):
  pred_train = [torch.argmax(logits[j]).item() for j in range(len(logits))]
  return accuracy_score(labels,pred_train)

def main():
  N0 = (abs(B1@B1.T).shape)[0]
  N1 = (abs(B2@B2.T).shape)[0]
  N2 = (abs(B2.T@B2).shape)[0]
  x1_0 = np.squeeze(X_tr)  # 1 simplexes
  x1_1 = x1_0@(B2@B2.T) + x1_0@(B1.T@B1)  # simplexes Hodge Laplacian
  x1_2 = x1_1@(B2@B2.T) + x1_1@(B1.T@B1)  # simplexes Hodge Laplacian

  #print("N0",N0)
  #print("N1",N1)
  #print("N2",N2)
  #print("x1_0",x1_0.shape)
  #print("x1_1",x1_1.shape)
  #print("x1_2",x1_2.shape)

  
  Z_ = []     
  for l in range(len(last_nodes)):  
      i = last_nodes[l]
      Z__ = np.zeros((B1.shape[0]))
      Z__[[int(j) for j in G.neighbors(i)]]=1
      Z_.append(list(Z__))
  Z_ = np.array(Z_)
  Z_tr_ = Z_[train_mask!=0]
  Z_test = Z_[test_mask!=0]

  #print("Z_",Z_.shape)
  #print("Z_tr_",Z_tr_.shape)
  #print("Z_tr_",Z_tr_[0])

  
  
  indices_all = np.array(list(range(len(y_tr))))
  np.random.seed(1)
  kf = StratifiedKFold(n_splits=5) # 5 folder cross validation
  kf.get_n_splits(indices_all,y_tr)
  
  
  

  model=network = Model_traj(d1=(X_tr.shape)[1],d2=(X_tr.shape)[1],d3=(X_tr.shape)[1],d4=(B1.shape)[0],d5=(B1.shape)[0],d6=6).to(device) #d6 = planar-17/mesh-7/ocean-6/syn-13
 

  
  for train_index, test_index in kf.split(indices_all,y_tr):
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for i in range(0, 15):
      total_loss = 0
      total_samples = 0
      total_acc = 0
      for j in range(0,len(indices_all)//10):  
            optimizer.zero_grad()	
            indices = np.random.choice(train_index,10,replace=False) 
            Z_tr = Z_tr_[indices]
            indices_val = test_index 
            x1_0_tr = x1_0[indices]
            x1_1_tr = x1_1[indices]
            x1_2_tr = x1_2[indices]
            ys = network(torch.Tensor(x1_0_tr).to(device),torch.Tensor(x1_1_tr).to(device),torch.Tensor(x1_2_tr).to(device),torch.Tensor(B1).to(device),torch.Tensor(Z_tr).to(device).type(torch.LongTensor))
            acc_tr = evaluate(ys.cpu(),(y_tr[indices]).type(torch.FloatTensor))
            loss = criterion(torch.squeeze(ys).type(torch.FloatTensor), (y_tr)[indices].type(torch.LongTensor))
            total_loss += loss.item() * len(indices)
            total_acc += acc_tr * len(indices)
            total_samples += len(indices)
            loss.backward()
            optimizer.step()
            
      average_loss = total_loss / total_samples
      acc_tr = total_acc / total_samples
      print ("-----------epoch = %d | training_loss = %f |"%(i,average_loss))
      print ("--------------------- | acc-tr =%f |"%(acc_tr.item()))
      network.eval()

      
      

if __name__ == '__main__':
  main()

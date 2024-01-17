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
import dgl
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import auc
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import itertools
from sklearn.metrics import accuracy_score
import pickle
import json 
from sklearn.preprocessing import MinMaxScaler
import networkx as nx


# Chargement des donnees ATTENTION, les donnees necesitte dlg version 0.9.1 les versions recentes causeront une erreur
path = "./gc/proteins"
training_graphs = np.load(path+'/training_graphs_concat_.npy',allow_pickle=True)
training_labels = np.load(path+'/training_labels_concat_.npy',allow_pickle=True)

val_boundaries = np.load(path+'/val_boundaries_concat_.npy',allow_pickle=True)
val_graphs = np.load(path+'/val_graphs_concat_.npy',allow_pickle=True)
val_labels = np.load(path+'/val_labels_concat_.npy',allow_pickle=True)

testing_boundaries = np.load(path+'/testing_boundaries_concat_.npy',allow_pickle=True)
testing_graphs = np.load(path+'/testing_graphs_concat_.npy',allow_pickle=True)
testing_labels = np.load(path+'/testing_labels_concat_.npy',allow_pickle=True)

# POUR k = 0 et t = 0 la representation est juste un one hot encoding du type d'atome
# x(k)_(t) : k = taille du simplexe (0 = les sommets, 1 = les aretes, 2 = les triangles), t = notion de t-hop de l'article
# Les donnees ci-dessous sont ceux de la partie "precomputing simplicially aware features" fournies par les auteurs d'une phase sans apprentissage.
x0_0_tr = np.load(path+'/x0_0_tr_concat_.npy',allow_pickle=True)
# En theorie la dimension devrait etre : 2*dimfeaturesx0_0tr + dimfeaturesx(-1)_0 + dimfeaturesx(1)_0 -> 2*3 + les autres dims mais la taille est de 6 -> Un vrai probleme theorique ici ?
# x0_1_tr = np.load(path+'/x0_1_tr_concat_.npy',allow_pickle=True)
# x0_2_tr = np.load(path+'/x0_2_tr_concat_.npy',allow_pickle=True)

# Pour le graphe 1 avec 159 sommets, on a 314 aretes et dim 3
x1_0_tr = np.load(path+'/x1_0_tr_concat_.npy',allow_pickle=True)
# dimension theorique de sortie = 3*2 + 3 (dim de x0_0) + 3 (dim de x2_0) -> ici c'est bon on a bien 12 mais donc on comprend pas pour x0_1tr 
# x1_1_tr = np.load(path+'/x1_1_tr_concat_.npy',allow_pickle=True)
# x1_2_tr = np.load(path+'/x1_2_tr_concat_.npy',allow_pickle=True)

# 83 triangles dans le graphe 1 du fold 0!!!
x2_0_tr = np.load(path+'/x2_0_tr_concat_.npy',allow_pickle=True)
# x2_1_tr = np.load(path+'/x2_1_tr_concat_.npy',allow_pickle=True)
# x2_2_tr = np.load(path+'/x2_2_tr_concat_.npy',allow_pickle=True)

# x0_0_val = np.load(path+'/x0_0_val_concat_.npy',allow_pickle=True)
# x0_1_val = np.load(path+'/x0_1_val_concat_.npy',allow_pickle=True)
# x0_2_val = np.load(path+'/x0_2_val_concat_.npy',allow_pickle=True)
# x1_0_val = np.load(path+'/x1_0_val_concat_.npy',allow_pickle=True)
# x1_1_val = np.load(path+'/x1_1_val_concat_.npy',allow_pickle=True)
# x1_2_val = np.load(path+'/x1_2_val_concat_.npy',allow_pickle=True)
# x2_0_val = np.load(path+'/x2_0_val_concat_.npy',allow_pickle=True)
# x2_1_val = np.load(path+'/x2_1_val_concat_.npy',allow_pickle=True)
# x2_2_val = np.load(path+'/x2_2_val_concat_.npy',allow_pickle=True)

# x0_0_test = np.load(path+'/x0_0_test_concat_.npy',allow_pickle=True)
# x0_1_test = np.load(path+'/x0_1_test_concat_.npy',allow_pickle=True)
# x0_2_test = np.load(path+'/x0_2_test_concat_.npy',allow_pickle=True)
# x1_0_test = np.load(path+'/x1_0_test_concat_.npy',allow_pickle=True)
# x1_1_test = np.load(path+'/x1_1_test_concat_.npy',allow_pickle=True)
# x1_2_test = np.load(path+'/x1_2_test_concat_.npy',allow_pickle=True)
# x2_0_test = np.load(path+'/x2_0_test_concat_.npy',allow_pickle=True)

# x2_1_test = np.load(path+'/x2_1_test_concat_.npy',allow_pickle=True)
# x2_2_test = np.load(path+'/x2_2_test_concat_.npy',allow_pickle=True)


class Model(nn.Module):
    # indim : dimension pour les donnees d'entree
    def __init__(self,indim,dimk1t1,dimk2t1,d4,d5,d6,d7,d8,n_c):
        super(Model,self).__init__()

        # Simplex de taille 0 (les sommets) pour t = 0,1,2
        # Critique : les dimensions ne correspondent pas a la formule donnee dans l'article, c'est pas regulier !
        # on peut verifier a partir des fichiers donnees
        self.g0_0 = nn.Sequential(nn.Linear(d1,d2),L_relu,nn.Linear(d2,d3),L_relu,nn.Linear(d3,d3),L_relu,nn.Linear(d3,d3),L_relu)
        # g0_1 doit prendre en entree une dimension superieure a 6 !
        self.g0_1 = nn.Sequential(nn.Linear(6,d2),L_relu,nn.Linear(d2,d3),L_relu,nn.Linear(d3,d3),L_relu,nn.Linear(d3,d3),L_relu)
        # 6 + 6 + ? + 3 = 15 + ? <- k = -1 c'est egal a 3 ? 
        self.g0_2 = nn.Sequential(nn.Linear(18,d2),L_relu,nn.Linear(d2,d3),L_relu,nn.Linear(d3,d3),L_relu,nn.Linear(d3,d3),L_relu)

        # Simplex de taille 1 (les aretes) pour t = 0,1,2
        self.g1_0 = nn.Sequential(nn.Linear(d1,d2),L_relu,nn.Linear(d2,d3),L_relu,nn.Linear(d3,d3),L_relu,nn.Linear(d3,d3),L_relu)
        # 3 + 3 + 3 + 3 = 12, c'est bon ici
        self.g1_1 = nn.Sequential(nn.Linear(12,d2),L_relu,nn.Linear(d2,d3),L_relu,nn.Linear(d3,d3),L_relu,nn.Linear(d3,d3),L_relu)
        # 12 + 12 + 6 + 9 = 39 -> c'est bon
        self.g1_2 = nn.Sequential(nn.Linear(39,d2),L_relu,nn.Linear(d2,d3),L_relu,nn.Linear(d3,d3),L_relu,nn.Linear(d3,d3),L_relu)
        
        # Simplex de taille 2 (les triangles) pour t = 0.1.2
        self.g2_0 = nn.Sequential(nn.Linear(d1,d2),L_relu,nn.Linear(d2,d3),L_relu,nn.Linear(d3,d3),L_relu,nn.Linear(d3,d3),L_relu)
        # 3 + 3 + 3 + ? = 9 + ? <- k = K+1 est egal a 0 ?
        self.g2_1 = nn.Sequential(nn.Linear(9,d2),L_relu,nn.Linear(d2,d3),L_relu,nn.Linear(d3,d3),L_relu,nn.Linear(d3,d3),L_relu)
        # 9 + 9 + 12 + ? = 30 + ? <- 0 aussi ?
        self.g2_2 = nn.Sequential(nn.Linear(30,d2),L_relu,nn.Linear(d2,d3),L_relu,nn.Linear(d3,d3),L_relu,nn.Linear(d3,d3),L_relu)

        #Un mlp tres basique
        self.D = nn.Sequential(nn.Linear(3*3*d3,d8),L_relu,nn.Linear(d8,d8),L_relu,nn.Linear(d8,d8),L_relu,nn.Linear(d8,n_c),sig) #nn.Softmax(dim=0) for multi-class
        self.dropout = nn.Dropout(0.00)
    
    def forward(self, x0_0, x0_1, x0_2, x1_0, x1_1, x1_2, x2_0, x2_1, x2_2):
        # Learning From simplicial aware features
        # gt_k, t =0,1,2 et k = 0,1,2
        out0_1 = self.g0_0(x0_0) 
        out0_2 = self.g0_1(x0_1)
        out0_3 = self.g0_2(x0_2) 
        out1_1 = self.g1_0(x1_0) 
        out1_2 = self.g1_1(x1_1) 
        out1_3 = self.g1_2(x1_2)
        out2_1 = self.g2_0(x2_0) 
        out2_2 = self.g2_1(x2_1) 
        out2_3 = self.g2_2(x2_2)
        
        # On calcul H(k) qui correspond a la concatenation de chaque t-hop pour un k donne
        xi_in0 = torch.cat((torch.sum((out0_1),0),torch.sum((out0_2),0),torch.sum((out0_3),0)),0)
        xi_in1 = torch.cat((torch.sum((out1_1),0),torch.sum((out1_2),0),torch.sum((out1_3),0)),0)
        xi_in2 = torch.cat((torch.sum((out2_1),0),torch.sum((out2_2),0),torch.sum((out2_3),0)),0)

        # La concatenation finale de tous les blocs "attention" avant le passage dans le MLP.
        phi_in = torch.cat(((xi_in0),(xi_in1),(xi_in2)))

        # On passe le tout dans un MLP
        final_out = self.D(phi_in) 
        return final_out





# one hot du type d'atome (quels sont les atomes ?)
#print(x0_0_tr[0][0])
#print(x1_0_tr[0][0])
#print(x2_0_tr[0][0])
options = {
    'node_color': 'black',
    'node_size': 20,
    'width': 1,
}

# Pour afficher un graphe
#print(training_graphs[0][0])
G = dgl.to_networkx(training_graphs[0][0])
#print("Les aretes ", len(G.out_edges))
plt.figure(figsize=[15,7])
nx.draw(G, **options)
# jolie proteine :D
plt.show()
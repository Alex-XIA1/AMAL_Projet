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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import torchmetrics as tm
from tqdm import notebook, tqdm
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, average_precision_score
from datetime import datetime
from datetime import date


# Ceci n'est pas un transformer mais un embedding, c'est rigolo.
class TransformH_k(nn.Module):
    """
    d1 : dim hidden input
    d2 : dim hidden
    dimhop : liste des dimensions des entrees pour t-hop > 0
    activation : fonction d'activation
    maxT : le t-hop maximum
    """
    def __init__(self,d1,d2, dimhop, maxT, activation = nn.LeakyReLU):
        super(TransformH_k,self).__init__()

        self.act = activation()
        self.size = maxT

        # chaque fonction gk_t k une taille de simplexe et t un t-hop
        tmp = [nn.Sequential(nn.Linear(dimhop[i],d1),self.act,nn.Linear(d1,d2),self.act,nn.Linear(d2,d2),self.act,nn.Linear(d2,d2),self.act) for i in range(len(dimhop))]

        self.gk_t = nn.ModuleList(tmp)
    
    def forward(self, data):
        # On verifie que la liste de donnee est bien coherente aux t-hops maximum
        assert self.size == len(data)
        #rint(data[0].shape)
        # on recupere les outputs de tous les t-hops et on somme pour n'avoir qu'une seule ligne
        outputs = [torch.sum(self.gk_t[i](data[i]),dim = 0) for i in range(self.size)]
        torchoutput = torch.cat(outputs, dim = 0)

        # juste pour verifier
        #print("Ok the output is ", torchoutput.size())
        return torchoutput


class GraphModel(nn.Module):
    def __init__(self,dimhop,d2,d3,d4,n_c, maxT, maxK, activation = nn.LeakyReLU, sortie = nn.Sigmoid):
        """
        d2 : dim hidden input
        d3 : dim hidden
        d4 : dim hidden du decodeur
        dimhop : liste de liste des dimensions des entrees pour t-hop > 0
        activation : fonction d'activation
        maxT : le t-hop maximum
        n_c : dimension finale
        sortie : activation de sortie
        """
        super(GraphModel,self).__init__()

        self.act = activation()
        self.maxt = maxT
        self.maxk = maxK

        # La liste des blocs H_k de tranformation de l'article
        self.h_k = nn.ModuleList([TransformH_k(d2,d3, dimhop[i], maxT, activation) for i in range(maxK)])

        self.decoder = nn.Sequential(nn.Linear(maxT*maxK*d3,d4),self.act,nn.Linear(d4,d4),self.act,nn.Linear(d4,d4),self.act,nn.Linear(d4,n_c),sortie())
    
    def forward(self,listeData):
        # vérifier qu'on a bien un nombre de simplexes correspondant
        assert self.maxk == len(listeData)

        outputs = [self.h_k[i](listeData[i]) for i in range(self.maxk)]
        phi = torch.cat(outputs,dim = 0)

        #print("final dim of embedding is ",phi.shape)

        return self.decoder(phi)


# On ne peut pas utiliser de custom dataset 
class CustomDset(Dataset):
    def __init__(self, x0_0, x0_1, x0_2, x1_0, x1_1, x1_2, x2_0, x2_1, x2_2, labels, transform=None, target_transform=None):
        self.x0_0 = x0_0
        self.x0_1 = x0_1
        self.x0_2 = x0_2
        self.x1_0 = x1_0
        self.x1_1 = x1_1
        self.x1_2 = x1_2
        self.x2_0 = x2_0
        self.x2_1 = x2_1
        self.x2_2 = x2_2
        self.labels = labels


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.x0_0[idx], self.x0_1[idx], self.x0_2[idx], self.x1_0[idx], self.x1_1[idx], self.x1_2[idx], self.x2_0[idx], self.x2_1[idx], self.x2_2[idx], self.labels[idx]

# L'auteur de l'article retrouvee ! -> une video de presentation par lui meme : Simple Yet Powerful Graph-aware and Simplicial-aware Neural Models
def testprecomputing(bk1,bk2,bk3,xk1,xk2,xk3):
    """
    bk1 : matrice incidence Nk-2 x Nk-1 -> inutile mais dans l'article
    bk2 : matrice incidence Nk-1 x Nk 
    bk3 : matrice incidence Nk x Nk+1
    xk2 : le vecteur entree (soit un one hot cf. Errica, soit un vecteur de 1 d'apres l'article )
    xk1 et xk3 : idem
    """
    # Les 4 matrices 
    # Dans le cas ou y'a un probleme (a cause de B(-1) et B(K+1)) <- dans le cas d'une erreur theorique

    # dim Nk x Nk
    aku = bk3@bk3.T
    # dim Nk x Dk
    yku = aku@xk2

    # dim Nk x Nk
    akl = bk2.T@bk2
    # dim Nk x Dk
    ykl = akl@xk2

    # dim Nk-1 
    #akb = bk1
    akb = bk2.T
    # dim Nk-2 x Dk-1 <<<<<<<- PROBLEME DE DIMENSION -----> l'auteur a ecrit dans une video que A(k-1),b = Bk.T
    # nouvelle dimension : Nk x Nk-1
    ykb = akb@xk1

    # dim Nk+1 x Nk
    #akc = bk3.T
    akc = bk3
    #print(akc.shape)
    #print(xk3.shape)
    # dim Nk+1 x Dk+1 <- probleme de dimension ENCORE <------ ICI AUSSI ???? ---> A(k+1),c = Bk+1
    # nouvelle dim : Nk x Nk+1
    ykc = akc@xk3

    #print(yku.shape,ykl.shape,ykb.shape,ykc.shape)
    return np.hstack((yku,ykl,ykb,ykc))
    #return np.concatenate((yku,ykl,ykc))

# matrice d'incidence en O((Nk-1 x Nk)**2) : polynomiale
def makeIncidence(edges, threeclique):
    """
    edges : sommets networkx
    threeclique : 3-cliques networkx
    """
    res = np.zeros((len(edges),len(threeclique)))
    edgesl = list(edges)
    threecliquel = list(threeclique)

    for i in range(len(edgesl)):
        count = 0
        tmp1 = edgesl[i][0]
        tmp2 = edgesl[i][1]
        for j in range(len(threecliquel)):
            tmp3 = threeclique[j]
            if tmp1 in tmp3 and tmp2 in tmp3:
                res[i,j] = 1.
                count+=1
                if count == 2 : break
    return res


# Dans leur donnees, les matrices qui n'existent pas sont mis a rien
def onehopprecompute(bk3,xk1,xk3):
    """
    bk2 : matrice incidence Nk-1 x Nk <- ce parametre pose un soucis de dimension
    bk3 : matrice incidence Nk x Nk+1
    xk1 : le vecteur entree (soit un one hot cf. Errica, soit un vecteur de 1 d'apres l'article )
    xk3 : idem
    """
    # Les 4 matrices 
    # Dans le cas ou y'a un probleme (a cause de B(-1) et B(K+1)) <- dans le cas d'une erreur theorique

    # dim Nk x Nk
    #print(bk3.shape)
    aku = bk3@bk3.T
    # dim Nk x Dk
    yku = aku@xk1

    # dim Nk+1 x Nk
    #akc = bk3.T
    akc = bk3
    #print(akc.shape)
    #print(xk3.shape)
    # dim Nk+1 x Dk+1 <- probleme de dimension ENCORE <------ ICI AUSSI ???? ---> A(k+1),c = Bk+1
    # nouvelle dim : Nk x Nk+1
    ykc = akc@xk3

    #print(yku.shape,ykl.shape,ykb.shape,ykc.shape)
    return np.hstack((yku,ykc))
    #return np.concatenate((yku,ykl,ykc))
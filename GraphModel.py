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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import torchmetrics as tm
from tqdm import notebook, tqdm
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
<<<<<<< HEAD
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, average_precision_score
=======
>>>>>>> 38f5a2656653dd545b693fa89c854dc99a38d9a4

"""
Cette version utilise les donnees fournies par les auteurs, les matrices d'incidences sont normalisees ce qui donne de la stabilité. Cependant,
ils ont signale dans l'article que dans certains cas, on perdait l'injectivite (certaines donnees seraient similaires ce qui causeraient des erreurs.)
"""
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

# POUR k = 0 et t = 0 la representation est juste un one hot encoding du type d'atome, pour k = 1 et 2, les matrices sont juste des matrices de 1 (on sait pas pourquoi)
# x(k)_(t) : k = taille du simplexe (0 = les sommets, 1 = les aretes, 2 = les triangles), t = notion de t-hop de l'article
# Les donnees ci-dessous sont ceux de la partie "precomputing simplicially aware features" fournies par les auteurs d'une phase sans apprentissage.
x0_0_tr = np.load(path+'/x0_0_tr_concat_.npy',allow_pickle=True)
# En theorie la dimension devrait etre : 2*dimfeaturesx0_0tr + dimfeaturesx(-1)_0 + dimfeaturesx(1)_0 -> 2*3 + les autres dims mais la taille est de 6 -> Un vrai probleme theorique ici ?
x0_1_tr = np.load(path+'/x0_1_tr_concat_.npy',allow_pickle=True)
x0_2_tr = np.load(path+'/x0_2_tr_concat_.npy',allow_pickle=True)

# Pour le graphe 1 avec 159 sommets, on a 314 aretes et dim 3
x1_0_tr = np.load(path+'/x1_0_tr_concat_.npy',allow_pickle=True)
# dimension theorique de sortie = 3*2 + 3 (dim de x0_0) + 3 (dim de x2_0) -> ici c'est bon on a bien 12 mais donc on comprend pas pour x0_1tr 
x1_1_tr = np.load(path+'/x1_1_tr_concat_.npy',allow_pickle=True)
x1_2_tr = np.load(path+'/x1_2_tr_concat_.npy',allow_pickle=True)

# 83 triangles dans le graphe 1 du fold 0!!!
x2_0_tr = np.load(path+'/x2_0_tr_concat_.npy',allow_pickle=True)
x2_1_tr = np.load(path+'/x2_1_tr_concat_.npy',allow_pickle=True)
x2_2_tr = np.load(path+'/x2_2_tr_concat_.npy',allow_pickle=True)

x0_0_val = np.load(path+'/x0_0_val_concat_.npy',allow_pickle=True)
x0_1_val = np.load(path+'/x0_1_val_concat_.npy',allow_pickle=True)
x0_2_val = np.load(path+'/x0_2_val_concat_.npy',allow_pickle=True)
x1_0_val = np.load(path+'/x1_0_val_concat_.npy',allow_pickle=True)
x1_1_val = np.load(path+'/x1_1_val_concat_.npy',allow_pickle=True)
x1_2_val = np.load(path+'/x1_2_val_concat_.npy',allow_pickle=True)
x2_0_val = np.load(path+'/x2_0_val_concat_.npy',allow_pickle=True)
x2_1_val = np.load(path+'/x2_1_val_concat_.npy',allow_pickle=True)
x2_2_val = np.load(path+'/x2_2_val_concat_.npy',allow_pickle=True)

x0_0_test = np.load(path+'/x0_0_test_concat_.npy',allow_pickle=True)
x0_1_test = np.load(path+'/x0_1_test_concat_.npy',allow_pickle=True)
x0_2_test = np.load(path+'/x0_2_test_concat_.npy',allow_pickle=True)
x1_0_test = np.load(path+'/x1_0_test_concat_.npy',allow_pickle=True)
x1_1_test = np.load(path+'/x1_1_test_concat_.npy',allow_pickle=True)
x1_2_test = np.load(path+'/x1_2_test_concat_.npy',allow_pickle=True)
x2_0_test = np.load(path+'/x2_0_test_concat_.npy',allow_pickle=True)

x2_1_test = np.load(path+'/x2_1_test_concat_.npy',allow_pickle=True)
x2_2_test = np.load(path+'/x2_2_test_concat_.npy',allow_pickle=True)

# Essai de fonction precomputing

# L'auteur de l'article retrouvee ! -> une video de presentation par lui meme : Simple Yet Powerful Graph-aware and Simplicial-aware Neural Models
# 
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

# Le modele utilise pour la tache de classification de graphe
class Model(nn.Module):
    # indim : dimension pour les donnees d'entree
    def __init__(self,d1,d2,d3,d4,n_c, activation = nn.LeakyReLU, sortie = nn.Sigmoid):
        super(Model,self).__init__()

        self.act = activation()

        # BLOC 1 du modele
        # Simplex de taille 0 (les sommets) pour t = 0,1,2
        # Critique : les dimensions ne correspondent pas a la formule donnee dans l'article, c'est pas regulier !
        # on peut verifier a partir des fichiers donnees
        self.g0_0 = nn.Sequential(nn.Linear(d1,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)
        # g0_1 doit prendre en entree une dimension superieure a 6 !
        self.g0_1 = nn.Sequential(nn.Linear(6,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)
        # 6 + 6 + ? + 3 = 15 + ? <- k = -1 c'est egal a 3 ? 
        self.g0_2 = nn.Sequential(nn.Linear(18,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)

        # BLOC 2 du modele
        # Simplex de taille 1 (les aretes) pour t = 0,1,2
        self.g1_0 = nn.Sequential(nn.Linear(d1,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)
        # 3 + 3 + 3 + 3 = 12, c'est bon ici
        self.g1_1 = nn.Sequential(nn.Linear(12,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)
        # 12 + 12 + 6 + 9 = 39 -> c'est bon
        self.g1_2 = nn.Sequential(nn.Linear(39,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)
        
        # BLOC 3 du modele
        # Simplex de taille 2 (les triangles) pour t = 0.1.2
        self.g2_0 = nn.Sequential(nn.Linear(d1,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)
        # 3 + 3 + 3 + ? = 9 + ? <- k = K+1 est egal a 0 ?
        self.g2_1 = nn.Sequential(nn.Linear(9,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)
        # 9 + 9 + 12 + ? = 30 + ? <- 0 aussi ?
        self.g2_2 = nn.Sequential(nn.Linear(30,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)

        #Un mlp tres basique
        self.D = nn.Sequential(nn.Linear(3*3*d3,d4),self.act,nn.Linear(d4,d4),self.act,nn.Linear(d4,d4),self.act,nn.Linear(d4,n_c),sortie()) #nn.Softmax(dim=0) for multi-class
    
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
# options = {
#     'node_color': 'black',
#     'node_size': 20,
#     'width': 1,
# }

# Pour afficher un graphe
#print(training_graphs[0][0])
#G = dgl.to_networkx(training_graphs[0][0])
#undirected = G.to_undirected() # On passe en non oriente car l'article le fait comme ca
#print("Les aretes ", len(undirected.edges))
#print(list(G.nodes(data=True))) # les atomes ne sont pas dans le dataset fourni (on prendra tel quel pour eviter de perdre du temps)
#plt.figure(figsize=[15,7])
#nx.draw(G, **options)
# jolie proteine :D
#plt.show()

# Trouver toutes les 3-cliques (2-simplex) avec networkx
#all_cliques= nx.enumerate_all_cliques(undirected)
#triad_cliques=[x for x in all_cliques if len(x)==3 ]
#print(triad_cliques)

# 2-clique = aretes

#twoclique = undirected.edges()

#oneclique = undirected.nodes()
#print(twoclique)

# Matrice d'incidencee b1 (taille N0 x N1)
#b1 = nx.incidence_matrix(undirected,oneclique,twoclique).todense()
# on essaie de faire le calcul 

# Matrice d'incidence b2 (taille N1 x N2)
#print(x0_0_tr[0][0])
#b2 = makeIncidence(twoclique,triad_cliques)

#ak1 = b2@b2.T
#ykl = ak1@x1_0_tr[0][0]

# On ne retrouve pas les memes valeurs, ils ne disent pas tout dans l'article
# print("nos valeurs pour les 3 premieres colonnes ",ykl[0])
# print("les leurs ", x1_1_tr[0][0][0])

#print(x1_1_tr[0][0][0])

# On test pour k = 1
#newx11 = testprecomputing(None,b1,b2,x0_0_tr[0][0],x1_0_tr[0][0],x2_0_tr[0][0])
# le precomputing ne redonne pas le meme resultat : c'est parce qu'ils utilisent la version normalisees source une reponse de l'auteur a ma question
#print("nos resultats ",newx11)
#print("leurs resultats ", x1_1_tr[0][0])

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



def train_epoch(train_data, labels, model, loss_fn, optim, device = None, num_classes = 2):
    if device == None: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    allLoss = []
    model.train()

    labels = torch.Tensor(labels).type(torch.FloatTensor).to(device)

    # Recuperation des donnees
    x00tr, x01tr, x02tr, x10tr, x11tr, x12tr, x20tr, x21tr, x22tr = train_data 

    batches = torch.randperm(len(labels))
    # on split en batch de 64 puisque dataloader ne marche pas
    splitted = batches.split(64)
    acc = tm.classification.BinaryAccuracy().to(device)

    for e in splitted:
        #print(len(splitted))
        yhat = torch.Tensor([]).to(device)
        # On doit recuperer pour un element du batch les Xk_t
        for b in e:
            x0_0 = x00tr[b]
            x0_1 = x01tr[b]
            x0_2 = x02tr[b]
            x1_0 = x10tr[b]
            x1_1 = x11tr[b]
            x1_2 = x12tr[b]
            x2_0 = x20tr[b]
            x2_1 = x21tr[b]
            x2_2 = x22tr[b]
            optim.zero_grad()
            # Predict de l'element du batch
            yhat = torch.cat((yhat, model(torch.tensor(x0_0).type(torch.FloatTensor).to(device),torch.tensor(x0_1).type(torch.FloatTensor).to(device),
		  	torch.tensor(x0_2).type(torch.FloatTensor).to(device),torch.Tensor(x1_0).type(torch.FloatTensor).to(device),torch.Tensor(x1_1).type(torch.FloatTensor).to(device),
		  	torch.Tensor(x1_2).type(torch.FloatTensor).to(device),torch.Tensor(x2_0).type(torch.FloatTensor).to(device),torch.Tensor(x2_1).type(torch.FloatTensor).to(device),
		  	torch.Tensor(x2_2).type(torch.FloatTensor).to(device))), 0)
        
        yhats = torch.where(yhat > 0.5, 1, 0)
        #print(f'yhat is {yhat.size()} and labels is {labels[e].size()}')
        loss = loss_fn(torch.squeeze(yhat).type(torch.FloatTensor).to(device), labels[e])
        allLoss.append(loss.item())
        acc(yhats, labels[e])
              
        # Optimization
        loss.backward()
        optim.step()
        optim.zero_grad()

    return np.array(allLoss).mean(), acc.compute().item()


def valida_epoch(valid_data, labels, model, loss_fn, device = None, num_classes = 2):
    if device == None: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    allLoss = []

    labels = torch.Tensor(labels).type(torch.FloatTensor).to(device)

    # Recuperation des donnees
    x00tr, x01tr, x02tr, x10tr, x11tr, x12tr, x20tr, x21tr, x22tr = valid_data 

    batches = torch.randperm(len(labels))
    # on split en batch de 64 puisque dataloader ne marche pas
    splitted = batches.split(64)
    acc = tm.classification.BinaryAccuracy().to(device)

    for e in splitted:
        yhat = torch.Tensor([]).to(device)
        # On doit recuperer pour un element du batch les Xk_t
        for b in e:
            x0_0 = x00tr[b]
            x0_1 = x01tr[b]
            x0_2 = x02tr[b]
            x1_0 = x10tr[b]
            x1_1 = x11tr[b]
            x1_2 = x12tr[b]
            x2_0 = x20tr[b]
            x2_1 = x21tr[b]
            x2_2 = x22tr[b]
            # Predict de l'element du batch
            yhat = torch.cat((yhat, model(torch.tensor(x0_0).type(torch.FloatTensor).to(device),torch.tensor(x0_1).type(torch.FloatTensor).to(device),
		  	torch.tensor(x0_2).type(torch.FloatTensor).to(device),torch.Tensor(x1_0).type(torch.FloatTensor).to(device),torch.Tensor(x1_1).type(torch.FloatTensor).to(device),
		  	torch.Tensor(x1_2).type(torch.FloatTensor).to(device),torch.Tensor(x2_0).type(torch.FloatTensor).to(device),torch.Tensor(x2_1).type(torch.FloatTensor).to(device),
		  	torch.Tensor(x2_2).type(torch.FloatTensor).to(device))), 0)
        
        yhats = torch.where(yhat > 0.5, 1, 0)
        #print(f'yhat is {yhat.size()} and labels is {labels[e].size()}')
        loss = loss_fn(torch.squeeze(yhat).type(torch.FloatTensor).to(device), labels[e])
        allLoss.append(loss.item())
        acc(yhats, labels[e])

    return np.array(allLoss).mean(), acc.compute().item()

def test_valide(valid_data, labels, model, loss_fn, device = None, num_classes = 2):
    if device == None: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    # les loss
    allLoss = []
    # le roc_auc
    allroc = []

    labels = torch.Tensor(labels).type(torch.FloatTensor).to(device)

    # Recuperation des donnees
    x00tr, x01tr, x02tr, x10tr, x11tr, x12tr, x20tr, x21tr, x22tr = valid_data 

    batches = torch.randperm(len(labels))
    # on split en batch de 64 puisque dataloader ne marche pas
    splitted = batches.split(64)
    acc = tm.classification.BinaryAccuracy().to(device)

    yhat = torch.Tensor([]).to(device)
    for i in range(len(labels)):
        # On doit recuperer pour un element du batch les Xk_t
        x0_0 = x00tr[i]
        x0_1 = x01tr[i]
        x0_2 = x02tr[i]
        x1_0 = x10tr[i]
        x1_1 = x11tr[i]
        x1_2 = x12tr[i]
        x2_0 = x20tr[i]
        x2_1 = x21tr[i]
        x2_2 = x22tr[i]
        # Predict de l'element du batch
        yhat = torch.cat((yhat, model(torch.tensor(x0_0).type(torch.FloatTensor).to(device),torch.tensor(x0_1).type(torch.FloatTensor).to(device),
		  	torch.tensor(x0_2).type(torch.FloatTensor).to(device),torch.Tensor(x1_0).type(torch.FloatTensor).to(device),torch.Tensor(x1_1).type(torch.FloatTensor).to(device),
		  	torch.Tensor(x1_2).type(torch.FloatTensor).to(device),torch.Tensor(x2_0).type(torch.FloatTensor).to(device),torch.Tensor(x2_1).type(torch.FloatTensor).to(device),
		  	torch.Tensor(x2_2).type(torch.FloatTensor).to(device))), 0)
        
    yhats = torch.where(yhat > 0.5, 1, 0)
    #print(f'yhat is {yhat.size()} and labels is {labels[e].size()}')
    loss = loss_fn(torch.squeeze(yhat).type(torch.FloatTensor).to(device), labels)
    acc(yhats, labels)

    fpr, tpr, _ = roc_curve(torch.Tensor(labels).type(torch.FloatTensor), torch.squeeze(yhat).type(torch.FloatTensor).detach().numpy())
<<<<<<< HEAD
    aucscore = auc(fpr,tpr)

    prec, recall, _ = precision_recall_curve(torch.Tensor(labels).type(torch.FloatTensor), torch.squeeze(yhat).type(torch.FloatTensor).detach().numpy())
    pr_ap = average_precision_score(torch.Tensor(labels).type(torch.FloatTensor), torch.squeeze(yhat).type(torch.FloatTensor).detach().numpy())
=======
>>>>>>> 38f5a2656653dd545b693fa89c854dc99a38d9a4

    cm = confusion_matrix(torch.Tensor(yhats).type(torch.FloatTensor), torch.Tensor(labels).type(torch.FloatTensor))
    #print(cm)

<<<<<<< HEAD
    return loss.item(), acc.compute().item(), fpr, tpr, cm, aucscore, prec, recall, pr_ap
=======
    return loss.item(), acc.compute().item(), fpr, tpr, cm 
>>>>>>> 38f5a2656653dd545b693fa89c854dc99a38d9a4


def run(model, tdata, tlabels, vdata, val_labels, testdata, testlabels, optim, loss_fn = nn.BCELoss(), num_epoch = 100):
    epochtrainloss = []
    epochtrainperfs = []
    epochvalidloss = []
    epochvalidperfs = []
    for epoch in tqdm(np.arange(num_epoch)):
        trainloss, trainacc = train_epoch(tdata, tlabels, model,loss_fn, optim)
        validloss, validacc = valida_epoch(vdata,val_labels,model, loss_fn)
        print(f'\nLoss train {trainloss} and accuracy {trainacc}\n')
        print(f'\nLoss validation {validloss} and accuracy {validacc}\n')
        epochtrainloss.append(trainloss)
        epochvalidloss.append(validloss)
        epochtrainperfs.append(trainacc)
        epochvalidperfs.append(validacc)
    
<<<<<<< HEAD
    testloss, testacc, fpr, tpr, cm, aucscore, precision, recall, pr_ap = test_valide(testdata, testlabels, model, loss_fn)

    print(f'The final loss for test is {testloss} its accuracy is {testacc}')
    # toutes les performances et loss pour train et validation
=======
    testloss, testacc, fpr, tpr, cm = test_valide(testdata, testlabels, model, loss_fn)

    print(f'The final loss for test is {testloss} its accuracy is {testacc}')

>>>>>>> 38f5a2656653dd545b693fa89c854dc99a38d9a4
    plt.subplot(2,2,1)
    plt.plot(np.array(epochtrainloss))
    plt.title("loss train")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    
    plt.subplot(2,2,2)
    plt.plot(np.array(epochtrainperfs))
    plt.title("performances train")
    plt.xlabel("epoch")
    plt.ylabel("performances")
    
    plt.subplot(2,2,3)
    plt.plot(np.array(epochvalidloss))
    plt.title("loss validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(2,2,4)
    plt.plot(np.array(epochvalidperfs))
    plt.title("performances validation")
    plt.xlabel("epoch")
    plt.ylabel("performances")

    plt.tight_layout()
    plt.show()

<<<<<<< HEAD
    # ROC AUC image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=f'ROC (AUC = {np.round(aucscore,2)})')
    # Precision Recall image
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall, estimator_name=f'PR (AP = {np.round(pr_ap,2)})')
    roc_display.plot(ax=ax1)
    pr_display.plot(ax=ax2)
    ax1.set_title("Courbe ROC sur données test")
    ax2.set_title("Courbe PR sur données test")
    plt.show()

    # matrice de confusion test
=======
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.title("courbe roc pour les tests")
    plt.show()

>>>>>>> 38f5a2656653dd545b693fa89c854dc99a38d9a4
    disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title("Matrice de confusion test")
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.001
model = Model(d1=3,d2=2*32,d3=2*32,d4=2*32,n_c=1).to(device)
optim = torch.optim.Adam(list(model.parameters()),lr = lr)
optim.zero_grad()

"""
On concatene toutes les donnees, ils avaient prevu une cross validation 10-folds 
mais, ils ont deja split le dataset en 3 datasets (train, validation, test) donc on n'en a probablement pas besoin
"""

limit = 3
allx00tr = np.array([])
for i in range(limit):
    allx00tr = np.concatenate((allx00tr,x0_0_tr[i]))

allx01tr = np.array([])
for i in range(limit):
    allx01tr = np.concatenate((allx01tr,x0_1_tr[i]))

allx02tr = np.array([])
for i in range(limit):
    allx02tr = np.concatenate((allx02tr,x0_2_tr[i]))

allx10tr = np.array([])
for i in range(limit):
    allx10tr = np.concatenate((allx10tr,x1_0_tr[i]))

allx11tr = np.array([])
for i in range(limit):
    allx11tr = np.concatenate((allx11tr,x1_1_tr[i]))

allx12tr = np.array([])
for i in range(limit):
    allx12tr = np.concatenate((allx12tr,x1_2_tr[i]))

allx20tr = np.array([])
for i in range(limit):
    allx20tr = np.concatenate((allx20tr,x2_0_tr[i]))

allx21tr = np.array([])
for i in range(limit):
    allx21tr = np.concatenate((allx21tr,x2_1_tr[i]))


alllabstr = np.array([])
allx22tr = np.array([])
for i in range(limit):
    allx22tr = np.concatenate((allx22tr,x2_2_tr[i]))
    alllabstr = np.hstack((alllabstr,training_labels[i]))

# validations
allx00val = np.array([])
for i in range(limit):
    allx00val = np.concatenate((allx00val,x0_0_val[i]))

allx01val = np.array([])
for i in range(limit):
    allx01val = np.concatenate((allx01val,x0_1_val[i]))

allx02val = np.array([])
for i in range(limit):
    allx02val = np.concatenate((allx02val,x0_2_val[i]))

allx10val = np.array([])
for i in range(limit):
    allx10val = np.concatenate((allx10val,x1_0_val[i]))

allx11val = np.array([])
for i in range(limit):
    allx11val = np.concatenate((allx11val,x1_1_val[i]))

allx12val = np.array([])
for i in range(limit):
    allx12val = np.concatenate((allx12val,x1_2_val[i]))

allx20val = np.array([])
for i in range(limit):
    allx20val = np.concatenate((allx20val,x2_0_val[i]))

allx21val = np.array([])
for i in range(limit):
    allx21val = np.concatenate((allx21val,x2_1_val[i]))

alllabsval = np.array([])
allx22val = np.array([])
for i in range(limit):
    allx22val = np.concatenate((allx22val,x2_2_val[i]))
    alllabsval = np.hstack((alllabsval,val_labels[i]))

# tests
allx00test = np.array([])
for i in range(limit):
    allx00test = np.concatenate((allx00test,x0_0_test[i]))

allx01test = np.array([])
for i in range(limit):
    allx01test = np.concatenate((allx01test,x0_1_test[i]))

allx02test = np.array([])
for i in range(limit):
    allx02test = np.concatenate((allx02test,x0_2_test[i]))

allx10test = np.array([])
for i in range(limit):
    allx10test = np.concatenate((allx10test,x1_0_test[i]))

allx11test = np.array([])
for i in range(limit):
    allx11test = np.concatenate((allx11test,x1_1_test[i]))

allx12test = np.array([])
for i in range(limit):
    allx12test = np.concatenate((allx12test,x1_2_test[i]))

allx20test = np.array([])
for i in range(limit):
    allx20test = np.concatenate((allx20test,x2_0_test[i]))

allx21test = np.array([])
for i in range(limit):
    allx21test = np.concatenate((allx21test,x2_1_test[i]))

alllabstest = np.array([])
allx22test = np.array([])
for i in range(limit):
    allx22test = np.concatenate((allx22test,x2_2_test[i]))
<<<<<<< HEAD
    # fold x 1 x datasize
    alllabstest = np.hstack((alllabstest,testing_labels[i][0]))

=======
    alllabstest = np.hstack((alllabstest,testing_labels[0][i]))
>>>>>>> 38f5a2656653dd545b693fa89c854dc99a38d9a4

# print(alllabstr.shape)
# print(alllabstest.shape)

# Nk x Nk
# N x Nk x Dk
# nik, ij -> nik 
# concatener
#indata = (x0_0_tr[0], x0_1_tr[0], x0_2_tr[0], x1_0_tr[0], x1_1_tr[0], x1_2_tr[0], x2_0_tr[0], x2_1_tr[0], x2_2_tr[0])
indata = (allx00tr, allx01tr, allx02tr, allx10tr, allx11tr, allx12tr, allx20tr, allx21tr, allx22tr)
#valdata = (x0_0_val[0], x0_1_val[0], x0_2_val[0], x1_0_val[0], x1_1_val[0], x1_2_val[0], x2_0_val[0], x2_1_val[0], x2_2_val[0])
valdata = (allx00val, allx01val, allx02val, allx10val, allx11val, allx12val, allx20val, allx21val, allx22val)
testdata = (allx00test, allx01test, allx02test, allx10test, allx11test, allx12test, allx20test, allx21test, allx22test)
run(model, indata ,alllabstr, valdata, alllabsval, testdata, alllabstest, optim)
#print(len(training_labels[0]))

# sur le fold 0

# ATTENTION on ne peut pas faire de dataloader car les matrices d'incidences sont de dimensions differentes en plus padder les simplexes n'aurait pas de sens.
# traindata = CustomDset(x0_0_tr[0], x0_1_tr[0], x0_2_tr[0], x1_0_tr[0], x1_1_tr[0], x1_2_tr[0], x2_0_tr[0], x2_1_tr[0], x2_2_tr[0], training_labels[0])
# train_dataloader = DataLoader(traindata, batch_size=64, shuffle=True)

# for x0_0, x0_1, x0_2, x1_0, x1_1, x1_2, x2_0, x2_1, x2_2, y in train_dataloader:
#     #args, y = args[:,:-1], args[:,-1]
#     out = model(x0_0, x0_1, x0_2, x1_0, x1_1, x1_2, x2_0, x2_1, x2_2)
#     break



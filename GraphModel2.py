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
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, average_precision_score

"""

Cette version est une adapation des donnees de l'auteur avec utilisation directe de l'exemple du papier au lieu de passer par des matrices normalisees.
+ Simplification avec seulement k = 0 et t = 1 car il serait potentiellement cher de calculer les matrices d'incidences pour chacun des graphes et egalement
les matrices Ak pour chaque k-simplex

Il faut trouver pour chaque graphe, les (k+1)-cliques, faire les matrices d'incidence pour chaque niveau de k-simplex et faire
tous les calculs matriciels pour chaque saut (hop) vers les informations. Ce qui peut etre tres long.

On a donc fait cela pour bien verifier que l'exemple donne par les auteurs fonctionne egalement sans normalisation.

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
# x0_1_tr = np.load(path+'/x0_1_tr_concat_.npy',allow_pickle=True)
# x0_2_tr = np.load(path+'/x0_2_tr_concat_.npy',allow_pickle=True)

# Pour le graphe 1 avec 159 sommets, on a 314 aretes et dim 3
x1_0_tr = np.load(path+'/x1_0_tr_concat_.npy',allow_pickle=True)
# dimension theorique de sortie = 3*2 + 3 (dim de x0_0) + 3 (dim de x2_0) -> ici c'est bon on a bien 12 mais donc on comprend pas pour x0_1tr 
# x1_1_tr = np.load(path+'/x1_1_tr_concat_.npy',allow_pickle=True)
# x1_2_tr = np.load(path+'/x1_2_tr_concat_.npy',allow_pickle=True)

# 83 triangles dans le graphe 1 du fold 0!!!
#x2_0_tr = np.load(path+'/x2_0_tr_concat_.npy',allow_pickle=True)
# x2_1_tr = np.load(path+'/x2_1_tr_concat_.npy',allow_pickle=True)
# x2_2_tr = np.load(path+'/x2_2_tr_concat_.npy',allow_pickle=True)

x0_0_val = np.load(path+'/x0_0_val_concat_.npy',allow_pickle=True)
# x0_1_val = np.load(path+'/x0_1_val_concat_.npy',allow_pickle=True)
# x0_2_val = np.load(path+'/x0_2_val_concat_.npy',allow_pickle=True)
x1_0_val = np.load(path+'/x1_0_val_concat_.npy',allow_pickle=True)
# x1_1_val = np.load(path+'/x1_1_val_concat_.npy',allow_pickle=True)
# x1_2_val = np.load(path+'/x1_2_val_concat_.npy',allow_pickle=True)
# x2_0_val = np.load(path+'/x2_0_val_concat_.npy',allow_pickle=True)
# x2_1_val = np.load(path+'/x2_1_val_concat_.npy',allow_pickle=True)
# x2_2_val = np.load(path+'/x2_2_val_concat_.npy',allow_pickle=True)

x0_0_test = np.load(path+'/x0_0_test_concat_.npy',allow_pickle=True)
# x0_1_test = np.load(path+'/x0_1_test_concat_.npy',allow_pickle=True)
# x0_2_test = np.load(path+'/x0_2_test_concat_.npy',allow_pickle=True)
x1_0_test = np.load(path+'/x1_0_test_concat_.npy',allow_pickle=True)
# x1_1_test = np.load(path+'/x1_1_test_concat_.npy',allow_pickle=True)
# x1_2_test = np.load(path+'/x1_2_test_concat_.npy',allow_pickle=True)
# x2_0_test = np.load(path+'/x2_0_test_concat_.npy',allow_pickle=True)

# x2_1_test = np.load(path+'/x2_1_test_concat_.npy',allow_pickle=True)
# x2_2_test = np.load(path+'/x2_2_test_concat_.npy',allow_pickle=True)

# Essai de fonction precomputing

# L'auteur de l'article retrouvee ! -> une video de presentation par lui meme : Simple Yet Powerful Graph-aware and Simplicial-aware Neural Models
def testprecomputing(bk1,bk2,bk3,xk1,xk2,xk3):
    """
    bk1 : matrice incidence Nk-2 x Nk-1
    bk2 : matrice incidence Nk-1 x Nk <- ce parametre pose un soucis de dimension
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

# Le modele utilise pour la tache de classification de graphe (adapte seulement pour les 0-simplex pour simplifier les calculs)
class Model(nn.Module):
    # indim : dimension pour les donnees d'entree
    def __init__(self,d1,d2,d3,d4,d5,d6,d7,d8,n_c, activation = nn.LeakyReLU, sortie = nn.Sigmoid):
        super(Model,self).__init__()

        self.act = activation()

        # BLOC 1 du modele
        # Simplex de taille 0 (les sommets) pour t = 0,1,2
        # Critique : les dimensions ne correspondent pas a la formule donnee dans l'article, c'est pas regulier !
        # on peut verifier a partir des fichiers donnees
        self.g0_0 = nn.Sequential(nn.Linear(d1,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)
        # g0_1 doit prendre en entree une dimension superieure a 6 !
        self.g0_1 = nn.Sequential(nn.Linear(6,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)


        #Un mlp tres basique
        self.D = nn.Sequential(nn.Linear(2*d3,d8),self.act,nn.Linear(d8,d8),self.act,nn.Linear(d8,d8),self.act,nn.Linear(d8,n_c),sortie()) #nn.Softmax(dim=0) for multi-class
    
    def forward(self, x0_0, x0_1):
        # Learning From simplicial aware features
        # gt_k, t =0,1,2 et k = 0,1,2
        out0_1 = self.g0_0(x0_0) 
        out0_2 = self.g0_1(x0_1)

        
        # On calcul H(k) qui correspond a la concatenation de chaque t-hop pour un k donne
        xi_in0 = torch.cat((torch.sum((out0_1),0),torch.sum((out0_2),0)),0)

        phi_in = xi_in0

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
# G = dgl.to_networkx(training_graphs[0][0])
# undirected = G.to_undirected() # On passe en non oriente car l'article le fait comme ca
#print("Les aretes ", len(undirected.edges))
#print(list(G.nodes(data=True))) # les atomes ne sont pas dans le dataset fourni (on prendra tel quel pour eviter de perdre du temps)
#plt.figure(figsize=[15,7])
#nx.draw(G, **options)
# jolie proteine :D
#plt.show()

# Trouver toutes les 3-cliques (2-simplex) avec networkx
# all_cliques= nx.enumerate_all_cliques(undirected)
# triad_cliques=[x for x in all_cliques if len(x)==3 ]
#print(triad_cliques)

# 2-clique = aretes

# twoclique = undirected.edges()

# oneclique = undirected.nodes()
#print(twoclique)

# Matrice d'incidencee b1 (taille N0 x N1)
# b1 = nx.incidence_matrix(undirected,oneclique,twoclique).todense()
# on essaie de faire le calcul 

# Matrice d'incidence b2 (taille N1 x N2)
#print(x0_0_tr[0][0])
# b2 = makeIncidence(twoclique,triad_cliques)

# ak1 = b2@b2.T
#ykl = ak1@x1_0_tr[0][0]

# On ne retrouve pas les memes valeurs, ils ne disent pas tout dans l'article
# print("nos valeurs pour les 3 premieres colonnes ",ykl[0])
# print("les leurs ", x1_1_tr[0][0][0])

#print(x1_1_tr[0][0][0])

# On test pour k = 1
#newx11 = testprecomputing(None,b1,b2,x0_0_tr[0][0],x1_0_tr[0][0],x2_0_tr[0][0])
# le precomputing ne redonne pas le meme resultat
# print("nos resultats ",newx11)
# print("leurs resultats ", x1_1_tr[0][0])


def train_epoch(train_data, labels, model, loss_fn, optim, device = None, num_classes = 2):
    if device == None: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    allLoss = []
    model.train()

    x00tr, x01tr = train_data

    labels = torch.Tensor(labels).type(torch.FloatTensor).to(device)

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
            optim.zero_grad()
            # Predict de l'element du batch
            yhat = torch.cat((yhat, model(torch.tensor(x0_0).type(torch.FloatTensor).to(device), torch.tensor(x0_1).type(torch.FloatTensor).to(device))), 0)
        
        yhats = torch.where(yhat > 0.5, 1, 0)
        #print(f'yhat is {yhat.size()} and labels is {labels[e].size()}')
        loss = loss_fn(torch.squeeze(yhat).type(torch.FloatTensor).to(device), labels[e])
        allLoss.append(loss.item())
        #allroc.append(roc_auc_score)
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
    # les loss
    allLoss = []
    # le roc_auc
    #allroc = []

    labels = torch.Tensor(labels).type(torch.FloatTensor).to(device)

    # Recuperation des donnees
    x00tr, x01tr = valid_data 

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
            # Predict de l'element du batch
            yhat = torch.cat((yhat, model(torch.tensor(x0_0).type(torch.FloatTensor).to(device),torch.tensor(x0_1).type(torch.FloatTensor).to(device))), 0)
        
        yhats = torch.where(yhat > 0.5, 1, 0)
        #roc_auc_score = roc_auc_score(torch.Tensor(labels[e]).type(torch.FloatTensor),torch.squeeze(yhat).type(torch.FloatTensor).detach().numpy())
        #print(f'yhat is {yhat.size()} and labels is {labels[e].size()}')
        loss = loss_fn(torch.squeeze(yhat).type(torch.FloatTensor).to(device), labels[e])
        allLoss.append(loss.item())
        #allroc.append(roc_auc_score)
        acc(yhats, labels[e])

    return np.array(allLoss).mean(), acc.compute().item()# np.array(allroc).mean()

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
    x00tr, x01tr = valid_data 

    batches = torch.randperm(len(labels))
    # on split en batch de 64 puisque dataloader ne marche pas
    splitted = batches.split(64)
    acc = tm.classification.BinaryAccuracy().to(device)

    yhat = torch.Tensor([]).to(device)
    for i in range(len(labels)):
        # On doit recuperer pour un element du batch les Xk_t
        x0_0 = x00tr[i]
        x0_1 = x01tr[i]
        # Predict de l'element du batch
        yhat = torch.cat((yhat, model(torch.tensor(x0_0).type(torch.FloatTensor).to(device),torch.tensor(x0_1).type(torch.FloatTensor).to(device))), 0)
        
    yhats = torch.where(yhat > 0.5, 1, 0)
    #print(f'yhat is {yhat.size()} and labels is {labels[e].size()}')
    loss = loss_fn(torch.squeeze(yhat).type(torch.FloatTensor).to(device), labels)
    acc(yhats, labels)

    fpr, tpr, _ = roc_curve(torch.Tensor(labels).type(torch.FloatTensor), torch.squeeze(yhat).type(torch.FloatTensor).detach().numpy())
    aucscore = auc(fpr,tpr)

    prec, recall, _ = precision_recall_curve(torch.Tensor(labels).type(torch.FloatTensor), torch.squeeze(yhat).type(torch.FloatTensor).detach().numpy())
    pr_ap = average_precision_score(torch.Tensor(labels).type(torch.FloatTensor), torch.squeeze(yhat).type(torch.FloatTensor).detach().numpy())

    cm = confusion_matrix(torch.Tensor(yhats).type(torch.FloatTensor), torch.Tensor(labels).type(torch.FloatTensor))

    return loss.item(), acc.compute().item(), fpr, tpr, cm, aucscore, prec, recall, pr_ap

def run(model, tdata, tlabels, vdata, val_labels, testdata, testlabels , optim, loss_fn = nn.BCELoss(), num_epoch = 200):
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
    
    testloss, testacc, fpr, tpr, cm, aucscore, precision, recall, pr_ap  = test_valide(testdata, testlabels, model, loss_fn)

    print(f'The final loss for test is {testloss} its accuracy is {testacc}')

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

    disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title("Matrice de confusion test")
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.001
model = Model(d1=3,d2=2*32,d3=2*32,d4=2*32,d5=2*32,d6=2*32,d7=2*32,d8=2*32,n_c=1).to(device)
optim = torch.optim.Adam(list(model.parameters()),lr = lr)
optim.zero_grad()

indata = x0_0_tr[0]
valdata = x0_0_val[0]
testdata = x0_0_test[0]

indatahop1 = []
valdatahhop1 = []
testdatahop1 = []
#precomputing
for i in range(len(indata)):
    g = dgl.to_networkx(training_graphs[0][i])
    # passage au non dirige
    undirected = g.to_undirected()
    twoclique = undirected.edges()
    oneclique = undirected.nodes()
    b1 = nx.incidence_matrix(undirected,oneclique,twoclique).todense()
    # all_cliques= nx.enumerate_all_cliques(undirected)
    # triad_cliques=[x for x in all_cliques if len(x)==3 ]
    # b2 = makeIncidence(twoclique,triad_cliques)
    indatahop1.append(onehopprecompute(b1,indata[i],x1_0_tr[0][i]))

for i in range(len(valdata)):
    g = dgl.to_networkx(val_graphs[0][i])
    # passage au non dirige
    undirected = g.to_undirected()
    twoclique = undirected.edges()
    oneclique = undirected.nodes()
    b1 = nx.incidence_matrix(undirected,oneclique,twoclique).todense()
    # all_cliques= nx.enumerate_all_cliques(undirected)
    # triad_cliques=[x for x in all_cliques if len(x)==3 ]
    # b2 = makeIncidence(twoclique,triad_cliques)
    valdatahhop1.append(onehopprecompute(b1,valdata[i],x1_0_val[0][i]))

for i in range(len(testdata)):
    g = dgl.to_networkx(testing_graphs[0][0][i])
    # passage au non dirige
    undirected = g.to_undirected()
    twoclique = undirected.edges()
    oneclique = undirected.nodes()
    b1 = nx.incidence_matrix(undirected,oneclique,twoclique).todense()
    # all_cliques= nx.enumerate_all_cliques(undirected)
    # triad_cliques=[x for x in all_cliques if len(x)==3 ]
    # b2 = makeIncidence(twoclique,triad_cliques)
    testdatahop1.append(onehopprecompute(b1,testdata[i],x1_0_test[0][i]))

indatahop1 = np.array(indatahop1,dtype=object)
trainentry = (indata, indatahop1)

valdatahhop1 = np.array(valdatahhop1,dtype=object)
valentry = (valdata, valdatahhop1)

testdatahop1 = np.array(testdatahop1,dtype=object)
testentry = (testdata, testdatahop1)


run(model, trainentry ,training_labels[0], valentry, val_labels[0], testentry, testing_labels[0][0], optim)

# The final loss for test is 0.4913697838783264 its accuracy is 0.7477477192878723
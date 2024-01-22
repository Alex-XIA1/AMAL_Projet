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

# Essai de fonction precomputing

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
    # dim Nk-2 x Dk-1 <<<<<<<- PROBLEME DE DIMENSION : CRITIQUE A METTRE DANS LE POSTER
    #ykb = akb@xk1

    # dim Nk+1 x Nk
    akc = bk3.T
    #akc = bk3
    #print(akc.shape)
    #print(xk3.shape)
    # dim Nk+1 x Dk+1 <- probleme de dimension ENCORE <------ ICI AUSSI CEST UNE CRITIQUE
    ykc = akc@xk3

    #return np.concatenate((yku,ykl,ykb,ykc))
    return np.concatenate((yku,ykl,ykc))


# matrice d'incidence en O((Nk-1 x Nk)**2)
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
    def __init__(self,d1,d2,d3,d4,d5,d6,d7,d8,n_c, activation = nn.LeakyReLU, sortie = nn.Sigmoid):
        super(Model,self).__init__()

        self.act = activation()

        # Simplex de taille 0 (les sommets) pour t = 0,1,2
        # Critique : les dimensions ne correspondent pas a la formule donnee dans l'article, c'est pas regulier !
        # on peut verifier a partir des fichiers donnees
        self.g0_0 = nn.Sequential(nn.Linear(d1,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)
        # g0_1 doit prendre en entree une dimension superieure a 6 !
        self.g0_1 = nn.Sequential(nn.Linear(6,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)
        # 6 + 6 + ? + 3 = 15 + ? <- k = -1 c'est egal a 3 ? 
        self.g0_2 = nn.Sequential(nn.Linear(18,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)

        # Simplex de taille 1 (les aretes) pour t = 0,1,2
        self.g1_0 = nn.Sequential(nn.Linear(d1,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)
        # 3 + 3 + 3 + 3 = 12, c'est bon ici
        self.g1_1 = nn.Sequential(nn.Linear(12,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)
        # 12 + 12 + 6 + 9 = 39 -> c'est bon
        self.g1_2 = nn.Sequential(nn.Linear(39,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)
        
        # Simplex de taille 2 (les triangles) pour t = 0.1.2
        self.g2_0 = nn.Sequential(nn.Linear(d1,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)
        # 3 + 3 + 3 + ? = 9 + ? <- k = K+1 est egal a 0 ?
        self.g2_1 = nn.Sequential(nn.Linear(9,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)
        # 9 + 9 + 12 + ? = 30 + ? <- 0 aussi ?
        self.g2_2 = nn.Sequential(nn.Linear(30,d2),self.act,nn.Linear(d2,d3),self.act,nn.Linear(d3,d3),self.act,nn.Linear(d3,d3),self.act)

        #Un mlp tres basique
        self.D = nn.Sequential(nn.Linear(3*3*d3,d8),self.act,nn.Linear(d8,d8),self.act,nn.Linear(d8,d8),self.act,nn.Linear(d8,n_c),sortie()) #nn.Softmax(dim=0) for multi-class
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
undirected = G.to_undirected() # On passe en non oriente car l'article le fait comme ca
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
#b1 = nx.incidence_matrix(undirected,oneclique,twoclique).todense()

# Matrice d'incidence b2 (taille N1 x N2)
#b2 = makeIncidence(twoclique,triad_cliques)

# On test pour k = 1
# test = testprecomputing(None,b1,b2,x0_0_tr[0][0],x1_0_tr[0][0],x2_0_tr[0][0])


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

    labels = torch.Tensor(labels).type(torch.FloatTensor).to(device)

    # Recuperation des donnees
    x00tr, x01tr, x02tr, x10tr, x11tr, x12tr, x20tr, x21tr, x22tr = train_data 

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


def run(model, tdata, tlabels, optim, loss_fn = nn.BCELoss(), num_epoch = 100):
    for epoch in tqdm(np.arange(num_epoch)):
        trainloss, trainacc = train_epoch(tdata, tlabels, model,loss_fn, optim)
        print(f'\nLoss train {trainloss} and accuracy {trainacc}\n')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 0.001
model = Model(d1=3,d2=2*32,d3=2*32,d4=2*32,d5=2*32,d6=2*32,d7=2*32,d8=2*32,n_c=1).to(device)
optim = torch.optim.Adam(list(model.parameters()),lr = lr)
optim.zero_grad()

indata = (x0_0_tr[0], x0_1_tr[0], x0_2_tr[0], x1_0_tr[0], x1_1_tr[0], x1_2_tr[0], x2_0_tr[0], x2_1_tr[0], x2_2_tr[0])
run(model, indata ,training_labels[0], optim)
# print(len(training_labels[0]))
# sur le fold 0

# ATTENTION on ne peut pas faire de dataloader car les matrices d'incidences sont de dimensions differentes en plus padder les simplexes n'aurait pas de sens.
# traindata = CustomDset(x0_0_tr[0], x0_1_tr[0], x0_2_tr[0], x1_0_tr[0], x1_1_tr[0], x1_2_tr[0], x2_0_tr[0], x2_1_tr[0], x2_2_tr[0], training_labels[0])
# train_dataloader = DataLoader(traindata, batch_size=64, shuffle=True)

# for x0_0, x0_1, x0_2, x1_0, x1_1, x1_2, x2_0, x2_1, x2_2, y in train_dataloader:
#     #args, y = args[:,:-1], args[:,-1]
#     out = model(x0_0, x0_1, x0_2, x1_0, x1_1, x1_2, x2_0, x2_1, x2_2)
#     break
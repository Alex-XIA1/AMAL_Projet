from GraphUtils import *

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


def train_epoch(train_data, labels, model, loss_fn, optim, device = None, num_classes = 2):
    if device == None: 
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    allLoss = []
    model.train()

    labels = torch.Tensor(labels).type(torch.FloatTensor).to(device)

    # Recuperation des donnees
    x00tr, x01tr, x02tr, x10tr, x11tr, x12tr, x20tr, x21tr, x22tr = train_data 
    # on a 901 graphes
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
            # yhat = torch.cat((yhat, model(torch.tensor(x0_0).type(torch.FloatTensor).to(device),torch.tensor(x0_1).type(torch.FloatTensor).to(device),
		  	# torch.tensor(x0_2).type(torch.FloatTensor).to(device),torch.Tensor(x1_0).type(torch.FloatTensor).to(device),torch.Tensor(x1_1).type(torch.FloatTensor).to(device),
		  	# torch.Tensor(x1_2).type(torch.FloatTensor).to(device),torch.Tensor(x2_0).type(torch.FloatTensor).to(device),torch.Tensor(x2_1).type(torch.FloatTensor).to(device),
		  	# torch.Tensor(x2_2).type(torch.FloatTensor).to(device))), 0)
            yhat = torch.cat((yhat, model([[torch.tensor(x0_0).type(torch.FloatTensor).to(device),torch.tensor(x0_1).type(torch.FloatTensor).to(device),
		  	torch.tensor(x0_2).type(torch.FloatTensor).to(device)],[torch.Tensor(x1_0).type(torch.FloatTensor).to(device),torch.Tensor(x1_1).type(torch.FloatTensor).to(device),
		  	torch.Tensor(x1_2).type(torch.FloatTensor).to(device)],[torch.Tensor(x2_0).type(torch.FloatTensor).to(device),torch.Tensor(x2_1).type(torch.FloatTensor).to(device),
		  	torch.Tensor(x2_2).type(torch.FloatTensor).to(device)]])), 0)
        
        # print("It works ")
        # exit()
        
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
            yhat = torch.cat((yhat, model([[torch.tensor(x0_0).type(torch.FloatTensor).to(device),torch.tensor(x0_1).type(torch.FloatTensor).to(device),
		  	torch.tensor(x0_2).type(torch.FloatTensor).to(device)],[torch.Tensor(x1_0).type(torch.FloatTensor).to(device),torch.Tensor(x1_1).type(torch.FloatTensor).to(device),
		  	torch.Tensor(x1_2).type(torch.FloatTensor).to(device)],[torch.Tensor(x2_0).type(torch.FloatTensor).to(device),torch.Tensor(x2_1).type(torch.FloatTensor).to(device),
		  	torch.Tensor(x2_2).type(torch.FloatTensor).to(device)]])), 0)
        
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
        yhat = torch.cat((yhat, model([[torch.tensor(x0_0).type(torch.FloatTensor).to(device),torch.tensor(x0_1).type(torch.FloatTensor).to(device),
		  	torch.tensor(x0_2).type(torch.FloatTensor).to(device)],[torch.Tensor(x1_0).type(torch.FloatTensor).to(device),torch.Tensor(x1_1).type(torch.FloatTensor).to(device),
		  	torch.Tensor(x1_2).type(torch.FloatTensor).to(device)],[torch.Tensor(x2_0).type(torch.FloatTensor).to(device),torch.Tensor(x2_1).type(torch.FloatTensor).to(device),
		  	torch.Tensor(x2_2).type(torch.FloatTensor).to(device)]])), 0)
        
    yhats = torch.where(yhat > 0.5, 1, 0)
    #print(f'yhat is {yhat.size()} and labels is {labels[e].size()}')
    loss = loss_fn(torch.squeeze(yhat).type(torch.FloatTensor).to(device), labels)
    acc(yhats, labels)

    fpr, tpr, _ = roc_curve(torch.Tensor(labels).type(torch.FloatTensor), torch.squeeze(yhat).type(torch.FloatTensor).detach().numpy())
    aucscore = auc(fpr,tpr)

    prec, recall, _ = precision_recall_curve(torch.Tensor(labels).type(torch.FloatTensor), torch.squeeze(yhat).type(torch.FloatTensor).detach().numpy())
    pr_ap = average_precision_score(torch.Tensor(labels).type(torch.FloatTensor), torch.squeeze(yhat).type(torch.FloatTensor).detach().numpy())

    cm = confusion_matrix(torch.Tensor(yhats).type(torch.FloatTensor), torch.Tensor(labels).type(torch.FloatTensor))
    #print(cm)

    return loss.item(), acc.compute().item(), fpr, tpr, cm, aucscore, prec, recall, pr_ap


def run(model, tdata, tlabels, vdata, val_labels, testdata, testlabels, optim, loss_fn = nn.BCELoss(), num_epoch = 300):
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
    
    testloss, testacc, fpr, tpr, cm, aucscore, precision, recall, pr_ap = test_valide(testdata, testlabels, model, loss_fn)
    date = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    path = './img/graphclassif/'

    print(f'The final loss for test is {testloss} its accuracy is {testacc}')
    #The final loss for test is 0.3034552335739136 its accuracy is 0.8835821151733398 for 128*128
    #The final loss for test is 0.7049200534820557 its accuracy is 0.7117117047309875 for 64 x 64 decoder on fold 0
    # toutes les performances et loss pour train et validation
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
    plt.savefig(f'{path}{num_epoch}_model_{date}.pdf')
    #plt.show()

    # ROC AUC image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, estimator_name=f'ROC (AUC = {np.round(aucscore,2)})')
    # Precision Recall image
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall, estimator_name=f'PR (AP = {np.round(pr_ap,2)})')
    roc_display.plot(ax=ax1)
    pr_display.plot(ax=ax2)
    ax1.set_title("Courbe ROC sur données test")
    ax2.set_title("Courbe PR sur données test")
    plt.savefig(f'{path}{num_epoch}_model_ROC_{date}.pdf')
    #plt.show()

    # matrice de confusion test
    disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title("Matrice de confusion test")
    plt.savefig(f'{path}{num_epoch}_model_cm_{date}.pdf')
    #plt.show()


def runCrossVal(tdata, tlabels, vdata, val_labels, testdata, testlabels, loss_fn = nn.BCELoss(), num_epoch = 150):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    foldtrainloss = []
    foldtrainperfs = []
    foldvalloss = []
    foldvalperfs = []
    foldtestperfs = []
    foldtestloss = []
    foldtpr = []
    x00_tr, x01_tr, x02_tr, x10_tr, x11_tr, x12_tr, x20_tr, x21_tr, x22_tr = tdata
    x00_val, x01_val, x02_val, x10_val, x11_val, x12_val, x20_val, x21_val, x22_val = vdata
    x00_test, x01_test, x02_test, x10_test, x11_test, x12_test, x20_test, x21_test, x22_test = testdata
    # le fpr au milieu
    mean_fpr = np.linspace(0, 1, 100)

    # 10 folds
    for fold in range(len(x00_tr)):
        print(f'fold {fold+1}')
        
        trainf = (x00_tr[fold], x01_tr[fold], x02_tr[fold], x10_tr[fold], x11_tr[fold], x12_tr[fold], x20_tr[fold], x21_tr[fold], x22_tr[fold])
        valf = (x00_val[fold], x01_val[fold], x02_val[fold], x10_val[fold], x11_val[fold], x12_val[fold], x20_val[fold], x21_val[fold], x22_val[fold])
        testf = (x00_test[fold], x01_test[fold], x02_test[fold], x10_test[fold], x11_test[fold], x12_test[fold], x20_test[fold], x21_test[fold], x22_test[fold])

        lr = 0.001
        dimin = 64
        #model = Model(d1=3,d2=dimin,d3=dimin,d4=dimin,n_c=1).to(device)
        model = GraphModel([[3, 6, 18], [3, 12, 39], [3, 9, 30]],dimin, dimin, dimin, 1, 3, 3 ).to(device)
        optim = torch.optim.Adam(list(model.parameters()),lr = lr)
        optim.zero_grad()
        epochtrainloss = []
        epochtrainperfs = []
        epochvalidloss = []
        epochvalidperfs = []

        # pour chaque fold on fait 150 epochs
        for epoch in tqdm(np.arange(num_epoch)):
            trainloss, trainacc = train_epoch(trainf, tlabels[fold], model,loss_fn, optim)
            validloss, validacc = valida_epoch(valf,val_labels[fold],model, loss_fn)
            print(f'\nLoss train {trainloss} and accuracy {trainacc}\n')
            print(f'\nLoss validation {validloss} and accuracy {validacc}\n')
            epochtrainloss.append(trainloss)
            epochvalidloss.append(validloss)
            epochtrainperfs.append(trainacc)
            epochvalidperfs.append(validacc)
        
        
        testloss, testacc, fpr, tpr, cm, aucscore, precision, recall, pr_ap = test_valide(testf, testlabels[fold][0], model, loss_fn)
        # test
        foldtestperfs.append(testacc)
        foldtestloss.append(testloss)
        # fold loss de train
        foldtrainloss.append(epochtrainloss)
        foldtrainperfs.append(epochtrainperfs)
        # validation
        foldvalloss.append(epochvalidloss)
        foldvalperfs.append(epochvalidperfs)

        plt.plot(fpr, tpr, 'b', alpha=0.15)
        tpr = np.interp(mean_fpr, fpr, tpr)
        tpr[0] = 0.0
        foldtpr.append(tpr)
    foldtpr = np.array(foldtpr)
    mean_tprs = foldtpr.mean(axis=0)
    mean_tprs[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tprs)
    
    date = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
    path = './img/graphclassif/'

    print(f'The final loss for test is {np.mean(foldtestloss)} its accuracy is {np.mean(foldtestperfs)} (std = {np.std(foldtestperfs)})')
    #The final loss for test is 0.5696458011865616 its accuracy is 0.7412564277648925 64 * 64
    # The final loss for test is 0.5759151399135589 its accuracy is 0.7565154314041138 (std = 0.053041954341405705)
    # The final loss for test is 0.8550497889518738 its accuracy is 0.7348777234554291 +- 0.03, 128 * 128
    # toutes les performances et loss pour train et validation
    # ROC AUC image
    plt.plot(mean_fpr,mean_tprs,'b',label=f'Mean ROC (AUC = {np.round(mean_auc,2)})')
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.savefig(f'{path}{num_epoch}_modelfold_ROC_{date}.pdf')
    plt.show()
    
    plt.subplot(2,2,1)
    plt.plot(np.array(np.mean(foldtrainloss,axis=0)))
    plt.title("loss train")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    
    plt.subplot(2,2,2)
    plt.plot(np.array(np.mean(foldtrainperfs,axis=0)))
    plt.title("performances train")
    plt.xlabel("epoch")
    plt.ylabel("performances")
    
    plt.subplot(2,2,3)
    plt.plot(np.array(np.mean(foldvalloss,axis=0)))
    plt.title("loss validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.subplot(2,2,4)
    plt.plot(np.array(np.mean(foldvalperfs,axis=0)))
    plt.title("performances validation")
    plt.xlabel("epoch")
    plt.ylabel("performances")

    plt.tight_layout()
    plt.savefig(f'{path}{num_epoch}_modelfold_{date}.pdf')
    #plt.show()

    

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# lr = 0.001
# dimin = 32
# model = Model(d1=3,d2=2*dimin,d3=2*dimin,d4=2*dimin,n_c=1).to(device)
# optim = torch.optim.Adam(list(model.parameters()),lr = lr)
# optim.zero_grad()



# print(alllabstr.shape)
# print(alllabstest.shape)

# Nk x Nk
# N x Nk x Dk
# nik, ij -> nik 
# concatener
#indata = (x0_0_tr[0], x0_1_tr[0], x0_2_tr[0], x1_0_tr[0], x1_1_tr[0], x1_2_tr[0], x2_0_tr[0], x2_1_tr[0], x2_2_tr[0])
#valdata = (x0_0_val[0], x0_1_val[0], x0_2_val[0], x1_0_val[0], x1_1_val[0], x1_2_val[0], x2_0_val[0], x2_1_val[0], x2_2_val[0])

indata = (x0_0_tr, x0_1_tr, x0_2_tr, x1_0_tr, x1_1_tr, x1_2_tr, x2_0_tr, x2_1_tr, x2_2_tr)
valdata = (x0_0_val, x0_1_val, x0_2_val, x1_0_val, x1_1_val, x1_2_val, x2_0_val, x2_1_val, x2_2_val)
testdata = (x0_0_test, x0_1_test, x0_2_test, x1_0_test, x1_1_test, x1_2_test, x2_0_test, x2_1_test, x2_2_test)

runCrossVal(indata ,training_labels, valdata, val_labels, testdata, testing_labels)

# lr = 0.001
# dimin = 64
# #model = Model(d1=3,d2=dimin,d3=dimin,d4=dimin,n_c=1).to(device)
# model = GraphModel([[3, 6, 18], [3, 12, 39], [3, 9, 30]],dimin, dimin, dimin, 1, 3, 3 ).to(device)
# optim = torch.optim.Adam(list(model.parameters()),lr = lr)
# optim.zero_grad()
#run(model, indata ,alllabstr, valdata, alllabsval, testdata, alllabstest, optim)
#print(len(training_labels[0]))

# sur le fold 0

# ATTENTION on ne peut pas faire de dataloader car les matrices d'incidences sont de dimensions differentes en plus padder les simplexes n'aurait pas de sens.
# traindata = CustomDset(x0_0_tr[0], x0_1_tr[0], x0_2_tr[0], x1_0_tr[0], x1_1_tr[0], x1_2_tr[0], x2_0_tr[0], x2_1_tr[0], x2_2_tr[0], training_labels[0])
# train_dataloader = DataLoader(traindata, batch_size=64, shuffle=True)

# for x0_0, x0_1, x0_2, x1_0, x1_1, x1_2, x2_0, x2_1, x2_2, y in train_dataloader:
#     #args, y = args[:,:-1], args[:,-1]
#     out = model(x0_0, x0_1, x0_2, x1_0, x1_1, x1_2, x2_0, x2_1, x2_2)
#     break
import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import logging
import os
import time
import datetime
import matplotlib.pyplot as plt
import scipy.io

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Prokaryotic
# Synthetic3d
# Aloi_deep
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
Dataname = 'Aloi_deep'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
# parser.add_argument("--num_neighbors", default=3)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "Aloi_deep":
    args.con_epochs = 200
    seed = 10

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, hs = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

def contrastive_train(epoch):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, y, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        # xrs, zs, hs, commonz, S = model(xs)
        xrs, zs, hs = model(xs)
        commonz, P = model.ASAF(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(1*criterion.CwCL(hs[v], commonz, P))
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
    return  tot_loss / len(data_loader)

losses = []
accs = []
nmis = []
purs = []
zses = []
hses = []
comZs = []
best_acc = 0
best_epoch_nmi = 0
best_epoch_pur = 0
best_epoch_ari = 0
best_epoch = 0
best_epoch_loss = 0
if not os.path.exists('./models'):
    os.makedirs('./models')
start_time = time.time()
T = 1
for i in range(T):
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, device).to(device)
    epoch = 1
    while epoch <= args.mse_epochs:
        pretrain(epoch)
        epoch += 1
    while epoch <= args.mse_epochs + args.con_epochs:
        loss_= contrastive_train(epoch)
        losses.append(loss_)
        # acc, nmi, pur, ari = valid(model, device, dataset, view, data_size, class_num, eval_h=True)
        acc, nmi, pur, ari, labels, commonZ = valid(model, device, dataset, view, data_size, class_num, eval_h=True)
        accs.append(acc)
        nmis.append(nmi)
        purs.append(pur)
        comZs.append(commonZ)

        if acc>best_acc:
            best_acc = np.copy(acc)
            best_epoch_nmi = np.copy(nmi)
            best_epoch_pur = np.copy(pur)
            best_epoch_ari = np.copy(ari)
            best_epoch = epoch
            best_epoch_loss = loss_
        if epoch == args.mse_epochs + args.con_epochs:
            print('---------train over---------')
            print(Dataname)
            print('Clustering results: ACC, NMI, PUR, ARI, EPOCH, LOSS')
            print('{:.4f} {:.4f} {:.4f} {:.4f} {} {:.6f}'.format(best_acc, best_epoch_nmi, best_epoch_pur, best_epoch_ari, best_epoch, best_epoch_loss))
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')
            print('Saving..')
        epoch += 1



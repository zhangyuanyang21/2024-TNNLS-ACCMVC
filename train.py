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
from utils import cal_weights_via_CAN
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
# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.
if args.dataset == "MNIST-USPS":
    args.con_epochs = 100
    # args.con_epochs = 11
    seed = 10
if args.dataset == "CCV":
    # args.con_epochs = 40
    args.con_epochs = 100
    # args.con_epochs = 19
    seed = 3
if args.dataset == "Caltech-2V":
    # args.con_epochs = 174
    args.con_epochs = 200
    seed = 10
if args.dataset == "Caltech-3V":
    args.con_epochs = 200
    # args.con_epochs = 151
    seed = 10
if args.dataset == "Caltech-4V":
    args.con_epochs = 200
    # args.con_epochs = 180
    seed = 10
if args.dataset == "Caltech-5V":
    args.con_epochs = 200
    # args.con_epochs = 140
    seed = 5
if args.dataset == "Hdigit":
    args.con_epochs =100
    # args.con_epochs =20
    # args.con_epochs = 18
    seed = 10
if args.dataset == "YouTubeFace":
    args.con_epochs = 100
    seed = 10
if args.dataset == "Cifar10":
    args.con_epochs = 100
    seed = 10
if args.dataset == "Cifar100":
    args.con_epochs = 200
    # args.con_epochs = 100
    seed = 10
if args.dataset == "Prokaryotic":
    args.con_epochs = 100
    # args.con_epochs = 65
    seed = 10
if args.dataset == "Synthetic3d":
    args.con_epochs = 100
    # args.con_epochs = 82
    # args.con_epochs = 90
    seed = 10
#add
if args.dataset == "BDGP":
    # args.con_epochs = 10
    args.con_epochs = 100
    seed = 10
if args.dataset == "Fashion":
    args.con_epochs = 100
    # args.con_epochs = 38
    # args.con_epochs = 78
    seed = 10
if args.dataset == "handwritten":
    args.con_epochs = 300
    seed = 10
if args.dataset == "COIL20-3v":
    args.con_epochs = 300
    seed = 10
if args.dataset == "100leaves":
    args.con_epochs =300
    seed = 10
if args.dataset == "Caltech101-7":
    args.con_epochs = 100
    # args.con_epochs = 42
    seed = 10
if args.dataset == "Mfeat":
    args.con_epochs =100
    seed = 10
if args.dataset == "NGs":
    # args.con_epochs = 34
    args.con_epochs = 58
    seed = 10
if args.dataset == "Scene_15":
    args.con_epochs =300
    seed = 10
if args.dataset == "Citeseer":
    args.con_epochs = 100
    seed = 10
if args.dataset == "Cora":
    args.con_epochs = 100
    seed = 10
if args.dataset == "uci-digit":
    # args.con_epochs = 43
    # args.con_epochs = 57
    # args.con_epochs = 61
    args.con_epochs = 100
    seed = 10
if args.dataset == "100leaves":
    args.con_epochs =100
    seed = 10
if args.dataset == "NoisyMNIST":
    args.con_epochs = 100
    seed = 10
if args.dataset == "Aloi_deep":
    args.con_epochs = 200
    seed = 10

# def constructW(Dist, n_neighbors):
#     # construct a knn graph
#     neighbors_graph = kneighbors_graph(
#         Dist, n_neighbors, mode='connectivity', include_self=False)
#     W = 0.5 * (neighbors_graph + neighbors_graph.T)
#     return W

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
        # xrs, zs, hs, _, _ = model(xs)
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
        commonz, S = model.GCFAgg(xs)
        # commonz = model.GCFAgg(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(1*criterion.forward_feature(hs[v], commonz, S))
            # loss_list.append(1*criterion.forward_feature_original(hs[v], commonz))
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))
    #返回loss是为了画loss
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
            print('{:.4f} {:.4f} {:.4f} {:.4f} {} {:.6f}'.format(acc, nmi, pur, ari, epoch, loss_))
            print('{:.4f} {:.4f} {:.4f} {:.4f} {} {:.6f}'.format(best_acc, best_epoch_nmi, best_epoch_pur, best_epoch_ari, best_epoch, best_epoch_loss))
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')
            print('Saving..')
        epoch += 1
#######
end_time = time.time()
total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))
print("Training time {} ".format(total_time_str))
#######
#logging
if not os.path.exists('log'):
    os.mkdir('log')
timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
log_filename = f'{Dataname}_{timestamp}.log'
log_path = os.path.join('log', log_filename)
logging.basicConfig(filename=log_path, level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s')
logging.info('---------train over---------')
logging.info(Dataname)
logging.info('Clustering results: ACC, NMI, PUR, ARI, EPOCH, LOSS')
logging.info('{:.4f} {:.4f} {:.4f} {:.4f} {} {:.6f}'.format(acc, nmi, pur, ari, epoch-1, loss_))
logging.info('{:.4f} {:.4f} {:.4f} {:.4f} {} {:.6f}'.format(best_acc, best_epoch_nmi, best_epoch_pur, best_epoch_ari,best_epoch, best_epoch_loss))
logging.info('Training time {} '.format(total_time_str))

#保存表示commonZ
# if not os.path.exists('commonZ'):
#     os.mkdir('commonZ')
# mdic1 = {"comz": comZs[0], "labels": labels}
# scipy.io.savemat(os.path.join('commonZ', f'{Dataname}_comz_0.mat'), mdic1)
# mdic5 = {"comz": comZs[5], "labels": labels}
# scipy.io.savemat(os.path.join('commonZ', f'{Dataname}_comz_5.mat'), mdic5)
# mdic10 = {"comz": comZs[10], "labels": labels}
# scipy.io.savemat(os.path.join('commonZ', f'{Dataname}_comz_10.mat'), mdic10)
# mdic15 = {"comz": comZs[15], "labels": labels}
# scipy.io.savemat(os.path.join('commonZ', f'{Dataname}_comz_15.mat'), mdic15)
# mdic20 = {"comz": comZs[20], "labels": labels}
# scipy.io.savemat(os.path.join('commonZ', f'{Dataname}_comz_20.mat'), mdic20)
# mdic2 = {"comz": comZs[25], "labels": labels}
# scipy.io.savemat(os.path.join('commonZ', f'{Dataname}_comz_25.mat'), mdic2)
# mdic21 = {"comz": comZs[37], "labels": labels}
# scipy.io.savemat(os.path.join('commonZ', f'{Dataname}_comz_37.mat'), mdic21)
# mdic3 = {"comz": comZs[50], "labels": labels}
# scipy.io.savemat(os.path.join('commonZ', f'{Dataname}_comz_50.mat'), mdic3)
# mdic4 = {"comz": comZs[75], "labels": labels}
# scipy.io.savemat(os.path.join('commonZ', f'{Dataname}_comz_75.mat'), mdic4)
# mdic15 = {"comz": comZs[99], "labels": labels}
# scipy.io.savemat(os.path.join('commonZ', f'{Dataname}_comz_100.mat'), mdic15)
# mdic6 = {"comz": comZs[125], "labels": labels}
# scipy.io.savemat(os.path.join('commonZ', f'{Dataname}_comz_125.mat'), mdic6)
# mdic7 = {"comz": comZs[150], "labels": labels}
# scipy.io.savemat(os.path.join('commonZ', f'{Dataname}_comz_150.mat'), mdic7)
# mdic8 = {"comz": comZs[175], "labels": labels}
# scipy.io.savemat(os.path.join('commonZ', f'{Dataname}_comz_175.mat'), mdic8)
# mdic9 = {"comz": comZs[199], "labels": labels}
# scipy.io.savemat(os.path.join('commonZ', f'{Dataname}_comz_200.mat'), mdic9)

#保存表示Hs, Zs
# if not os.path.exists('commonZ'):
#     os.mkdir('commonZ')
# mdic1 = {"Hs": Hs[0], "labels": labels}
# scipy.io.savemat(os.path.join('commonZ', f'{Dataname}_Hs.mat'), mdic1)
# mdic2 = {"Zs": Zs[0], "labels": labels}
# scipy.io.savemat(os.path.join('commonZ', f'{Dataname}_Zs.mat'), mdic2)
######

#######plot clustering performance
x = np.arange(0, args.con_epochs, 1)
fig, ax = plt.subplots(figsize=(8, 6))
# plot clustering performance
ax.plot(x, accs, 'orange', label='ACC', linewidth='2')
ax.plot(x, nmis, 'lime', label='NMI', linewidth='2')
ax.plot(x, purs, 'skyblue', label='Purity', linewidth='2')

# plot loss
ax2 = ax.twinx()
ax2.plot(x, losses, 'r--', label='Loss', linewidth='2')

# set axis labels and titles
ax.set_title(Dataname, fontsize=10, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=10, fontweight='bold')
ax.set_ylabel('Clustering Performance', fontsize=10, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=10, fontweight='bold')

# set tick parameters
ax.tick_params(axis='both', direction='in', labelsize=10)
ax2.tick_params(axis='both', direction='in', labelsize=10)

# set grid and legend
ax.grid(color='gray', linestyle='-', linewidth=0.25, alpha=0.5)
# ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))
# ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 0.95))
fig.legend(loc=1, bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
# set image size and save, Set the figure size to match the CVPR paper format
fig.tight_layout()
fig.set_size_inches(3.39, 2.5)  # set size to 3.39 inches x 2.5 inches
fig.savefig('{}.png'.format(Dataname), dpi=300, bbox_inches='tight')
# adjust the popup window size
mng = plt.get_current_fig_manager()
mng.resize(800, 600)
plt.show()


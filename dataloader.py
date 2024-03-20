from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import h5py

class ALOI_DEEP():
    def __init__(self, path):
        # scaler = MinMaxScaler()
        data = scipy.io.loadmat(path + 'aloideep3v.mat')
        self.Y = data['truth'].astype(np.int32).reshape(10800,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)
        # self.V1 = scaler.fit_transform(data['X'][0][0].T.astype(np.float32))
        # self.V2 = scaler.fit_transform(data['X'][0][1].T.astype(np.float32))
        # self.V3 = scaler.fit_transform(data['X'][0][2].T.astype(np.float32))
    def __len__(self):
        return 10800
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "Aloi_deep":
        dataset = ALOI_DEEP('./data/')
        dims = [2048, 4096, 2048]
        view = 3
        data_size = 10800
        class_num = 100
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num

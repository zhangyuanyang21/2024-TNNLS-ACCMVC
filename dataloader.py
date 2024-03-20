from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch
import h5py

class BDGP(Dataset):
    def __init__(self, path):
        # x = scipy.io.loadmat(path+'BDGP.mat')
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()
class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000
    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Scene_15(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Scene-15.mat')['Y'].astype(np.int32).reshape(4485,)
        mat = scipy.io.loadmat(path + 'Scene-15.mat')
        data_x = mat['X'][0]
        self.V1 = data_x[0].astype(np.float32)
        self.V2 = data_x[1].astype(np.float32)
        self.V3 = data_x[2].astype(np.float32)
    def __len__(self):
        return 4485
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3= self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

def normalize(x):
        """Normalize"""
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x

class Caltech101(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Caltech101-20.mat')['Y'].astype(np.int32).reshape(2386,)
        mat = scipy.io.loadmat(path + 'Caltech101-20.mat')
        data_x = mat['X'][0]
        self.V1 = normalize(data_x[0]).astype(np.float32)
        self.V2 = normalize(data_x[1]).astype(np.float32)
        self.V3 = normalize(data_x[2]).astype(np.float32)
        self.V4 = normalize(data_x[3]).astype(np.float32)
        self.V5 = normalize(data_x[4]).astype(np.float32)
        self.V6 = normalize(data_x[5]).astype(np.float32)

    def __len__(self):
        return 2386
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4), torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400


    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
class leaves():
    def __init__(self, path):
        data = scipy.io.loadmat(path + '100leaves.mat')
        scaler = MinMaxScaler()
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(1600,)
        self.V1 = scaler.fit_transform(data['data'][0][0].T.astype(np.float32))
        self.V2 = scaler.fit_transform(data['data'][0][1].T.astype(np.float32))
        self.V3 = scaler.fit_transform(data['data'][0][2].T.astype(np.float32))
    def __len__(self):
        return 1600
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3= self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Hdigit():
    def __init__(self, path):
        scaler = MinMaxScaler()
        data = scipy.io.loadmat(path + 'Hdigit.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(10000,)
        self.V1 = scaler.fit_transform(data['data'][0][0].T.astype(np.float32))
        self.V2 = scaler.fit_transform(data['data'][0][1].T.astype(np.float32))
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Mfeat():
    def __init__(self, path):
        scaler = MinMaxScaler()
        data = scipy.io.loadmat(path + 'Mfeat.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(2000,)
        self.V1 = scaler.fit_transform(data['data'][0][0].T.astype(np.float32))
        self.V2 = scaler.fit_transform(data['data'][0][1].T.astype(np.float32))
        self.V3 = scaler.fit_transform(data['data'][0][2].T.astype(np.float32))
        self.V4 = scaler.fit_transform(data['data'][0][3].T.astype(np.float32))
        self.V5 = scaler.fit_transform(data['data'][0][4].T.astype(np.float32))
        self.V6 = scaler.fit_transform(data['data'][0][5].T.astype(np.float32))
    def __len__(self):
        return 2000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4), torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class MSRC():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'MSRC-v1.mat')
        self.Y = data['Y'].astype(np.int32).reshape(210,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)
        self.V4 = data['X'][0][3].astype(np.float32)
    def __len__(self):
        return 210
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class ORL():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'ORL.mat')
        self.Y = data['gt'].astype(np.int32).reshape(400,)
        self.V1 = data['X'][0][0].T.astype(np.float32)
        self.V2 = data['X'][0][1].T.astype(np.float32)
        self.V3 = data['X'][0][2].T.astype(np.float32)
    def __len__(self):
        return 400
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Yale():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Yale.mat')
        self.Y = data['gt'].astype(np.int32).reshape(165,)
        self.V1 = data['X'][0][0].T.astype(np.float32)
        self.V2 = data['X'][0][1].T.astype(np.float32)
        self.V3 = data['X'][0][2].T.astype(np.float32)
    def __len__(self):
        return 165
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Caltech101_7():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Caltech101-7.mat')
        self.Y = data['y'].astype(np.int32).reshape(1474,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)
        self.V4 = data['X'][0][3].astype(np.float32)
        self.V5 = data['X'][0][4].astype(np.float32)
        self.V6 = data['X'][0][5].astype(np.float32)
    def __len__(self):
        return 1474
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4), torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Sources_3():
    def __init__(self, path):
        data = scipy.io.loadmat(path + '3Sources.mat')
        self.Y = data['gt'].astype(np.int32).reshape(169,)
        self.V1 = data['X'][0][0].T.astype(np.float32)
        self.V2 = data['X'][0][1].T.astype(np.float32)
        self.V3 = data['X'][0][2].T.astype(np.float32)
    def __len__(self):
        return 169
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class WebKB():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'WebKB.mat')
        self.Y = data['gt'].astype(np.int32).reshape(203,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][0][1].T.astype(np.float32)
        self.V3 = data['data'][0][2].T.astype(np.float32)
    def __len__(self):
        return 203
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Citeseer():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Citeseer.mat')
        self.Y = data['y'].astype(np.int32).reshape(3312,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
    def __len__(self):
        return 3312
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Cora():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Cora.mat')
        self.Y = data['y'].astype(np.int32).reshape(2708,)
        self.V1 = data['coracites'].astype(np.float32)
        self.V2 = data['coracontent'].astype(np.float32)
    def __len__(self):
        return 2708
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Washington():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Washington.mat')
        self.Y = data['Y'][0][0].astype(np.int32).reshape(230,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
    def __len__(self):
        return 230
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class COIL20_3v():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'COIL20-3v.mat')
        self.Y = data['Y'].astype(np.int32).reshape(1440,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)
    def __len__(self):
        return 1440
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Animal():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Animal.mat')
        self.Y = data['Y'].astype(np.int32).reshape(11673,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)
        self.V4 = data['X'][0][3].astype(np.float32)
    def __len__(self):
        return 11673
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
# class NoisyMNIST(Dataset):
#     def __init__(self, path):
#         self.Y = scipy.io.loadmat(path + 'NoisyMNIST.mat')['tuneLabel'].astype(np.int32).reshape(10000,)
#         self.V1 = scipy.io.loadmat(path + 'NoisyMNIST.mat')['XV1'].astype(np.float32)
#         self.V2 = scipy.io.loadmat(path + 'NoisyMNIST.mat')['XV2'].astype(np.float32)
#
#     def __len__(self):
#         return 10000
#     def __getitem__(self, idx):
#
#         x1 = self.V1[idx].reshape(784)
#         x2 = self.V2[idx].reshape(784)
#         return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class DataSet_NoisyMNIST(object):
    def __init__(self, images1, images2, labels, fake_data=False, one_hot=False,
                 dtype=np.float32):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        if dtype not in (np.uint8, np.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' % dtype)

        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images1.shape[0] == labels.shape[0], (
                    'images1.shape: %s labels.shape: %s' % (images1.shape,
                                                            labels.shape))
            assert images2.shape[0] == labels.shape[0], (
                    'images2.shape: %s labels.shape: %s' % (images2.shape,
                                                            labels.shape))
            self._num_examples = images1.shape[0]

            if dtype == np.float32 and images1.dtype != np.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                # print("type conversion view 1")
                images1 = images1.astype(np.float32)

            if dtype == np.float32 and images2.dtype != np.float32:
                # print("type conversion view 2")
                images2 = images2.astype(np.float32)

        self._images1 = images1
        self._images2 = images2
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images1(self):
        return self._images1

    @property
    def images2(self):
        return self._images2

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [fake_image for _ in range(batch_size)], [fake_label for _
                                                                                                      in range(
                    batch_size)]

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images1 = self._images1[perm]
            self._images2 = self._images2[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        return self._images1[start:end], self._images2[start:end], self._labels[start:end]
    # elif data_name in ['NoisyMNIST']:
    #     data = sio.loadmat('./data/NoisyMNIST.mat')
    #     train = DataSet_NoisyMNIST(data['X1'], data['X2'], data['trainLabel'])
    #     tune = DataSet_NoisyMNIST(data['XV1'], data['XV2'], data['tuneLabel'])
    #     test = DataSet_NoisyMNIST(data['XTe1'], data['XTe2'], data['testLabel'])
    #     # X_list.append(np.concatenate([tune.images1, test.images1], axis=0))
    #     # X_list.append(np.concatenate([tune.images2, test.images2], axis=0))
    #     # Y_list.append(np.concatenate([np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))
    #     X_list.append(np.concatenate([train.images1, tune.images1, test.images1], axis=0))
    #     X_list.append(np.concatenate([train.images2, tune.images2, test.images2], axis=0))
    #     Y_list.append(np.concatenate([np.squeeze(train.labels[:, 0]), np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))
    #     print(Y_list[0])
    #     x1 = X_list[0]
    #     x2 = X_list[1]
    #     xx1 = np.copy(x1)
    #     xx2 = np.copy(x2)
    #     Y = np.copy(Y_list[0])
    #     index = [i for i in range(70000)]
    #     np.random.seed(784)
    #     np.random.shuffle(index)
    #     for i in range(70000):
    #         xx1[i] = x1[index[i]]                    # (70000, 784)
    #         xx2[i] = x2[index[i]]                    # (70000, 784)
    #         Y[i] = Y_list[0][index[i]]
    #     print(Y)
    #     X_list = [xx1, xx2]
    #     Y_list = [Y]
    # return X_list, Y_list
class NoisyMNIST(Dataset):
    def __init__(self, path):
        X_list = []
        Y_list = []
        data = scipy.io.loadmat(path + 'NoisyMNIST.mat')
        train = DataSet_NoisyMNIST(data['X1'], data['X2'], data['trainLabel'])
        tune = DataSet_NoisyMNIST(data['XV1'], data['XV2'], data['tuneLabel'])
        test = DataSet_NoisyMNIST(data['XTe1'], data['XTe2'], data['testLabel'])
        X_list.append(np.concatenate([train.images1, tune.images1, test.images1], axis=0))
        X_list.append(np.concatenate([train.images2, tune.images2, test.images2], axis=0))
        Y_list.append(np.concatenate(
            [np.squeeze(train.labels[:, 0]), np.squeeze(tune.labels[:, 0]), np.squeeze(test.labels[:, 0])]))
        print(Y_list[0])
        x1 = X_list[0]
        x2 = X_list[1]
        xx1 = np.copy(x1)
        xx2 = np.copy(x2)
        Y = np.copy(Y_list[0])
        index = [i for i in range(70000)]
        np.random.seed(784)
        np.random.shuffle(index)
        for i in range(70000):
            xx1[i] = x1[index[i]]  # (70000, 784)
            xx2[i] = x2[index[i]]  # (70000, 784)
            Y[i] = Y_list[0][index[i]]
        print(Y)
        self.Y = Y
        self.V1 = xx1
        self.V2 = xx2

    def __len__(self):
        return 70000
    def __getitem__(self, idx):

        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class NUSWIDEOBJ():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'NUSWIDEOBJ.mat')
        self.Y = data['Y'].astype(np.int32).reshape(30000,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
        self.V4 = data['X'][3][0].astype(np.float32)
        self.V5 = data['X'][4][0].astype(np.float32)
    def __len__(self):
        return 30000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4), torch.from_numpy(x5)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class YoutubeFace_sel_fea():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'YoutubeFace_sel_fea.mat')
        self.Y = data['Y'].astype(np.int32).reshape(101499,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
        self.V4 = data['X'][3][0].astype(np.float32)
        self.V5 = data['X'][4][0].astype(np.float32)
    def __len__(self):
        return 101499
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4), torch.from_numpy(x5)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class cifar_10():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar10.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class cifar_100():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'cifar100.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(50000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)
    def __len__(self):
        return 50000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class BDGP_fea():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'BDGP_fea.mat')
        self.Y = data['Y'].astype(np.int32).reshape(2500, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)

    def __len__(self):
        return 2500

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()
class fmnist():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'fmnist.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(60000, )
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)

    def __len__(self):
        return 60000

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()
class handwritten():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'handwritten.mat')
        self.Y = data['Y'].astype(np.int32).reshape(2000,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
        self.V4 = data['X'][3][0].astype(np.float32)
        self.V5 = data['X'][4][0].astype(np.float32)
        self.V6 = data['X'][5][0].astype(np.float32)
    def __len__(self):
        return 2000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4), torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class MNIST_fea():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'MNIST_fea.mat')
        self.Y = data['Y'].astype(np.int32).reshape(60000,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 60000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class NGs():
    def __init__(self, path):
        scaler = MinMaxScaler()
        data = scipy.io.loadmat(path + 'NGs.mat')
        self.Y = data['Y'].astype(np.int32).reshape(500,)
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))
    def __len__(self):
        return 500
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class prokaryotic():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'prokaryotic.mat')
        self.Y = data['Y'].astype(np.int32).reshape(551,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 551
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class proteinFold():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'proteinFold.mat')
        self.Y = data['Y'].astype(np.int32).reshape(694,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
        self.V4 = data['X'][3][0].astype(np.float32)
        self.V5 = data['X'][4][0].astype(np.float32)
        self.V6 = data['X'][5][0].astype(np.float32)
        self.V7 = data['X'][6][0].astype(np.float32)
        self.V8 = data['X'][7][0].astype(np.float32)
        self.V9 = data['X'][8][0].astype(np.float32)
        self.V10 = data['X'][9][0].astype(np.float32)
        self.V11 = data['X'][10][0].astype(np.float32)
        self.V12 = data['X'][11][0].astype(np.float32)
    def __len__(self):
        return 694
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        x7 = self.V7[idx]
        x8 = self.V8[idx]
        x9 = self.V9[idx]
        x10 = self.V10[idx]
        x11 = self.V11[idx]
        x12 = self.V12[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3),torch.from_numpy(x4), torch.from_numpy(x5), torch.from_numpy(x6),
                torch.from_numpy(x7), torch.from_numpy(x8), torch.from_numpy(x9),torch.from_numpy(x10), torch.from_numpy(x11), torch.from_numpy(x12)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class synthetic3d():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'synthetic3d.mat')
        self.Y = data['Y'].astype(np.int32).reshape(600,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 600
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class tinyimage():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'tinyimage.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(100000, )
        self.V1 = data['data'][0][0].T.astype(np.float32)
        self.V2 = data['data'][1][0].T.astype(np.float32)
        self.V3 = data['data'][2][0].T.astype(np.float32)

    def __len__(self):
        return 100000

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()
class uci_digit():
    def __init__(self, path):
        scaler = MinMaxScaler()
        data = scipy.io.loadmat(path + 'uci-digit.mat')
        self.Y = data['Y'].astype(np.int32).reshape(2000,)
        self.V1 = scaler.fit_transform(data['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(data['X'][1][0].astype(np.float32))
        self.V3 = scaler.fit_transform(data['X'][2][0].astype(np.float32))
        # self.V1 = data['X'][0][0].astype(np.float32)
        # self.V2 = data['X'][1][0].astype(np.float32)
        # self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 2000
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Wiki_fea():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Wiki_fea.mat')
        self.Y = data['Y'].astype(np.int32).reshape(2866,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
    def __len__(self):
        return 2866
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class WikipediaArticles():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'WikipediaArticles.mat')
        self.Y = data['Y'].astype(np.int32).reshape(693,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
    def __len__(self):
        return 693
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class caltech101_3view():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'caltech101_3view.mat')
        self.Y = data['Y'].astype(np.int32).reshape(512,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
    def __len__(self):
        return 512
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Caltech101_all_fea():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Caltech101-all_fea.mat')
        self.Y = data['Y'].astype(np.int32).reshape(9144,)
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)
        self.V3 = data['X'][2][0].astype(np.float32)
        self.V4 = data['X'][3][0].astype(np.float32)
        self.V5 = data['X'][4][0].astype(np.float32)
    def __len__(self):
        return 9144
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4), torch.from_numpy(x5)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class voc():
    def __init__(self, path):
        data = np.load(path + 'voc.npz')
        self.Y = data['labels'].astype(np.int32).reshape(5649,)
        self.V1 = data['view_0'].astype(np.float32)
        self.V2 = data['view_1'].astype(np.float32)
    def __len__(self):
        return 5649
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class rgbd():
    def __init__(self, path):
        data = np.load(path + 'rgbd.npz')
        self.Y = data['labels'].astype(np.int32).reshape(1449,)
        self.V1 = data['view_0'].astype(np.float32)
        self.V2 = data['view_1'].astype(np.float32)
    def __len__(self):
        return 1449
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class SUNRGBD_fea():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'SUNRGBD_fea.mat')
        self.Y = data['Y'].astype(np.int32).reshape(10335, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)

    def __len__(self):
        return 10335

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class AwA_fea():
    def __init__(self, path):
        mat = h5py.File(path + 'AwA_fea.mat')
        data = scipy.io.loadmat(path + 'AwA_fea.mat')
        self.Y = data['Y'].astype(np.int32).reshape(10335, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][1][0].astype(np.float32)

    def __len__(self):
        return 10335

    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class Caltech101_all():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Caltech101-all.mat')
        self.Y = data['Y'].astype(np.int32).reshape(9144, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
        self.V3 = data['X'][0][2].astype(np.float32)
        self.V4 = data['X'][0][3].astype(np.float32)
        self.V5 = data['X'][0][4].astype(np.float32)
        self.V6 = data['X'][0][5].astype(np.float32)
    def __len__(self):
        return 9144
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        x3 = self.V3[idx]
        x4 = self.V4[idx]
        x5 = self.V5[idx]
        x6 = self.V6[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3), torch.from_numpy(x4),
                torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
class bbcsport():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'bbcsport_2.mat')
        self.Y = data['Y'].astype(np.int32).reshape(544, )
        self.V1 = data['X'][0][0].astype(np.float32)
        self.V2 = data['X'][0][1].astype(np.float32)
    def __len__(self):
        return 544
    def __getitem__(self, idx):
        x1 = self.V1[idx]
        x2 = self.V2[idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
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
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "Scene_15":
        dataset = Scene_15('./data/')
        dims = [20, 59, 40]
        view = 3
        data_size = 4485
        class_num = 15

    elif dataset == "Caltech101-20":
        dataset = Caltech101('./data/')
        dims = [48, 40, 254, 1984, 512, 928]
        view = 6
        data_size = 2386
        class_num = 20
    elif dataset == "100leaves":
        dataset = leaves('./data/')
        dims = [64, 64, 64]
        view = 3
        data_size = 1600
        class_num = 100
    elif dataset == "Hdigit":
        dataset = Hdigit('./data/')
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
    elif dataset == "Mfeat":
        dataset = Mfeat('./data/')
        dims = [216, 76, 64, 6, 240, 47]
        view = 6
        data_size = 2000
        class_num = 10
    elif dataset == "MSRC-v1":
        dataset = MSRC('./data/')
        dims = [24, 512, 256, 254]
        view = 4
        data_size = 210
        class_num = 7
    elif dataset == "ORL":
        dataset = ORL('./data/')
        dims = [4096, 3304, 6750]
        view = 3
        data_size = 400
        class_num = 40
    elif dataset == "Yale":
        dataset = Yale('./data/')
        dims = [4096, 3304, 6750]
        view = 3
        data_size = 165
        class_num = 15
    elif dataset == "Caltech101-7":
        dataset = Caltech101_7('./data/')
        dims = [48, 40, 254, 1984, 512, 928]
        view = 6
        data_size = 1474
        class_num = 7
    elif dataset == "3Sources":
        dataset = Sources_3('./data/')
        dims = [3560, 3631, 3068]
        view = 3
        data_size = 169
        class_num = 6
    elif dataset == "WebKB":
        dataset = WebKB('./data/')
        dims = [1703, 230, 230]
        view = 3
        data_size = 203
        class_num = 4
    elif dataset == "Citeseer":
        dataset = Citeseer('./data/')
        dims = [3312, 3703]
        view = 2
        data_size = 3312
        class_num = 6
    elif dataset == "Cora":
        dataset = Cora('./data/')
        dims = [2708, 1433]
        view = 2
        data_size = 2708
        class_num = 7
    elif dataset == "Washington":
        dataset = Washington('./data/')
        dims = [230, 1703]
        view = 2
        data_size = 230
        class_num = 5
    elif dataset == "COIL20-3v":
        dataset = COIL20_3v('./data/')
        dims = [1024, 3304, 6750]
        view = 3
        data_size = 1440
        class_num = 20
    elif dataset == "Animal":
        dataset = Animal('./data/')
        dims = [2689, 2000, 2001, 2000]
        view = 4
        data_size = 11673
        class_num = 20
    elif dataset == "NoisyMNIST":
        dataset = NoisyMNIST('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 70000
    elif dataset == "NUSWIDEOBJ":
        dataset = NUSWIDEOBJ('./data/')
        dims = [65, 226, 145, 74, 129]
        view = 5
        data_size = 30000
        class_num = 31
    elif dataset == "YouTubeFace":
        dataset = YoutubeFace_sel_fea('./data/')
        dims = [64, 512, 64, 647, 838]
        view = 5
        data_size = 101499
        class_num = 31
    elif dataset == "cifar10":
        dataset = cifar_10('./data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 10
    elif dataset == "cifar100":
        dataset = cifar_100('./data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 100
    elif dataset == "BDGP_fea":
        dataset = BDGP_fea('./data/')
        dims = [1000, 500, 250]
        view = 3
        data_size = 2500
        class_num = 5
    elif dataset == "fmnist":
        dataset = fmnist('./data/')
        dims = [512, 512, 1280]
        view = 3
        data_size = 60000
        class_num = 10
    elif dataset == "handwritten":
        dataset = handwritten('./data/')
        dims = [240, 76, 216,47,64,6]
        view = 6
        data_size = 2000
        class_num = 10
    elif dataset == "MNIST_fea":
        dataset = MNIST_fea('./data/')
        dims = [342, 1024, 64]
        view = 3
        data_size = 60000
        class_num = 10
    elif dataset == 'NGs':
        dataset = NGs('./data/')
        dims = [2000, 2000, 2000]
        view = 3
        data_size = 500
        class_num = 5
    elif dataset == 'Prokaryotic':
        dataset = prokaryotic('./data/')
        dims = [438, 3, 393]
        view = 3
        data_size = 551
        class_num = 4
    elif dataset == 'proteinFold':
        dataset = proteinFold('./data/')
        dims = [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,27]
        view = 12
        data_size = 694
        class_num = 27
    elif dataset == 'Synthetic3d':
        dataset = synthetic3d('./data/')
        dims = [3, 3, 3]
        view = 3
        data_size = 600
        class_num = 3
    elif dataset == 'tinyimage':
        dataset = tinyimage('./data/')
        dims = [512, 512, 1280]
        view = 3
        data_size = 100000
        class_num = 200
    elif dataset == 'uci-digit':
        dataset = uci_digit('./data/')
        dims = [64, 76, 216]
        view = 3
        data_size = 2000
        class_num = 10
    elif dataset == 'Wiki_fea':
        dataset = Wiki_fea('./data/')
        dims = [128, 10]
        view = 2
        data_size = 2866
        class_num = 10
    elif dataset == 'WikipediaArticles':
        dataset = WikipediaArticles('./data/')
        dims = [128, 10]
        view = 2
        data_size = 693
        class_num = 10
    elif dataset == 'caltech101_3view':
        dataset = caltech101_3view('./data/')
        dims = [254, 512, 36]
        view = 3
        data_size = 512
        class_num = 11
    elif dataset == 'Caltech101-all_fea':
        dataset = Caltech101_all_fea('./data/')
        dims = [48, 40, 254, 512, 928]
        view = 5
        data_size = 9144
        class_num = 102
    elif dataset == 'voc':
        dataset = voc('./data/')
        dims = [512, 399]
        view = 2
        data_size = 5649
        class_num = 20
    elif dataset == 'rgbd':
        dataset = rgbd('./data/')
        dims = [2048, 300]
        view = 2
        data_size = 1449
        class_num = 13
    elif dataset == 'SUNRGBD_fea':
        dataset = SUNRGBD_fea('./data/')
        dims = [4096, 4096]
        view = 2
        data_size = 10335
        class_num = 45
    elif dataset == 'AwA_fea':
        dataset = AwA_fea('./data/')
        dims = [4096, 4096]
        view = 2
        data_size = 10335
        class_num = 45
    # elif dataset == 'cifar10_yuan':
    #     dataset = cifar10_yuan('./data/cifar-10-matlab/cifar-10-batches-mat/')
    #     dims = [3072, 3072, 3072, 3072, 3072]
    #     view = 5
    #     data_size = 50000
    #     class_num = 10
    elif dataset == 'Caltech101-all':
        dataset = Caltech101_all('./data/')
        dims = [48, 40, 254, 1984, 512, 928]
        view = 6
        data_size = 9144
        class_num = 102
    elif dataset == 'bbcsport':
        dataset = bbcsport('./data/')
        dims = [3183, 3203]
        view = 2
        data_size = 544
        class_num = 5
    elif dataset == "Cifar10":
        dataset = cifar_10('./data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 10
    elif dataset == "Cifar100":
        dataset = cifar_100('./data/')
        dims = [512, 2048, 1024]
        view = 3
        data_size = 50000
        class_num = 100
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
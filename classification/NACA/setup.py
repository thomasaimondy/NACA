# -*- coding: utf-8 -*-
import os
import pickle
import subprocess
import sys

import numpy as np
import scipy.io as sio
import scipy.io.wavfile as wav
import torch
from python_speech_features import fbank
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from tqdm import tqdm
import utils


def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        print("=== The available CUDA GPU will be used for computations.")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    utils.device = device

    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    if args.dataset == "MNIST":
        print("=== Loading the MNIST dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_mnist(args, kwargs)
    elif args.dataset == "tidigits":
        print("=== Loading and augmenting the tidigits dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_tidigits(args, kwargs)
    else:
        print("=== ERROR - Unsupported dataset ===")
        sys.exit(1)
    args.regression = (args.dataset == "regression_synth")

    return (device, train_loader, traintest_loader, test_loader)


def get_gpu_memory_usage():
    if sys.platform == "win32":
        curr_dir = os.getcwd()
        nvsmi_dir = r"C:\Program Files\NVIDIA Corporation\NVSMI"
        os.chdir(nvsmi_dir)
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
        os.chdir(curr_dir)
    else:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
    gpu_memory = [int(x) for x in result.decode('utf-8').strip().split('\n')]

    return gpu_memory


def load_dataset_mnist(args, kwargs):
    roots = '../../Dataset/'
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(roots + '/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0, ), (1.0, ))])), batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)
    traintest_loader = torch.utils.data.DataLoader(datasets.MNIST(roots + '/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0, ), (1.0, ))])), batch_size=args.test_batch_size, shuffle=False, drop_last=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST(roots + '/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0, ), (1.0, ))])), batch_size=args.test_batch_size, shuffle=False, drop_last=True, **kwargs)
    args.input_size = 28
    args.input_channels = 1
    args.label_features = 10

    return (train_loader, traintest_loader, test_loader)


def read_data(path, n_bands, n_frames):
    overlap = 0.5
    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.waV') and file[0] != 'O':
                filelist.append(os.path.join(root, file))
    n_samples = len(filelist)

    def keyfunc(x):
        s = x.split('/')
        return (s[-1][0], s[-2], s[-1][1])  # BH/1A_endpt.wav: sort by '1', 'BH', 'A'

    filelist.sort(key=keyfunc)
    feats = np.empty((n_samples, 1, n_bands, n_frames))
    labels = np.empty((n_samples, ), dtype=np.long)
    with tqdm(total=len(filelist)) as pbar:
        for i, file in enumerate(filelist):
            pbar.update(1)
            label = file.split('\\')[-1][0]  # if using windows, change / into \\
            if label == 'Z':
                labels[i] = np.long(0)
            else:
                labels[i] = np.long(label)
            rate, sig = wav.read(file)
            duration = sig.size / rate
            winlen = duration / (n_frames * (1 - overlap) + overlap)
            winstep = winlen * (1 - overlap)
            feat, energy = fbank(sig, rate, winlen, winstep, nfilt=n_bands, nfft=4096, winfunc=np.hamming)
            final_feat = feat[:n_frames]
            final_feat = normalize(final_feat, norm='l1', axis=0)
            feats[i] = np.expand_dims(np.array(final_feat), axis=0)
    np.random.seed(42)
    p = np.random.permutation(n_samples)
    feats, labels = feats[p], labels[p]
    n_train_samples = int(n_samples * 0.7)
    print('n_train_samples:', n_train_samples)
    train_set = (feats[:n_train_samples], labels[:n_train_samples])
    test_set = (feats[n_train_samples:], labels[n_train_samples:])

    return train_set, train_set, test_set


class Tidigits(Dataset):
    def __init__(self, train_or_test, input_channel, n_bands, n_frames, transform=None, target_transform=None):
        super(Tidigits, self).__init__()
        self.n_bands = n_bands
        self.n_frames = n_frames
        roots = '../../Dataset'
        dataname = roots + '/tidigits/packed_tidigits_nbands_' + str(n_bands) + '_nframes_' + str(n_frames) + '.pkl'
        if os.path.exists(dataname):
            with open(dataname, 'rb') as fr:
                [train_set, val_set, test_set] = pickle.load(fr)
        else:
            print('Tidigits Dataset Has not been Processed, now do it.')
            train_set, val_set, test_set = read_data(path=roots + '/tidigits/isolated_digits_tidigits', n_bands=n_bands, n_frames=n_frames)  #(2900, 1640) (2900,)
            with open(dataname, 'wb') as fw:
                pickle.dump([train_set, val_set, test_set], fw)

        if train_or_test == 'train':
            self.x_values = train_set[0]
            self.y_values = train_set[1]
        elif train_or_test == 'test':
            self.x_values = test_set[0]
            self.y_values = test_set[1]
        elif train_or_test == 'valid':
            self.x_values = val_set[0]
            self.y_values = val_set[1]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample = self.x_values[index]
        label = self.y_values[index]
        return sample, label

    def __len__(self):
        return len(self.x_values)


def load_dataset_tidigits(args, kwargs):
    n_bands = 30
    n_frames = 30
    args.input_size = n_bands
    args.input_channels = 1
    args.label_features = 10

    train_dataset = Tidigits('train', args.input_channels, n_bands, n_frames, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0, ), (1.0, ))]))
    traintest_dataset = Tidigits('valid', args.input_channels, n_bands, n_frames, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0, ), (1.0, ))]))
    test_dataset = Tidigits('test', args.input_channels, n_bands, n_frames, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0, ), (1.0, ))]))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    traintest_loader = torch.utils.data.DataLoader(dataset=traintest_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False, drop_last=True)

    return (train_loader, traintest_loader, test_loader)

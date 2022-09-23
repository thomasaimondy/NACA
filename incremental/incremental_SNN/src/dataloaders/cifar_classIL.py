import os, sys
import pickle
import time
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


########################################################################################################################
class Cifar10(Dataset):
    def __init__(self, train_or_test, transform=None, target_transform=None):
        super(Cifar10, self).__init__()
        if os.getlogin() == 'jiashuncheng':
            roots = '/home/jiashuncheng/code/EAST/DATASETS/cifar10.pkl'
        elif os.getlogin() == 'xushuang4':
            roots = '/mnt/lustre/xushuang4/jiashuncheng/code/EAST/classification/others_methods/BRP'
        else:
            roots = '/home/user/jiashuncheng/code/EAST/DATASETS/cifar10.pkl'
        labels = []
        datas = []
        timeStamp = os.stat(roots).st_mtime
        timeArray = time.localtime(timeStamp)
        otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
        print(otherStyleTime)
        with open(roots, 'rb') as f:
            data = pickle.load(f)
        for i in range(len(data)):
            datas.append(data[i][0].cpu().numpy())
            labels.append(data[i][1].cpu().numpy())
        datas = np.array(datas)
        datas = datas.reshape(-1, datas.shape[-1])
        labels = np.array(labels)
        labels = labels.reshape(-1, )
        if train_or_test == 'train':
            self.x_values = datas[:50000]
            self.y_values = labels[:50000]
        elif train_or_test == 'test':
            self.x_values = datas[50000:]
            self.y_values = labels[50000:]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample = self.x_values[index]
        label = self.y_values[index]
        return sample, label

    def __len__(self):
        return len(self.x_values)


def get(seed=0, mini=False, fixed_order=False, pc_valid=0):
    data = {}
    taskcla = []
    size = [3, 32, 32]
    labelsize = 10  #2
    seeds = np.array(list(range(labelsize)), dtype=int)
    np.random.shuffle(seeds)
    print(seeds)

    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    dat = {}

    dat['train'] = datasets.CIFAR10('/home/jiashuncheng/code/CASNN/casnn/DATASETS/CIFAR10/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    dat['test'] = datasets.CIFAR10('/home/jiashuncheng/code/CASNN/casnn/DATASETS/CIFAR10/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    data[0] = {}
    data[0]['name'] = 'cifar10-{}'.format(seeds[0])
    data[0]['ncla'] = 10

    data[1] = {}
    data[1]['name'] = 'cifar10-{}'.format(seeds[1])
    data[1]['ncla'] = 10

    data[2] = {}
    data[2]['name'] = 'cifar10-{}'.format(seeds[2])
    data[2]['ncla'] = 10

    data[3] = {}
    data[3]['name'] = 'cifar10-{}'.format(seeds[3])
    data[3]['ncla'] = 10

    data[4] = {}
    data[4]['name'] = 'cifar10-{}'.format(seeds[4])
    data[4]['ncla'] = 10

    data[5] = {}
    data[5]['name'] = 'cifar10-{}'.format(seeds[5])
    data[5]['ncla'] = 10

    data[6] = {}
    data[6]['name'] = 'cifar10-{}'.format(seeds[6])
    data[6]['ncla'] = 10

    data[7] = {}
    data[7]['name'] = 'cifar10-{}'.format(seeds[7])
    data[7]['ncla'] = 10

    data[8] = {}
    data[8]['name'] = 'cifar10-{}'.format(seeds[8])
    data[8]['ncla'] = 10

    data[9] = {}
    data[9]['name'] = 'cifar10-{}'.format(seeds[9])
    data[9]['ncla'] = 10

    for s in ['train', 'test']:
        loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
        data[0][s] = {'x': [], 'y': []}
        data[1][s] = {'x': [], 'y': []}
        data[2][s] = {'x': [], 'y': []}
        data[3][s] = {'x': [], 'y': []}
        data[4][s] = {'x': [], 'y': []}
        data[5][s] = {'x': [], 'y': []}
        data[6][s] = {'x': [], 'y': []}
        data[7][s] = {'x': [], 'y': []}
        data[8][s] = {'x': [], 'y': []}
        data[9][s] = {'x': [], 'y': []}

        counter = 0
        for image, target in loader:
            label = target.numpy()[0]

            if mini:
                # For fast computation
                if counter > 1000:
                    counter = 0
                    break
                else:
                    counter = counter + 1

            if label == seeds[0]:
                data[0][s]['x'].append(image)
                data[0][s]['y'].append(label)
            elif label == seeds[1]:
                data[1][s]['x'].append(image)
                data[1][s]['y'].append(label)
            elif label == seeds[2]:
                data[2][s]['x'].append(image)
                data[2][s]['y'].append(label)
            elif label == seeds[3]:
                data[3][s]['x'].append(image)
                data[3][s]['y'].append(label)
            elif label == seeds[4]:
                data[4][s]['x'].append(image)
                data[4][s]['y'].append(label)
            elif label == seeds[5]:
                data[5][s]['x'].append(image)
                data[5][s]['y'].append(label)
            elif label == seeds[6]:
                data[6][s]['x'].append(image)
                data[6][s]['y'].append(label)
            elif label == seeds[7]:
                data[7][s]['x'].append(image)
                data[7][s]['y'].append(label)
            elif label == seeds[8]:
                data[8][s]['x'].append(image)
                data[8][s]['y'].append(label)
            elif label == seeds[9]:
                data[9][s]['x'].append(image)
                data[9][s]['y'].append(label)

    # "Unify" and save
    for n in range(10):
        for s in ['train', 'test']:
            data[n][s]['x'] = torch.stack(data[n][s]['x']).view(-1, size[0], size[1], size[2])
            data[n][s]['y'] = torch.LongTensor(np.array(data[n][s]['y'], dtype=int)).view(-1)

    # Validation
    for t in data.keys():
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'].clone()
        data[t]['valid']['y'] = data[t]['train']['y'].clone()

    # Others
    n = 0
    for t in data.keys():
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    return data, taskcla, size, labelsize


########################################################################################################################

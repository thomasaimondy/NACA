import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Alphabet(Dataset):
    def __init__(self, train_or_test, transform=None, target_transform=None):
        super(Alphabet, self).__init__()
        roots = '../../Dataset/alphabet'
        with open(os.path.join(roots, 'train_alphabet.pkl'), 'rb') as f:
            data_train = pickle.load(f)
        with open(os.path.join(roots, 'test_alphabet.pkl'), 'rb') as f:
            data_test = pickle.load(f)

        index = data_train[0].shape[0]
        data = np.r_[data_train[0], data_test[0]]
        data = data / 255
        data = (data - data[:index].mean()) / (data[:index].std())
        data_train_values = data[:index]
        data_test_values = data[index:]

        if train_or_test == 'train':
            self.x_values = data_train_values
            self.y_values = data_train[1]
        elif train_or_test == 'test':
            self.x_values = data_test_values
            self.y_values = data_test[1]
        self.transforms = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample = self.x_values[index]
        label = self.y_values[index]
        return sample, label

    def __len__(self):
        return len(self.x_values)


def get(mini=False, fixed_order=False):
    data = {}
    taskcla = []
    size = [1, 28, 28]
    labsize = 26
    seeds = np.array(list(range(labsize)), dtype=int)
    if not fixed_order:
        np.random.shuffle(seeds)
    print(seeds)

    dat = {}

    dat['train'] = Alphabet('train')
    dat['test'] = Alphabet('test')
    print('load...')
    for i, r in enumerate(seeds):
        data[i] = {}
        data[i]['name'] = 'letters-{:d}'.format(r)
        data[i]['ncla'] = labsize
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
    for s in ['train', 'test']:
        loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
        for image, target in tqdm(loader):
            aux = image.view(-1).numpy()
            image = torch.FloatTensor(aux).view(size)
            index = np.where(target.numpy()[0] == seeds)[0][0]
            # Separate different samples into different tasks
            data[index][s]['x'].append(image)
            data[index][s]['y'].append(target.numpy()[0])
    for i, r in enumerate(seeds):
        for s in ['train', 'test']:
            data[i][s]['x'] = torch.stack(data[i][s]['x']).view(-1, size[0], size[1], size[2])
            data[i][s]['y'] = torch.LongTensor(np.array(data[i][s]['y'], dtype=int)).view(-1)

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

    return data, taskcla, size, labsize

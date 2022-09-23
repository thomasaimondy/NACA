import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
import utils

SEED = utils.args

########################################################################################################################


def get(seed=0, mini=False, fixed_order=False, pc_valid=0):
    data = {}
    taskcla = []
    size = [1, 28, 28]
    labelsize = 10  #2
    seeds = np.array(list(range(labelsize)), dtype=int)
    np.random.shuffle(seeds)

    print(seeds)

    # MNIST
    mean = (0.1307, )
    std = (0.3081, )
    dat = {}
    if os.getlogin() == 'jiashuncheng':
        roots = '/home/jiashuncheng/code/EDF-classification'
    elif os.getlogin() == 'xushuang4':
        roots = '/mnt/lustre/xushuang4/jiashuncheng/code/EAST/incremental/src'
    elif os.getlogin() == 'user':
        roots = '/home/user/jiashuncheng/code/EAST/incremental_ANN/src'
    else:
        roots = '/home/snn/jiashuncheng/Dataset'
    dat['train'] = datasets.MNIST(roots + '/dat/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))
    dat['test'] = datasets.MNIST(roots + '/dat/', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]))

    data[0] = {}
    data[0]['name'] = 'mnist-{}'.format(seeds[0])
    data[0]['ncla'] = 10

    data[1] = {}
    data[1]['name'] = 'mnist-{}'.format(seeds[1])
    data[1]['ncla'] = 10

    data[2] = {}
    data[2]['name'] = 'mnist-{}'.format(seeds[2])
    data[2]['ncla'] = 10

    data[3] = {}
    data[3]['name'] = 'mnist-{}'.format(seeds[3])
    data[3]['ncla'] = 10

    data[4] = {}
    data[4]['name'] = 'mnist-{}'.format(seeds[4])
    data[4]['ncla'] = 10

    data[5] = {}
    data[5]['name'] = 'mnist-{}'.format(seeds[5])
    data[5]['ncla'] = 10

    data[6] = {}
    data[6]['name'] = 'mnist-{}'.format(seeds[6])
    data[6]['ncla'] = 10

    data[7] = {}
    data[7]['name'] = 'mnist-{}'.format(seeds[7])
    data[7]['ncla'] = 10

    data[8] = {}
    data[8]['name'] = 'mnist-{}'.format(seeds[8])
    data[8]['ncla'] = 10

    data[9] = {}
    data[9]['name'] = 'mnist-{}'.format(seeds[9])
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

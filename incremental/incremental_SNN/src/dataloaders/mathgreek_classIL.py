import os, sys
import pickle
import time
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from tqdm import tqdm

# import pandas as pd


########################################################################################################################
class Mathgreek(Dataset):
    def __init__(self, train_or_test, transform=None, target_transform=None):
        super(Mathgreek, self).__init__()
        if os.getlogin() == 'jiashuncheng':
            roots = '/home/jiashuncheng/code/EAST/DATASETS/mathgreek/'
        elif os.getlogin() == 'xushuang4':
            roots = '/mnt/lustre/xushuang4/jiashuncheng/code/EAST/classification/others_methods/BRP'
        else:
            roots = '/home/user/jiashuncheng/code/EAST/DATASETS/mathgreek/'
        with open(roots + 'train_mathgreek_erode4.pkl', 'rb') as f:
            data_train = pickle.load(f)
        with open(roots + 'test_mathgreek_erode4.pkl', 'rb') as f:
            data_test = pickle.load(f)

        index = data_train[0].shape[0]
        data = np.r_[data_train[0], data_test[0]]
        # data = 1 - data
        data = (data - data[:index].mean()) / data[:index].std()
        data_train_values = data[:index]
        data_test_values = data[index:]

        if train_or_test == 'train':
            self.x_values = data_train_values
            self.y_values = data_train[1]
        elif train_or_test == 'test':
            self.x_values = data_test_values
            self.y_values = data_test[1]
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
    # size=[1,32,32]
    size = [1, 45, 45]
    labsize = 46
    seeds = np.array(list(range(labsize)), dtype=int)
    np.random.shuffle(seeds)
    print(seeds)

    mean = (0.1307, )
    std = (0.3081, )
    dat = {}

    dat['train'] = Mathgreek('train', transform=transforms.ToTensor())
    dat['test'] = Mathgreek('test', transform=transforms.ToTensor())
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

    len_train = []
    len_test = []
    for index in range(46):
        len_train.append(len(data[index]['train']['x']))
        len_test.append(len(data[index]['test']['x']))
    for index in range(46):
        data[index]['train']['x'] = data[index]['train']['x'][:500]
        data[index]['train']['y'] = data[index]['train']['y'][:500]
        data[index]['test']['x'] = data[index]['test']['x'][:500]
        data[index]['test']['y'] = data[index]['test']['y'][:500]

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


########################################################################################################################

if __name__ == '__main__':
    folders = os.listdir('C:/Users/Administrator/Downloads/mathgreek/mathgreek')
    data = []
    label = []
    m = 0
    for each in folders:
        currentFolder = 'C:/Users/Administrator/Downloads/mathgreek/mathgreek/' + each
        for i, file in enumerate(os.listdir(currentFolder)):
            im = cv2.imread((os.path.join(currentFolder, file)))
            img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(img, (45, 45))
            img = img = np.array(resized_image)
            img = img.ravel()
            img = img.tolist()
            data.append(img)
            label.append(m)
        m += 1
    df = pd.DataFrame(data)
    df["label"] = label

    df = df.sample(frac=1)
    df.head()
    df_new = df.sample(frac=1)
    X_train, X_test, y_train, y_test = train_test_split(df_new.iloc[:, :-1], df_new.iloc[:, -1], test_size=0.2, shuffle=False)
    X_train = 1 - np.array(X_train) / 255
    X_test = 1 - np.array(X_test) / 255
    y_train = 1 - np.array(y_train) / 255
    y_test = 1 - np.array(y_test) / 255

    with open('train_mathgreek.pkl', 'wb') as f:
        pickle.dump((np.array(X_train), np.array(y_train)), f)
    with open('test_mathgreek.pkl', 'wb') as f:
        pickle.dump((np.array(X_test), np.array(y_test)), f)

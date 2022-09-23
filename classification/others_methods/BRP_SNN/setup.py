# -*- coding: utf-8 -*-
import time
import torch
import torchvision
from torchvision import transforms,datasets
import numpy as np
import os
import sys
import subprocess
#加载tidigits数据集
from python_speech_features import fbank
import numpy as np
import scipy.io.wavfile as wav
from sklearn.preprocessing import normalize
import os
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import scipy.io as sio

class SynthDataset(torch.utils.data.Dataset):

    def __init__(self, select, type):
        self.dataset, self.input_size, self.input_channels, self.label_features = torch.load( './DATASETS/'+select+'/'+type+'.pt')

    def __len__(self):
        return len(self.dataset[1])

    def __getitem__(self, index):
        return self.dataset[0][index], self.dataset[1][index]

def setup(args):
    args.cuda = not args.cpu and torch.cuda.is_available()
    if args.cuda:
        print("=== The available CUDA GPU will be used for computations.")
        #memory_load = get_gpu_memory_usage()
        #cuda_device = np.argmin(memory_load).item()
        #torch.cuda.set_device(cuda_device)
        #device = torch.cuda.current_device()
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')

    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    if args.dataset == "regression_synth":
        print("=== Loading the synthetic regression dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_regression_synth(args, kwargs)
    elif args.dataset == "classification_synth":
        print("=== Loading the synthetic classification dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_classification_synth(args, kwargs)
    elif args.dataset == "MNIST":
        print("=== Loading the MNIST dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_mnist(args, kwargs)
    elif args.dataset == "CIFAR10":
        print("=== Loading the CIFAR-10 dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_cifar10(args, kwargs)
    elif args.dataset == "CIFAR10aug":
        print("=== Loading and augmenting the CIFAR-10 dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_cifar10_augmented(args, kwargs)
    elif args.dataset == "tidigits":
        print("=== Loading and augmenting the tidifits dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_tidigits(args, kwargs)
    elif args.dataset == "CIFAR100":
        print("=== Loading and augmenting the CIFAR100 dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_cifar100(args, kwargs)
    elif args.dataset == "nettalk":
        print("=== Loading and augmenting the NMNIST dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_nettalk(args, kwargs)
    elif args.dataset == "gesture":
        print("=== Loading and augmenting the gesture dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_gesture(args, kwargs)
    elif args.dataset == "FashionMNIST":
        print("=== Loading and augmenting the FashionMNIST dataset...")
        (train_loader, traintest_loader, test_loader) = load_dataset_FashionMNIST(args, kwargs)
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
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
        os.chdir(curr_dir)
    else:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'])
    gpu_memory = [int(x) for x in result.decode('utf-8').strip().split('\n')]
    return gpu_memory

def load_dataset_regression_synth(args, kwargs):

    trainset = SynthDataset("regression","train")
    testset  = SynthDataset("regression", "test")

    train_loader     = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,      shuffle=True , **kwargs)
    traintest_loader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader      = torch.utils.data.DataLoader(testset , batch_size=args.test_batch_size, shuffle=False, **kwargs)

    args.input_size     = trainset.input_size
    args.input_channels = trainset.input_channels
    args.label_features = trainset.label_features

    return (train_loader, traintest_loader, test_loader)

def load_dataset_classification_synth(args, kwargs):

    trainset = SynthDataset("classification","train")
    testset  = SynthDataset("classification", "test")

    train_loader     = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,      shuffle=True , **kwargs)
    traintest_loader = torch.utils.data.DataLoader(trainset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader      = torch.utils.data.DataLoader(testset , batch_size=args.test_batch_size, shuffle=False, **kwargs)

    args.input_size     = trainset.input_size
    args.input_channels = trainset.input_channels
    args.label_features = trainset.label_features

    return (train_loader, traintest_loader, test_loader)

def load_dataset_FashionMNIST(args, kwargs):
    if os.getlogin() == 'jiashuncheng':
        roots = '/home/jiashuncheng/code/EDF-classification'
    elif os.getlogin() == 'xushuang4':
        roots = '/mnt/lustre/xushuang4/jiashuncheng/code/EAST/classification/others_methods/BRP'
    else:
        roots = '/home/user/jiashuncheng/code/EAST'
    train_loader     = torch.utils.data.DataLoader(datasets.FashionMNIST(roots+'/DATASETS/FashionMNIST', train=True,  download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])), batch_size=args.batch_size,      shuffle=True ,drop_last=True, **kwargs)
    traintest_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(roots+'/DATASETS/FashionMNIST', train=True,  download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])), batch_size=args.test_batch_size, shuffle=False,drop_last=True, **kwargs)
    test_loader      = torch.utils.data.DataLoader(datasets.FashionMNIST(roots+'/DATASETS/FashionMNIST', train=False, download=False, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])), batch_size=args.test_batch_size, shuffle=False,drop_last=True, **kwargs)
    args.input_size     = 28
    args.input_channels = 1
    args.label_features = 10

    return (train_loader, traintest_loader, test_loader)

def load_dataset_mnist(args, kwargs):
    if os.getlogin() == 'jiashuncheng':
        roots = '/home/jiashuncheng/code/EDF-classification'
    elif os.getlogin() == 'xushuang4':
        roots = '/mnt/lustre/xushuang4/jiashuncheng/code/EAST/classification/others_methods/BRP'
    else:
        roots = '/home/snn/jiashuncheng/Dataset/DATASETS'
    train_loader     = torch.utils.data.DataLoader(datasets.MNIST(roots+'/DATASETS', train=True,  download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])), batch_size=args.batch_size,      shuffle=True ,drop_last=True, **kwargs)
    traintest_loader = torch.utils.data.DataLoader(datasets.MNIST(roots+'/DATASETS', train=True,  download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])), batch_size=args.test_batch_size, shuffle=False,drop_last=True, **kwargs)
    test_loader      = torch.utils.data.DataLoader(datasets.MNIST(roots+'/DATASETS', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))])), batch_size=args.test_batch_size, shuffle=False,drop_last=True, **kwargs)
    args.input_size     = 28
    args.input_channels = 1
    args.label_features = 10

    return (train_loader, traintest_loader, test_loader)

class Cifar10(Dataset):
    def __init__(self, train_or_test, transform = None, target_transform = None):
        super(Cifar10, self).__init__()
        if os.getlogin() == 'jiashuncheng':
            roots = '/home/jiashuncheng/code/EDF-classification'
        elif os.getlogin() == 'xushuang4':
            roots = '/mnt/lustre/xushuang4/jiashuncheng/code/EAST/classification/others_methods/BRP'
        elif os.getlogin() == 'aa':
            roots = '/home/aa/Desktop/jiashuncheng/Dataset'
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
        labels = labels.reshape(-1,)
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

def load_dataset_cifar10(args, kwargs):
    if args.pre_encoding:
        train_dataset = Cifar10('train', transform=transforms.ToTensor())
        test_dataset = Cifar10('test', transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True,drop_last=True)
        traintest_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True)
        test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False,drop_last=True)

        args.input_size     = 1000
        args.input_channels = 1
        args.label_features = 10

    else:
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        # transform_cifar10 = transforms.Compose([transforms.ToTensor(),normalize,])
        # transform_cifar10 = transforms.Compose([
        #                                 transforms.Resize((224, 224)),
        #                                 transforms.RandomHorizontalFlip(p=0.5),
        #                                 transforms.ToTensor(),
        #                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        transform_cifar10 = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomGrayscale(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_cifar10_test = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if os.getlogin() == 'jiashuncheng':
            roots = '/home/jiashuncheng/code/EDF-classification'
        elif os.getlogin() == 'xushuang4':
            roots = '/mnt/lustre/xushuang4/jiashuncheng/code/EAST/classification/others_methods/BRP'
        else:
            roots = '/home/user/jiashuncheng/code/WeightScale/data'

        train_loader     = torch.utils.data.DataLoader(datasets.CIFAR10(roots, train=True,  download=False, transform=transform_cifar10), batch_size=args.batch_size,      shuffle=True , drop_last=False,**kwargs)
        traintest_loader = torch.utils.data.DataLoader(datasets.CIFAR10(roots, train=True,  download=False, transform=transform_cifar10_test), batch_size=args.test_batch_size, shuffle=False, drop_last=False, **kwargs)
        test_loader      = torch.utils.data.DataLoader(datasets.CIFAR10(roots, train=False, download=False, transform=transform_cifar10_test), batch_size=args.test_batch_size, shuffle=False, drop_last=False, **kwargs)

        args.input_size     = 32
        args.input_channels = 3
        args.label_features = 10

    return (train_loader, traintest_loader, test_loader)

def load_dataset_cifar10_augmented(args, kwargs):
    #Source: https://zhenye-na.github.io/2018/09/28/pytorch-cnn-cifar10.html

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                             std=[x/255.0 for x in [63.0, 62.1, 66.7]]),
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]], std=[x/255.0 for x in [63.0, 62.1, 66.7]]),])

    trainset = torchvision.datasets.CIFAR10('./DATASETS', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    traintestset = torchvision.datasets.CIFAR10('./DATASETS', train=True, download=True, transform=transform_test)
    traintest_loader = torch.utils.data.DataLoader(traintestset, batch_size=args.test_batch_size, shuffle=False)

    testset = torchvision.datasets.CIFAR10('./DATASETS', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)

    args.input_size     = 32
    args.input_channels = 3
    args.label_features = 10

    return (train_loader, traintest_loader, test_loader)

def read_data(path, n_bands, n_frames):
    overlap = 0.5

    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.waV') and file[0] != 'O':
                filelist.append(os.path.join(root, file))
    # filelist = filelist[:1002]

    n_samples = len(filelist)

    def keyfunc(x):
        s = x.split('/')
        return (s[-1][0], s[-2], s[-1][1]) # BH/1A_endpt.wav: sort by '1', 'BH', 'A'
    filelist.sort(key=keyfunc)

    feats = np.empty((n_samples, 1, n_bands, n_frames))
    labels = np.empty((n_samples,), dtype=np.long)
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
            # feat = np.log(feat)
            final_feat = feat[:n_frames]
            final_feat = normalize(final_feat, norm='l1', axis=0)
            feats[i] = np.expand_dims(np.array(final_feat),axis=0)
        # feats[i] = feat[:n_frames].flatten() # feat may have 41 or 42 frames
        # feats[i] = feat.flatten() # feat may have 41 or 42 frames

    # feats = normalize(feats, norm='l2', axis=1)
    # normalization
    # feats = preprocessing.scale(feats)

    np.random.seed(42)
    p = np.random.permutation(n_samples)
    feats, labels = feats[p], labels[p]

    n_train_samples = int(n_samples * 0.7)
    print('n_train_samples:',n_train_samples)

    train_set = (feats[:n_train_samples], labels[:n_train_samples])
    test_set = (feats[n_train_samples:], labels[n_train_samples:])

    return train_set, train_set, test_set

def datatobatch(args,train_loader):
    temp, temp2 = [], []
    label, label2 = [], []
    for i, data in enumerate(train_loader[0]):
        if i % args.batch_size == 0 and i != 0:
            temp2.append(temp)
            label2.append(label)
            temp, label = [], []
            temp.append(data)
            label.append(train_loader[1][i])
        else:
            temp.append(data)
            label.append(train_loader[1][i])
    temp2 = torch.tensor(temp2)
    label2 = torch.tensor(label2)
    a = (temp2, label2)
    return a

class Tidigits(Dataset):
    def __init__(self,train_or_test,input_channel,n_bands,n_frames,transform=None, target_transform = None):
        super(Tidigits, self).__init__()
        self.n_bands = n_bands
        self.n_frames = n_frames

        if os.getlogin() == 'jiashuncheng':
            roots = '/home/jiashuncheng/code/EDF-classification'
        elif os.getlogin() == 'xushuang4':
            roots = '/mnt/lustre/xushuang4/jiashuncheng/code/newdrtp'
        else:
            roots = '/home/snn/jiashuncheng/Dataset'

        dataname = roots+'/DATASETS/tidigits/packed_tidigits_nbands_'+str(n_bands)+'_nframes_' + str(n_frames)+'.pkl'
        if os.path.exists(dataname):
            with open(dataname,'rb') as fr:
                [train_set, val_set, test_set] = pickle.load(fr)
        else:
            print('Tidigits Dataset Has not been Processed, now do it.')
            train_set, val_set, test_set = read_data(path=roots+'/DATASETS/tidigits/isolated_digits_tidigits', n_bands=n_bands, n_frames=n_frames)#(2900, 1640) (2900,)
            with open(dataname,'wb') as fw:
                pickle.dump([train_set, val_set, test_set],fw)
        if train_or_test == 'train':
            self.x_values = train_set[0]
            self.y_values = train_set[1]

        elif train_or_test == 'test':
            self.x_values = test_set[0]
            self.y_values = test_set[1]
        elif train_or_test == 'valid':
            self.x_values = val_set[0]
            self.y_values = val_set[1]
        self.transform =transform
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

    train_dataset = Tidigits('train',args.input_channels, n_bands,n_frames,transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))]))
    traintest_dataset = Tidigits('valid',args.input_channels,n_bands,n_frames,transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))]))
    test_dataset = Tidigits('test',args.input_channels,n_bands,n_frames,transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.0,),(1.0,))]))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,shuffle = True, drop_last = True)
    traintest_loader = torch.utils.data.DataLoader(dataset=traintest_dataset, batch_size=args.test_batch_size,shuffle = False,drop_last = True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size,shuffle = False,drop_last = True)

    return (train_loader, traintest_loader, test_loader)

class Gesture(Dataset):
    def __init__(self, train_or_test, transform = None, target_transform = None):
        super(Gesture, self).__init__()
        rootfile = '/home/user/jiashuncheng/code/EAST'
        # rootfile = '.'
        mat_fname = rootfile+'/DATASETS/DVS_gesture_100.mat'
        mat_contents = sio.loadmat(mat_fname)
        if train_or_test == 'train':
            self.x_values = mat_contents['train_x_100']
            self.y_values = mat_contents['train_y']
        elif train_or_test == 'test':
            self.x_values = mat_contents['test_x_100']
            self.y_values = mat_contents['test_y']
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample = self.x_values[index, :, :]
        sample = torch.reshape(torch.tensor(sample), (sample.shape[0], 32, 32)).unsqueeze(0)
        label = self.y_values[index].astype(np.float32)
        label = torch.topk(torch.tensor(label), 1)[1].squeeze(0)
        return sample, label

    def __len__(self):
        return len(self.x_values)

class Gesture_pre(Dataset):
    def __init__(self, train_or_test, transform = None, target_transform = None):
        super(Gesture_pre, self).__init__()
        if os.getlogin() == 'jiashuncheng':
            roots = '/home/jiashuncheng/code/EDF-classification'
        elif os.getlogin() == 'xushuang4':
            roots = '/mnt/lustre/xushuang4/jiashuncheng/code/EAST/classification/others_methods/BRP'
        else:
            roots = '/home/user/jiashuncheng/code/EAST/DATASETS/dvsgesture_1.pkl'
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
        labels = labels.reshape(-1,)
        if train_or_test == 'train':
            self.x_values = datas[:1176]
            self.y_values = labels[:1176]
        elif train_or_test == 'test':
            self.x_values = datas[1176:]
            self.y_values = labels[1176:]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample = self.x_values[index]
        label = self.y_values[index]
        return sample, label

    def __len__(self):
        return len(self.x_values)

def load_dataset_gesture(args, kwargs):
    if args.pre_encoding:
        train_dataset = Gesture_pre('train', transform=transforms.ToTensor())
        test_dataset = Gesture_pre('test', transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True,drop_last=True)
        traintest_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True)
        test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False,drop_last=True)

        args.input_size     = 2000
        args.input_channels = 1
        args.label_features = 11
    else:
        args.input_size     = 32*32 
        args.input_channels = 1
        args.label_features = 11

        train_dataset = Gesture('train', transform=transforms.ToTensor())
        test_dataset = Gesture('test', transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size = args.batch_size,shuffle = True,drop_last=True)
        traintest_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = args.batch_size, shuffle=False,drop_last=True)
        test_loader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size = args.batch_size,shuffle = False,drop_last=True)

    return (train_loader, traintest_loader, test_loader)

class Nettalk(Dataset):
    def __init__(self, train_or_test, transform = None, target_transform = None):
        super(Nettalk, self).__init__()
        mat_fname = 'DATASETS/nettalk_small.mat'
        mat_contents = sio.loadmat(mat_fname)
        if train_or_test == 'train':
            self.x_values = mat_contents['train_x']
            self.y_values = mat_contents['train_y']
        elif train_or_test == 'test':
            self.x_values = mat_contents['test_x']
            self.y_values = mat_contents['test_y']
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        sample = self.x_values[index]
        label = self.y_values[index]
        return sample, label

    def __len__(self):
        return len(self.x_values)

def load_dataset_nettalk(args, **kwargs):
    args.input_size = 189
    args.input_channels = 1
    args.label_features = 26

    train_dataset = Nettalk('train', transform=transforms.ToTensor())
    test_dataset = Nettalk('test', transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True,drop_last=True)
    traintest_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False,drop_last=True)

    return (train_loader, traintest_loader, test_loader)

class Cifar100(Dataset):
    def __init__(self, train_or_test, transform = None, target_transform = None):
        super(Cifar100, self).__init__()
        if os.getlogin() == 'jiashuncheng':
            roots = '/home/jiashuncheng/code/EDF-classification'
        elif os.getlogin() == 'xushuang4':
            roots = '/mnt/lustre/xushuang4/jiashuncheng/code/EAST/classification/others_methods/BRP'
        else:
            roots = '/home/user/jiashuncheng/code/EAST/DATASETS/cifar100.pkl'
        print(roots)
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
        labels = labels.reshape(-1,)
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

def load_dataset_cifar100(args, kwargs):
    if args.pre_encoding:
        train_dataset = Cifar100('train', transform=transforms.ToTensor())
        test_dataset = Cifar100('test', transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True,drop_last=True)
        traintest_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False,drop_last=True)
        test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = args.batch_size, shuffle = False,drop_last=True)

        args.input_size     = 1000
        args.input_channels = 1
        args.label_features = 100
        
    else:
        transform_cifar100 = transforms.Compose([transforms.ToTensor(),])

        if os.getlogin() == 'jiashuncheng':
            roots = '/home/jiashuncheng/code/EDF-classification'
        elif os.getlogin() == 'xushuang4':
            roots = '/mnt/lustre/xushuang4/jiashuncheng/code/EAST/classification/others_methods/BRP'
        else:
            roots = '/home/user/jiashuncheng/code/EAST'

        train_loader     = torch.utils.data.DataLoader(datasets.CIFAR100(roots+'/DATASETS', train=True,  download=False, transform=transform_cifar100), batch_size=args.batch_size,      shuffle=True , drop_last=True, **kwargs)
        traintest_loader = torch.utils.data.DataLoader(datasets.CIFAR100(roots+'/DATASETS', train=True,  download=False, transform=transform_cifar100), batch_size=args.test_batch_size, shuffle=False, drop_last=True, **kwargs)
        test_loader      = torch.utils.data.DataLoader(datasets.CIFAR100(roots+'/DATASETS', train=False, download=False, transform=transform_cifar100), batch_size=args.test_batch_size, shuffle=False, drop_last=True, **kwargs)

        args.input_size     = 32
        args.input_channels = 3
        args.label_features = 100

    return (train_loader, traintest_loader, test_loader)

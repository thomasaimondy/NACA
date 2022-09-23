import pickle
import pprint
import numpy as np
import setup
'''
file=open("./packed_tidigits_nbands_20_nframes_20.pkl","rb")
[train_set, val_set, test_set] = pickle.load(file)
x1=train_set[0]
x2=test_set[0]
'''
(train_loader, traintest_loader, test_loader)=setup.load_dataset_tidigits()
for batch_idx, (data, label) in enumerate(train_loader):
    print('batch_index:',batch_idx+1)
    print('data:',data.shape)
    print('label:',label.shape)
    

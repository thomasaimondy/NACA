# -*- coding: utf-8 -*-
import argparse
import train
import setup
import os
import torch
import numpy as np
import time
import utils


def mkd(args):
    try:
        os.makedirs('output/' + args.codename)
    except FileExistsError:
        print('file already exists')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--dataset', type=str, choices=['regression_synth', 'classification_synth', 'MNIST', 'CIFAR10', 'CIFAR10aug', 'tidigits', 'CIFAR100', 'nettalk', 'gesture', 'FashionMNIST'], default='MNIST')
    parser.add_argument('--train-mode', choices=['BP', 'FA', 'DFA', 'DRTP', 'sDFA', 'shallow', 'BRP'], default='DRTP')
    parser.add_argument('--optimizer', choices=['SGD', 'NAG', 'Adam', 'RMSprop'], default='Adam')
    parser.add_argument('--loss', choices=['MSE', 'BCE', 'CE'], default='MSE')
    parser.add_argument('--freeze-conv-layers', action='store_true', default=False)
    parser.add_argument('--fc-zero-init', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--test-batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--topology', type=str, default='FC_1000_FC_10')
    parser.add_argument('--spike-window', type=int, default=100)
    parser.add_argument('--conv-act', type=str, choices={'tanh', 'sigmoid', 'relu'}, default='tanh')
    parser.add_argument('--hidden-act', type=str, choices={'tanh', 'sigmoid', 'relu'}, default='tanh')
    parser.add_argument('--output-act', type=str, choices={'sigmoid', 'tanh', 'none', 'relu'}, default='sigmoid')
    parser.add_argument('--thresh', type=float, default=0.5)
    parser.add_argument('--randKill', type=float, default=1)
    parser.add_argument('--lens', type=float, default=0.5)
    parser.add_argument('--decay', type=float, default=0.2)
    parser.add_argument('--codename', type=str, default='tidigits1-1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--brpscale', type=float, default=5e-3)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--pool', type=str, choices=['Avg', 'Max'], default='Avg')
    parser.add_argument('--quickexit', action='store_true')
    parser.add_argument('--pre-encoding', action='store_true')
    timeclock = time.strftime("%Y%m%d%Hh%Mm%Ss", time.localtime())

    args = parser.parse_args()
    utils.args = args
    args.codename = timeclock + '_' + args.codename + '_' + str(args.seed) + '_' + str(args.train_mode)
    if args.train_mode == 'BRP':
        args.codename += str(args.brpscale)
    mkd(args)
    filepath = 'output/' + args.codename
    file = open(filepath + '/para.txt', 'w')
    file.write('pid:' + str(os.getpid()) + '\n')
    file.write(str(vars(args)).replace(',', '\n'))
    file.close()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    (device, train_loader, traintest_loader, test_loader) = setup.setup(args)
    train.train(args, device, train_loader, traintest_loader, test_loader)


if __name__ == '__main__':
    main()

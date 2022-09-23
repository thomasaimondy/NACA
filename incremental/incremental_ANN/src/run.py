##########################################################################################
#### Code for Incremental learning of ANNs using NACA
#### Multiple benchmark datasets and ANNs using different training algorithms (EWC, SGD, Joint, ...)
#### 2022-09-06， CASIA, Tielin Zhang et al.
#### Based on architectures of HAT [Serrà, 2018, ICML]
#########################################################################################

import sys, os, argparse, time
import numpy as np
import torch
import utils

tstart = time.time()
import time

# Arguments
parser = argparse.ArgumentParser(description='')
# common parameters for all methods
parser.add_argument('--seed', type=int, default=1, help='(default=%(default)d)')  #(0,1,4,100,1300)
parser.add_argument('--mini', action='store_true', help='the mini dataset')
parser.add_argument('--experiment', default='mnist_classIL', type=str, required=False, choices=['mnist_classIL', 'cifar_classIL', 'gesture_classIL', 'alphabet_classIL', 'mathgreek_classIL'], help='(default=%(default)s)')
parser.add_argument('--approach', default='naca', type=str, required=False, choices=['sgd', 'ewc', 'naca'], help='(default=%(default)s)')
parser.add_argument('--output', default='', type=str, required=False, help='(default=%(default)s)')
parser.add_argument('--nepochs', default=100, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--lr', default=5e-5, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--lr_factor', default=1, type=float, required=False, help='(default=%(default)f)')
parser.add_argument('--parameter', type=str, default='', help='(default=%(default)s)')
parser.add_argument('--gpu', type=str, default='0', help='(default=%(default)s)')
parser.add_argument('--multi_output', action='store_true', default=False, help='the type of ouput layer')
parser.add_argument('--nhid', type=int, default=1000, help='(default=%(default)d)')
parser.add_argument('--sbatch', type=int, default=256, help='(default=%(default)d)')
parser.add_argument('--nlayers', type=int, default=1, help='(default=%(default)d)')
parser.add_argument('--fixed_order', action='store_true')
# parameters for naca
parser.add_argument('--bias', type=float, default=None)
parser.add_argument('--delta_bias', type=float, default=0.15)
parser.add_argument('--lambda_inv', type=int, default=0.5)
parser.add_argument('--theta_max', type=int, default=1.2)
parser.add_argument('--distribution', type=str, default='uniform', required=False, choices=['uniform', 'normal', 'beta'], help='(default=%(default)s)')
args = parser.parse_args()

if args.delta_bias is not None:
    args.bias = None
utils.args = args
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

timeclock = time.strftime("%Y%m%d%H%M%S", time.localtime())
rootpath = '../res/' + args.experiment + '_' + args.approach
if args.output == '':
    if args.multi_output:
        args.output = rootpath + '/' + timeclock + '_MultiHead'
    else:
        args.output = rootpath + '/' + timeclock + '_SingleHead'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
else:
    rootpath = rootpath + '_' + args.output
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    args.output = rootpath + '/' + timeclock + '_nhid' + str(args.nhid) + '_nalyers' + str(args.nlayers) + str(args.nlayers) + '_alpha' + str(args.bias) + '_delta_alpha' + str(args.delta_bias)
print('=' * 100)
print('Arguments =')
for arg in vars(args):
    print('\t' + arg + ':', getattr(args, arg))
print('=' * 100)
f = open(args.output + '_configure_seed_{}.txt'.format(args.seed), 'w+')
f.write('pid:' + str(os.getpid()) + '\n')
f.write(str(vars(args)).replace(',', '\n'))
f.close()

# Seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
else:
    print('[CUDA unavailable]')
    sys.exit()

# Args -- Experiment
if args.experiment == 'mnist_classIL':
    from dataloaders import mnist_classIL as dataloader
elif args.experiment == 'cifar_classIL':
    from dataloaders import cifar_classIL as dataloader
elif args.experiment == 'gesture_classIL':
    from dataloaders import gesture_classIL as dataloader
elif args.experiment == 'alphabet_classIL':
    from dataloaders import alphabet_classIL as dataloader
elif args.experiment == 'mathgreek_classIL':
    from dataloaders import mathgreek_classIL as dataloader

# Args -- Approachs -- Networks
if args.approach == 'sgd':
    from approaches import sgd as approach
    from networks import mlp as network
elif args.approach == 'ewc':
    from approaches import ewc as approach
    from networks import mlp as network
elif args.approach == 'naca':
    from approaches import naca as approach
    from networks import mlp_naca as network

# Load
print('Load data...')
data, taskcla, inputsize, labsize = dataloader.get(mini=args.mini, fixed_order=args.fixed_order)
print('Input size =', inputsize, '\nTask info =', taskcla)

# Inits
print('Inits...')
if args.approach == 'naca':
    net = network.Net(args, inputsize, taskcla, labsize, nhid=args.nhid, nlayers=args.nlayers).cuda()
    appr = approach.Appr(net, labsize, nepochs=args.nepochs, lr=args.lr, lr_factor=args.lr_factor, args=args, sbatch=args.sbatch)
else:
    net = network.Net(args, inputsize, taskcla, nhid=args.nhid, nlayers=args.nlayers).cuda()
    appr = approach.Appr(net, nepochs=args.nepochs, lr=args.lr, args=args, sbatch=args.sbatch, lr_factor=args.lr_factor)

print(appr.criterion)
utils.print_optimizer_config(appr.optimizer)
print('-' * 100)

f = open(args.output + '_configure_seed_{}.txt'.format(args.seed), 'a+')
f.write('\n\n' + str(net) + '\n')
f.close()

# Loop tasks
acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)

print('Inits...')
if args.approach in ['naca', 'nacasnn']:
    net = network.Net(args, inputsize, taskcla, labsize, nhid=args.nhid, nlayers=args.nlayers).cuda()
else:
    net = network.Net(args, inputsize, taskcla, nhid=args.nhid, nlayers=args.nlayers).cuda()

if args.approach in ['naca', 'nacasnn']:
    appr = approach.Appr(net, labsize, nepochs=args.nepochs, lr=args.lr, lr_factor=args.lr_factor, args=args, sbatch=args.sbatch)
else:
    appr = approach.Appr(net, nepochs=args.nepochs, lr=args.lr, args=args, sbatch=args.sbatch, lr_factor=args.lr_factor)

for t, ncla in taskcla:
    print('*' * 100)
    print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    xtrain = data[t]['train']['x'].cuda()
    ytrain = data[t]['train']['y'].cuda()
    print(min(ytrain), max(ytrain))
    xvalid = data[t]['valid']['x'].cuda()
    yvalid = data[t]['valid']['y'].cuda()
    task = t

    appr.train(task, xtrain, ytrain, xvalid, yvalid)

    print('-' * 100)

    # Test
    utils.train_mode = 'test'
    for u in range(t + 1):
        utils.u = u
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()
        if args.approach in ['naca', 'nacasnn']:
            ytest = torch.zeros(ytest.shape[0], labsize).cuda().scatter_(1, ytest.unsqueeze(1).long(), 1.0).cuda()
        test_loss, test_acc = appr.eval(u, xtest, ytest)
        print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u, data[u]['name'], test_loss, 100 * test_acc))
        acc[t, u] = test_acc
        lss[t, u] = test_loss
    cost = [np.array(utils.epoch) + 1, (np.array(utils.epoch) + 1).mean(), (np.array(utils.epoch) + 1).std()]
    print(cost)

    # Save
    print('Save at ' + args.output)
    np.savetxt(args.output + '_acc_seed_{}.txt'.format(args.seed), acc, '%.4f')
    file = open(args.output + '_cost_seed_{}.txt'.format(args.seed), 'w')
    file.write(str(cost))
    file.close()

print('Done!')

for i in range(acc.shape[0]):
    print('{:5.1f}% '.format(100 * acc[i, :i + 1].mean()), end='')

print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))

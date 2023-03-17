import sys, os, argparse, time
import numpy as np
import torch
import utils
import time
sys.path.append('..')

tstart = time.time()
# Arguments
parser = argparse.ArgumentParser(description='')
# Common parameters for all methods
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--mini', action='store_true', help='Use the mini dataset')
parser.add_argument('--experiment', default='mnist_classIL', type=str, required=False, choices=['mnist_classIL', 'cifar_classIL', 'gesture_classIL', 'alphabet_classIL', 'mathgreek_classIL'])
parser.add_argument('--approach', default='nacasnn', type=str, required=False, choices=['sgd', 'ewc', 'naca', 'sgdsnn', 'ewcsnn', 'nacasnn'])
parser.add_argument('--output', default='', type=str, required=False)
parser.add_argument('--nepochs', default=100, type=int, required=False)
parser.add_argument('--lr', default=5e-4, type=float, required=False) # 5e-4 is the best parameters for nacasnn in MNIST dataset
parser.add_argument('--lr_factor', default=1, type=float, required=False)
parser.add_argument('--parameter', type=str, default='')
parser.add_argument('--gpu', type=str, default='0', help='Number of used gpu')
parser.add_argument('--multi_output', action='store_true', default=False, help='the type of ouput layer')
parser.add_argument('--nhid', type=int, default=1000)
parser.add_argument('--sbatch', type=int, default=256)
parser.add_argument('--nlayers', type=int, default=1)
parser.add_argument('--fixed_order', action='store_true', help='The labels are sorted in increasing order')
# Parameters for SNNs
parser.add_argument('--thresh', type=float, default=0.5, help='Threshold of lif neuron')
parser.add_argument('--lens', type=float, default=0.2, help='V_window in pseudo-BP')
parser.add_argument('--decay', type=float, default=0.2, help='Decay time constant for lif neuron')
parser.add_argument('--spike_windows', type=int, default=20)
# Parameters for ANNs
parser.add_argument('--bias', type=float, default=None, help='Between the maximum and minimum input value')
parser.add_argument('--delta_bias', type=float, default=0.2, help='Avoid the zero during training') # 0.2 is the best parameter for naca in MNIST dataset
parser.add_argument('--lambda_inv', type=int, default=0.5)
parser.add_argument('--theta_max', type=int, default=1.2)
parser.add_argument('--distribution', type=str, default='uniform', required=False, choices=['uniform', 'normal', 'beta'])
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
    args.output = rootpath + '/' + timeclock + '_nhid' + str(args.nhid) + '_nalyers' + str(args.nlayers) + str(args.nlayers) + '_bias' + str(args.bias) + '_delta_bias' + str(args.delta_bias)
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
elif args.approach == 'sgdsnn':
    from approaches import sgd as approach
    from networks import mlp_snn as network
elif args.approach == 'ewcsnn':
    from approaches import ewc as approach
    from networks import mlp_snn as network
elif args.approach == 'nacasnn':
    from approaches import naca as approach
    from networks import mlp_naca_snn as network

# Load
print('Load data...')
data, taskcla, inputsize, labsize = dataloader.get(mini=args.mini, fixed_order=args.fixed_order)
print('Input size =', inputsize, '\nTask info =', taskcla)

# Inits
print('Inits...')
net = network.Net(args, inputsize, taskcla, labsize, nhid=args.nhid, nlayers=args.nlayers).cuda()
appr = approach.Appr(net, labsize, nepochs=args.nepochs, lr=args.lr, lr_factor=args.lr_factor, args=args, sbatch=args.sbatch)

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

for t, ncla in taskcla:
    print('*' * 100)
    print('Task {:2d} ({:s})'.format(t, data[t]['name']))
    print('*' * 100)

    # Get data
    xtrain = data[t]['train']['x'].cuda()
    ytrain = data[t]['train']['y'].cuda()
    xvalid = data[t]['valid']['x'].cuda()
    yvalid = data[t]['valid']['y'].cuda()

    # Train
    appr.train(t, xtrain, ytrain, xvalid, yvalid)

    print('-' * 100)

    # Test
    for u in range(t + 1):
        xtest = data[u]['test']['x'].cuda()
        ytest = data[u]['test']['y'].cuda()
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

import sys
import os

import torch

sys.path.append("..")
import utils

os.environ["CUDA_VISIBLE_DEVICES"] = utils.args.gpu
spike_args = {}
spike_args['thresh'] = utils.args.thresh
spike_args['lens'] = utils.args.lens
spike_args['decay'] = utils.args.decay


class Net(torch.nn.Module):
    def __init__(self, args, inputsize, taskcla, nhid, nlayers=1):
        super(Net, self).__init__()

        ncha, size, size2 = inputsize
        self.taskcla = taskcla
        self.args = args
        self.nlayers = nlayers
        self.spike_window = args.spike_windows

        self.fc1 = SpikeLinear(self.args, ncha * size * size2, nhid, bias=False)
        if self.nlayers > 1:
            self.fc2 = SpikeLinear(self.args, nhid, nhid, bias=False)
        if self.nlayers > 2:
            self.fc3 = SpikeLinear(self.args, nhid, nhid, bias=False)

        self.last = torch.nn.ModuleList()
        for t, n in self.taskcla:
            if args.multi_output:
                self.last.append(SpikeLinear(nhid, n))
            else:
                self.last = SpikeLinear(self.args, nhid, n)
        return

    def forward(self, x):
        # h = self.maxpool(self.drop(self.relu(self.conv1(x))))
        input_ = x.view(x.size(0), -1)

        for step in range(self.spike_window):
            h = x.reshape(x.size(0), -1)
            rand = torch.empty(h.size()).to(x.device)
            rand = rand.uniform_(h.min(), h.max())
            h = (h > rand.float().to(x.device))

            h = self.fc1(h.float())
            if self.nlayers > 1:
                h = self.fc2(h)
            if self.nlayers > 2:
                h = self.fc3(h)
            self.last(h)
        # output encoding

        if self.args.multi_output:
            y = self.last[t].sumspike / self.spike_window
        else:
            y = self.last.sumspike / self.spike_window

        return y


class SpikeLinear(torch.nn.Module):
    def __init__(self, args, in_features, out_features, bias=True):
        super(SpikeLinear, self).__init__()
        self.args = args
        self.fc = torch.nn.Linear(in_features, out_features, bias=bias)
        self.bn = torch.nn.BatchNorm1d(in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.mem = torch.zeros((args.sbatch, self.out_features)).to(self.fc.weight.device)
        self.spike = torch.zeros((args.sbatch, self.out_features)).to(self.fc.weight.device)
        self.sumspike = torch.zeros((args.sbatch, self.out_features)).to(self.fc.weight.device)
        self.time_counter = 0
        self.spike_window = args.spike_windows

        # initialization with zero y
        self.y_old = 0

    # t: current task, x: input, y: output, e: epochs
    def forward(self, x):
        if self.time_counter == 0:
            batchsize = x.shape[0]
            self.mem = torch.zeros((batchsize, self.out_features)).to(self.fc.weight.device)
            self.spike = torch.zeros((batchsize, self.out_features)).to(self.fc.weight.device)
            self.sumspike = torch.zeros((batchsize, self.out_features)).to(self.fc.weight.device)

        self.time_counter += 1
        self.mem, self.spike = mem_update(self.fc, self.bn, x, self.mem, self.spike)
        self.sumspike += self.spike
        if self.time_counter == self.spike_window:
            self.time_counter = 0

        return self.spike


# Approximate BP
class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(spike_args['thresh']).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - spike_args['thresh']) < spike_args['lens']
        return grad_input * temp.float()


act_fun = ActFun.apply


# Membrane potential
def mem_update(ops, batchnorm, x, mem, spike, lateral=None):
    a = torch.sigmoid(ops(x))
    mem = mem * spike_args['decay'] * (1. - spike) + a
    if lateral:
        mem += lateral(spike)
    spike = act_fun(mem)
    return mem, spike

import sys

import numpy as np
import torch
import torch.nn as nn
from numpy import prod
from torch.distributions.beta import Beta

sys.path.append("..")
import torch.nn.functional as F
import utils

spike_args = {}
spike_args['thresh'] = utils.args.thresh
spike_args['lens'] = utils.args.lens
spike_args['decay'] = utils.args.decay


class Net(torch.nn.Module):
    def __init__(self, args, inputsize, taskcla, nlab, nhid=40, nlayers=3):
        super(Net, self).__init__()

        self.spike_window = args.spike_windows
        self.args = args
        ncha, size, size2 = inputsize
        self.taskcla = taskcla
        self.labsize = nlab
        self.layers = nn.ModuleList()
        self.nlayers = nlayers

        self.fcs = nn.ModuleList()

        # hidden layer
        for i in range(nlayers):
            if i == 0:
                fc = SpikeLinear(self.args, size * size2 * ncha, nhid, nlab, layer=i)
            else:
                fc = SpikeLinear(self.args, nhid, nhid, nlab, layer=i)
            self.fcs.append(fc)

        if not args.multi_output:
            self.last = SpikeLinear(self.args, nhid, nlab, nlab, layer=-1)
        else:
            self.last = torch.nn.ModuleList()
            for t, n in self.taskcla:
                self.last.append(SpikeLinear(self.args, nhid, n, nlab, layer=-1))

        return

    def forward(self, t, x, laby, e=-1):
        input_ = x.reshape(x.size(0), -1).clone()
        for step in range(self.spike_window):
            u = x.reshape(x.size(0), -1)
            rand = torch.empty(u.size(0), u.size(1)).cuda()
            rand = rand.uniform_(u.min(), u.max())
            u = (u > rand.float().cuda()).float()

            for li in range(len(self.fcs)):
                u = self.fcs[li](u, laby, input_)
                u = u.detach()
            # output
            if self.args.multi_output:
                self.last[t](u, laby, input_)
            else:
                self.last(u, laby, input_)

        # output encoding
        if self.args.multi_output:
            y = self.last[t].sumspike / self.spike_window
        else:
            y = self.last.sumspike / self.spike_window

        hidden_out = u
        return y, hidden_out


class SpikeLinear(torch.nn.Module):
    def __init__(self, args, in_features, out_features, nlab, layer=None):
        super(SpikeLinear, self).__init__()
        self.args = args
        if layer != -1:
            self.fc = torch.nn.Linear(in_features, out_features, bias=False)
            self.NI = torch.empty(2 * nlab, out_features).cuda()
            self.NI = reset_weights_NI(self.NI)
            self.NI.requires_grad = False
        else:
            self.fc = torch.nn.Linear(in_features, out_features, bias=True)
        self.in_features = in_features
        self.out_features = out_features
        self.nlab = nlab
        self.mem = None
        self.spike = None
        self.sumspike = None
        self.time_counter = 0
        self.layer = layer
        self.spike_window = args.spike_windows
        self.old_spike = None

    # t: current task, x: input, y: output
    def forward(self, x, y, input_):
        if self.time_counter == 0:
            batchsize = x.shape[0]
            self.mem = torch.zeros((batchsize, self.out_features)).cuda()
            self.spike = torch.zeros((batchsize, self.out_features)).cuda()
            self.sumspike = torch.zeros((batchsize, self.out_features)).cuda()
            self.old_spike = torch.zeros((batchsize, self.out_features)).cuda()
        self.time_counter += 1
        self.mem, self.spike = mem_update(self.fc, x, self.mem, self.spike, self.old_spike)
        if self.layer != -1:
            u_mask = torch.mean(input_, 0, False)
            u_mask = F.interpolate(u_mask.unsqueeze(0).unsqueeze(0), size=[self.out_features])
            u_mask = u_mask.squeeze(0)
            if utils.bias[utils.args.experiment] is not None:
                bias = utils.bias[utils.args.experiment]  # bias is the threshold, u_mask may be zeros
            else:
                bias = u_mask.max() - utils.delta_bias[utils.args.experiment]  # ensure the u_mask is not zeros
            u_mask = torch.sigmoid(1000 * (u_mask - bias))
            self.spike = self.spike * u_mask.expand_as(self.spike)
        self.sumspike += self.spike

        self.old_spike = self.spike.clone().detach()

        # y=None for inference
        if self.time_counter == self.spike_window:
            self.time_counter = 0
            if y is not None:
                # Hidden layers
                if self.layer != -1:
                    neuromodulator_level = expectation(y).mm(self.NI.view(-1, prod(self.NI.shape[1:]))).view(self.spike.shape)
                    self.spike.backward(gradient=local_modulation(neuromodulator_level), retain_graph=True)

                # Output layers
                else:
                    # MSE
                    err = (self.sumspike / self.spike_window) - y
                    err = torch.matmul(err, torch.eye(err.shape[1]).to(err.device))
                    self.spike.backward(gradient=err, retain_graph=True)

        return self.spike


def expectation(labels):
    sigma = 1
    delta_mu = 1
    max_len = labels.shape[1]
    a = np.array([np.sqrt(2) * np.sqrt(np.log(max_len / i)) * sigma for i in range(1, max_len + 1)])
    a = a / a.max() * (2 * (max_len - delta_mu))
    b = delta_mu + a
    a = torch.tensor(a.astype('int')).to(labels.device)
    assert len(set(a.cpu().numpy().tolist())) == len(a.cpu().numpy().tolist()), 'error in expectation'
    b = torch.tensor(b.astype('int')).to(labels.device)
    Ea = a[torch.max(labels, 1)[1].cpu()]
    Eb = b[torch.max(labels, 1)[1].cpu()]
    Ea = torch.zeros(labels.shape[0], 2 * labels.shape[1], device=labels.device).scatter_(1, Ea.unsqueeze(1).long(), 1.0)
    Eb = torch.zeros(labels.shape[0], 2 * labels.shape[1], device=labels.device).scatter_(1, Eb.unsqueeze(1).long(), 1.0)
    return (Ea + Eb) / 2


def reset_weights_NI(NI):
    if utils.args.distribution == 'uniform':
        torch.nn.init.uniform_(NI)
    elif utils.args.distribution == 'normal':
        torch.nn.init.normal_(NI, mean=0.5, std=1)
        NI.clamp_(0, 1)
    elif utils.args.distribution == 'beta':
        dist = Beta(torch.ones_like(NI) * 0.5, torch.ones_like(NI) * 0.5)
        NI.data = dist.sample()
    return NI


def local_modulation(neuromodulator_level):
    lambda_inv = utils.args.lambda_inv
    theta_max = utils.args.theta_max
    with torch.no_grad():
        nl_ = neuromodulator_level.clone().detach()
        modulation = torch.zeros_like(neuromodulator_level).cuda()
        phase_one = theta_max - (theta_max - 1) * (4 * nl_ - lambda_inv).pow(2) / lambda_inv**2
        phase_two = 4 * (nl_ - lambda_inv).pow(2) / lambda_inv**2
        phase_three = -4 * ((2 * lambda_inv - nl_) - lambda_inv).pow(2) / lambda_inv**2
        phase_four = (theta_max - 1) * (4 * (2 * lambda_inv - nl_) - lambda_inv).pow(2) / lambda_inv**2 - theta_max

        modulation[neuromodulator_level <= 0.5 * lambda_inv] = phase_one[neuromodulator_level <= 0.5 * lambda_inv]
        modulation[(0.5 * lambda_inv < neuromodulator_level) & (neuromodulator_level <= lambda_inv)] = phase_two[(0.5 * lambda_inv < neuromodulator_level) & (neuromodulator_level <= lambda_inv)]
        modulation[(lambda_inv < neuromodulator_level) & (neuromodulator_level <= 1.5 * lambda_inv)] = phase_three[(lambda_inv < neuromodulator_level) & (neuromodulator_level <= 1.5 * lambda_inv)]
        modulation[1.5 * lambda_inv < neuromodulator_level] = phase_four[1.5 * lambda_inv < neuromodulator_level]

    return modulation


# simplified STDP
class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, old_spike):
        ctx.save_for_backward(input, old_spike)
        result = input.gt(spike_args['thresh']).float()
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, old_spike = ctx.saved_tensors
        grad_input = grad_output.clone()
        new_spike = input > spike_args['thresh'] - spike_args['lens']
        delta_spike = new_spike.float() - old_spike

        return grad_input * delta_spike, None, None


act_fun = ActFun.apply


# Membrane potential (non-linear sigmoid before the neuron for better performance)
def mem_update(ops, x, mem, spike, old_spike, drop=None, lateral=None):
    if drop is None:
        mem = mem * spike_args['decay'] * (1. - spike) + torch.sigmoid(ops(x))
    else:
        mem = mem * spike_args['decay'] * (1. - spike) + torch.sigmoid(drop(ops(x)))

    if lateral:
        mem += lateral(spike)
    spike = act_fun(mem, old_spike)
    return mem, spike

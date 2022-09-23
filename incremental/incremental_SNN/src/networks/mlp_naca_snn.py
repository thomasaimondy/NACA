import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy import prod

sys.path.append("..")
import torch.nn.functional as F
import utils

spike_args = {}
spike_args['thresh'] = utils.args.thresh
spike_args['lens'] = utils.args.lens
spike_args['decay'] = utils.args.decay


class Net(torch.nn.Module):
    def __init__(self, args, inputsize, taskcla, nlab, nlayers=3, nhid=40, pdrop1=0, pdrop2=0, spike_windows=2):
        super(Net, self).__init__()

        self.spike_window = spike_windows
        self.args = args
        ncha, size, size2 = inputsize
        self.taskcla = taskcla
        self.labsize = nlab
        self.layers = nn.ModuleList()
        self.nlayers = nlayers

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.drop1 = torch.nn.Dropout(pdrop1)
        self.drop2 = torch.nn.Dropout(pdrop2)

        self.fcs = nn.ModuleList()
        self.efcs = nn.ModuleList()

        # hidden layer
        for i in range(nlayers):
            if i == 0:
                fc = SpikeLinear(self.args, size * size2 * ncha, nhid, nlab, layer=i)
            else:
                fc = SpikeLinear(self.args, nhid, nhid, nlab, layer=i)
            self.fcs.append(fc)

        if not args.multi_output:
            self.last = SpikeLinear(self.args, nhid, nlab, nlab, layer=-1)  # 单头
        else:
            self.last = torch.nn.ModuleList()
            for t, n in self.taskcla:
                self.last.append(SpikeLinear(self.args, nhid, n, nlab, layer=-1))  # 多头

        self.gate = torch.nn.Sigmoid()

        return

    def forward(self, t, x, laby, e=-1):
        input_ = x.reshape(x.size(0), -1).clone()
        for step in range(self.spike_window):
            h = x.reshape(x.size(0), -1)
            rand = torch.empty(h.size(0), h.size(1)).cuda()
            rand = rand.uniform_(h.min(), h.max())
            h = (h > rand.float().cuda()).float()

            for li in range(len(self.fcs)):
                h = self.fcs[li](h, laby, t, input_)
                h = h.detach()
            # output
            if self.args.multi_output:
                self.last[t](h, laby, t, input_)
            else:
                self.last(h, laby, t, input_)

        # output encoding
        if self.args.multi_output:
            y = self.last[t].sumspike / self.spike_window
        else:
            y = self.last.sumspike / self.spike_window

        hidden_out = h
        return y, hidden_out


class SpikeLinear(torch.nn.Module):
    def __init__(self, args, in_features, out_features, nlab, layer=None):
        super(SpikeLinear, self).__init__()
        self.args = args
        if layer != -1:
            self.fc = torch.nn.Linear(in_features, out_features, bias=False)
            self.NI = torch.empty(2 * nlab, out_features).cuda()
            self.NI = NIclass.reset_weights_NI(self.args, self.NI)
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

        self.input_ = None
        self.label_ = None
        self.h = None

    def reset_weights(self):
        torch.nn.init.uniform_(self.NI)
        self.NI.requires_grad = False

    def local_modulation(self, neuromodulator_level):
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

    def expectation(self, labels):
        sigma = 1
        delta_mu = 1
        max_len = labels.shape[1]
        a = np.array([np.sqrt(2) * np.sqrt(np.log(max_len / i)) * sigma for i in range(1, max_len + 1)])
        a = a / a.max() * (2 * (max_len - delta_mu))
        b = delta_mu + a
        a = torch.tensor(a.astype('int')).to(labels.device)
        b = torch.tensor(b.astype('int')).to(labels.device)
        Ea = a[torch.max(labels, 1)[1].cpu()]
        Eb = b[torch.max(labels, 1)[1].cpu()]
        Ea = torch.zeros(labels.shape[0], 2 * labels.shape[1], device=labels.device).scatter_(1, Ea.unsqueeze(1).long(), 1.0)
        Eb = torch.zeros(labels.shape[0], 2 * labels.shape[1], device=labels.device).scatter_(1, Eb.unsqueeze(1).long(), 1.0)
        return (Ea + Eb) / 2

    # t: current task, x: input, y: output, e: epochs
    def forward(self, x, y, t, input_):
        self.input_ = x
        self.label_ = y
        if self.time_counter == 0:
            batchsize = x.shape[0]
            self.mem = torch.zeros((batchsize, self.out_features)).cuda()
            self.spike = torch.zeros((batchsize, self.out_features)).cuda()
            self.sumspike = torch.zeros((batchsize, self.out_features)).cuda()
            self.block_weights = torch.empty(self.fc.weight.data.size()).cuda()
            self.old_spike = torch.zeros((batchsize, self.out_features)).cuda()
        self.time_counter += 1
        self.mem, self.spike = mem_update(self.fc, x, self.mem, self.spike, self.old_spike)
        # A mask to spikes
        if self.args.STDP and self.layer != -1:
            h_mask = torch.mean(input_, 0, False)
            h_mask = F.interpolate(h_mask.unsqueeze(0).unsqueeze(0), size=[self.out_features])
            h_mask = h_mask.squeeze(0)
            if utils.args.alpha is not None:  # alpha is the threshold, h_mask may be zeros
                alpha = utils.args.alpha
            elif utils.args.delta_alpha is not None:  # ensure h_mask is not filled with zeros
                alpha = h_mask.max() - utils.args.delta_alpha
            else:
                sorted, indices = torch.sort(h_mask.reshape(1, -1))  # prop
                alpha = sorted[0][int(0.99 * h_mask.shape[1])]
            h_mask = torch.sigmoid(1000 * (h_mask - alpha))
            self.spike = self.spike * h_mask.expand_as(self.spike)
        self.sumspike += self.spike

        self.old_spike = self.spike.clone().detach()

        # y=None for inference
        if self.time_counter == self.spike_window:
            self.time_counter = 0
            if y is not None:
                # Hidden layers
                if self.layer != -1:
                    neuromodulator_level = self.expectation(y).mm(self.NI.view(-1, prod(self.NI.shape[1:]))).view(self.spike.shape)
                    self.spike.backward(gradient=self.local_modulation(neuromodulator_level), retain_graph=True)

                # Output layers
                else:
                    # MSE
                    err = (self.sumspike / self.spike_window) - y
                    # CE
                    err = torch.matmul(err, torch.eye(err.shape[1]).to(err.device))
                    self.spike.backward(gradient=err, retain_graph=True)

        return self.spike


class NIclass():
    def reset_weights_NI(args, NI):
        if args.NI_type == 'Regions_Standard':
            torch.nn.init.constant_(NI, 0)
            out, hid = NI.shape
            w = int(hid / out)
            b = torch.empty(out, w).cuda()
            nn.init.uniform_(b, a=-0.5, b=0.5)
            for i in range(out):
                NI[i, i * w:(i + 1) * w] = b[i, :]
        elif args.NI_type == 'Regions_Orthogonal_gain_10':
            torch.nn.init.constant_(NI, 0)
            out, hid = NI.shape
            w = int(hid / out)
            b = torch.empty(out, w).cuda()
            nn.init.orthogonal_(b, gain=10)
            for i in range(out):
                NI[i, i * w:(i + 1) * w] = b[i, :]
        elif args.NI_type == 'Regions_Orthogonal_gain_1':
            torch.nn.init.constant_(NI, 0)
            out, hid = NI.shape
            w = int(hid / out)
            b = torch.empty(out, w).cuda()
            nn.init.orthogonal_(b, gain=1)
            for i in range(out):
                NI[i, i * w:(i + 1) * w] = b[i, :]
        elif args.NI_type == 'Orthogonal':
            torch.nn.init.orthogonal_(NI, gain=1)
        elif args.NI_type == 'Uniform':
            nn.init.uniform_(NI, a=0, b=1)
        elif args.NI_type == 'Ones_like':
            nn.init.ones_(NI)
            print()
        elif args.NI_type == 'minus_Ones_like':
            nn.init.ones_(NI)
            NI = -1 * NI
            print()
        else:
            nn.init.kaiming_uniform_(NI)
        return NI


# Approximate BP
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


# Membrane potential
def mem_update(ops, x, mem, spike, old_spike, drop=None, lateral=None):
    if drop is None:
        mem = mem * spike_args['decay'] * (1. - spike) + torch.sigmoid(ops(x))
    else:
        mem = mem * spike_args['decay'] * (1. - spike) + torch.sigmoid(drop(ops(x)))

    if lateral:
        mem += lateral(spike)
    spike = act_fun(mem, old_spike)
    return mem, spike
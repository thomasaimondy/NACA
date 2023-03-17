# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import utils
from numpy import prod
import numpy as np
from torch.distributions.beta import Beta

spike_args = {}


class NetworkBuilder(nn.Module):
    def __init__(self, topology, input_size, input_channels, label_features, train_batch_size, dropout, fc_zero_init, spike_window, device, thresh, randKill, lens, decay, conv_act, hidden_act, output_act):
        super(NetworkBuilder, self).__init__()

        self.layers = nn.ModuleList()
        self.batch_size = train_batch_size
        self.spike_window = spike_window
        self.randKill = randKill
        self.device = device
        spike_args['thresh'] = thresh
        spike_args['lens'] = lens
        spike_args['decay'] = decay

        topology = topology.split('_')
        self.topology = topology
        topology_layers = []
        num_layers = 0
        for elem in topology:
            if not any(i.isdigit() for i in elem):
                num_layers += 1
                topology_layers.append([])
            topology_layers[num_layers - 1].append(elem)
        for i in range(num_layers):
            layer = topology_layers[i]
            try:
                if layer[0] == "CONV" and utils.args.network == 'ANN':
                    in_channels = input_channels if (i == 0) else out_channels
                    out_channels = int(layer[1])
                    input_dim = input_size if (i == 0) else int(output_dim / 2)
                    output_dim = int((input_dim - int(layer[2]) + 2 * int(layer[4])) / int(layer[3])) + 1
                    self.layers.append(CNN_block(in_channels=in_channels, out_channels=int(layer[1]), kernel_size=int(layer[2]), stride=int(layer[3]), padding=int(layer[4]), bias=True, activation=conv_act, dim_hook=[2 * label_features, out_channels, output_dim, output_dim], batch_size=self.batch_size))
                elif layer[0] == "FC" and utils.args.network == 'ANN':
                    if (i == 0):
                        input_dim = input_size**2
                        self.conv_to_fc = 0
                    elif topology_layers[i - 1][0] == "CONV":
                        input_dim = pow(int(output_dim / 2), 2) * int(topology_layers[i - 1][1])
                        self.conv_to_fc = i
                    else:
                        input_dim = output_dim

                    output_dim = int(layer[1])
                    output_layer = (i == (num_layers - 1))
                    self.layers.append(FC_block(in_features=input_dim, out_features=output_dim, bias=True, activation=output_act if output_layer else hidden_act, dropout=dropout, dim_hook=None if output_layer else [2 * label_features, output_dim], fc_zero_init=fc_zero_init, batch_size=train_batch_size))
                elif layer[0] == "CONV" and utils.args.network == 'SNN':
                    in_channels = input_channels if (i == 0) else out_channels
                    out_channels = int(layer[1])
                    input_dim = input_size if (i == 0) else int(output_dim / 2)
                    output_dim = int((input_dim - int(layer[2]) + 2 * int(layer[4])) / int(layer[3])) + 1
                    self.layers.append(SpikeCNN_block(in_channels=in_channels, out_channels=int(layer[1]), kernel_size=int(layer[2]), stride=int(layer[3]), padding=int(layer[4]), bias=True, dim_hook=[2 * label_features, out_channels, output_dim, output_dim], batch_size=self.batch_size, spike_window=self.spike_window))
                elif layer[0] == "FC" and utils.args.network == 'SNN':
                    if (i == 0):
                        input_dim = input_size**2
                        self.conv_to_fc = 0
                    elif topology_layers[i - 1][0] == "CONV":
                        input_dim = pow(int(output_dim / 2), 2) * int(topology_layers[i - 1][1])
                        self.conv_to_fc = i
                    else:
                        input_dim = output_dim

                    output_dim = int(layer[1])
                    output_layer = (i == (num_layers - 1))
                    self.layers.append(SpikeFC_block(in_features=input_dim, out_features=output_dim, bias=True, dropout=dropout, dim_hook=None if output_layer else [2 * label_features, output_dim], fc_zero_init=fc_zero_init, batch_size=train_batch_size, spike_window=self.spike_window))
                else:
                    raise NameError("=== ERROR: layer construct " + str(elem) + " not supported")
            except ValueError as e:
                raise ValueError("=== ERROR: unsupported layer parameter format: " + str(e))

    def forward(self, input, labels):
        input = input.float().to(self.device)

        if utils.args.network == 'SNN':
            for _ in range(self.spike_window):
                x = input > torch.rand(input.size()).float().to(utils.device) * self.randKill
                x = x.float()
                for i in range(len(self.layers)):
                    if i == self.conv_to_fc:
                        x = x.reshape(x.size(0), -1)
                    x = x.detach()
                    x = self.layers[i](x, labels)
            x = self.layers[-1].sumspike / self.spike_window

        elif utils.args.network == 'ANN':
            x = input.float().to(self.device)
            for i in range(len(self.layers)):
                if i == self.conv_to_fc:
                    x = x.reshape(x.size(0), -1)
                x = x.detach()
                x = self.layers[i](x, labels, None)

        return x


def expectation(labels):
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


def local_modulation(neuromodulator_level):
    lambda_inv = utils.args.lambda_inv
    theta_max = utils.args.theta_max
    with torch.no_grad():
        nl_ = neuromodulator_level.clone().detach()
        modulation = torch.zeros_like(neuromodulator_level).to(utils.device)
        phase_one = theta_max - (theta_max - 1) * (4 * nl_ - lambda_inv).pow(2) / lambda_inv**2
        phase_two = 4 * (nl_ - lambda_inv).pow(2) / lambda_inv**2
        phase_three = -4 * ((2 * lambda_inv - nl_) - lambda_inv).pow(2) / lambda_inv**2
        phase_four = (theta_max - 1) * (4 * (2 * lambda_inv - nl_) - lambda_inv).pow(2) / lambda_inv**2 - theta_max

        modulation[neuromodulator_level <= 0.5 * lambda_inv] = phase_one[neuromodulator_level <= 0.5 * lambda_inv]
        modulation[(0.5 * lambda_inv < neuromodulator_level) & (neuromodulator_level <= lambda_inv)] = phase_two[(0.5 * lambda_inv < neuromodulator_level) & (neuromodulator_level <= lambda_inv)]
        modulation[(lambda_inv < neuromodulator_level) & (neuromodulator_level <= 1.5 * lambda_inv)] = phase_three[(lambda_inv < neuromodulator_level) & (neuromodulator_level <= 1.5 * lambda_inv)]
        modulation[1.5 * lambda_inv < neuromodulator_level] = phase_four[1.5 * lambda_inv < neuromodulator_level]

    return modulation


def mem_update(ops, x, mem, spike, old_spike, drop=None, lateral=None):
    if drop is None:
        mem = mem.clone().detach() * spike_args['decay'] * (1. - spike) + ops(x)
    else:
        mem = mem.clone().detach() * spike_args['decay'] * (1. - spike) + drop(ops(x))

    if lateral:
        mem += lateral(spike)
    spike = act_fun(mem, old_spike)

    return mem, spike


class SpikeFC_block(nn.Module):
    def __init__(self, in_features, out_features, bias, dropout, dim_hook, fc_zero_init, batch_size, spike_window):
        super(SpikeFC_block, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.spike_window = spike_window
        self.dropout = dropout
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        if fc_zero_init:
            torch.zero_(self.fc.weight.data)
        if dropout != 0:
            self.drop = nn.Dropout(p=dropout)
        self.mem = None
        self.spike = None
        self.sumspike = None
        self.time_counter = 0
        self.old_spike = None

        self.dim_hook = dim_hook
        if self.dim_hook is not None:
            self.NI = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
            self.reset_weights()

    def reset_weights(self):
        self.NI.requires_grad = False
        if utils.args.lambda_0:
            torch.nn.init.zeros_(self.NI)
            return
        if utils.args.distribution == 'uniform':
            torch.nn.init.uniform_(self.NI)
        elif utils.args.distribution == 'normal':
            torch.nn.init.normal_(self.NI, mean=0.5, std=1)
            self.NI.clamp_(0, 1)
        elif utils.args.distribution == 'beta':
            dist = Beta(torch.ones_like(self.NI) * 0.5, torch.ones_like(self.NI) * 0.5)
            self.NI.data = dist.sample()

    def forward(self, x, labels):
        if self.time_counter == 0:
            self.mem = torch.zeros((self.batch_size, self.out_features)).to(utils.device)
            self.spike = torch.zeros((self.batch_size, self.out_features)).to(utils.device)
            self.sumspike = torch.zeros((self.batch_size, self.out_features)).to(utils.device)
            self.old_spike = torch.zeros((self.batch_size, self.out_features)).to(utils.device)

        self.time_counter += 1
        if self.dropout != 0:
            self.mem, self.spike = mem_update(self.fc, x, self.mem, self.spike, self.old_spike, self.drop)
        else:
            self.mem, self.spike = mem_update(self.fc, x, self.mem, self.spike, self.old_spike)
        self.sumspike += self.spike

        if self.time_counter == self.spike_window and labels is not None and self.dim_hook is not None:
            neuromodulator_level = expectation(labels).mm(self.NI.view(-1, prod(self.NI.shape[1:]))).view(self.sumspike.shape)
            self.spike.backward(gradient=local_modulation(neuromodulator_level), retain_graph=True)

        self.old_spike = self.spike.clone().detach()

        if self.time_counter == self.spike_window:
            self.time_counter = 0

        return self.spike


class SpikeCNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, dim_hook, batch_size, spike_window):
        super(SpikeCNN_block, self).__init__()
        self.spike_window = spike_window
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.mem = None
        self.spike = None
        self.sumspike = None
        self.time_counter = 0
        self.batch_size = batch_size
        self.out_channels = out_channels
        self.old_spike = None

        self.dim_hook = dim_hook
        if self.dim_hook is not None:
            self.NI = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
            self.reset_weights()

    def reset_weights(self):
        self.NI.requires_grad = False
        if utils.args.lambda_0:
            torch.nn.init.zeros_(self.NI)
            return
        if utils.args.distribution == 'uniform':
            torch.nn.init.uniform_(self.NI)
        elif utils.args.distribution == 'normal':
            torch.nn.init.normal_(self.NI, mean=0.5, std=1)
            self.NI.clamp_(0, 1)
        elif utils.args.distribution == 'beta':
            dist = Beta(torch.ones_like(self.NI) * 0.5, torch.ones_like(self.NI) * 0.5)
            self.NI.data = dist.sample()

    def forward(self, x, labels):
        if self.time_counter == 0:
            self.mem = torch.zeros((self.batch_size, self.out_channels, x.size()[-2], x.size()[-1])).to(utils.device)
            self.spike = torch.zeros((self.batch_size, self.out_channels, x.size()[-2], x.size()[-1])).to(utils.device)
            self.sumspike = torch.zeros((self.batch_size, self.out_channels, x.size()[-2], x.size()[-1])).to(utils.device)
            self.old_spike = torch.zeros((self.batch_size, self.out_channels, x.size()[-2], x.size()[-1])).to(utils.device)
        self.time_counter += 1
        self.mem, self.spike = mem_update(self.conv, x, self.mem, self.spike, self.old_spike)
        self.sumspike += self.spike

        if self.time_counter == self.spike_window and labels is not None and self.dim_hook is not None:
            neuromodulator_level = expectation(labels).mm(self.NI.view(-1, prod(self.NI.shape[1:]))).view(self.spike.shape)
            self.spike.backward(gradient=local_modulation(neuromodulator_level), retain_graph=True)

        if self.time_counter == self.spike_window:
            self.time_counter = 0

        self.old_spike = self.spike.clone().detach()

        x = self.pool(self.spike)

        return x


class FC_block(nn.Module):
    def __init__(self, in_features, out_features, bias, activation, dropout, dim_hook, fc_zero_init, batch_size):
        super(FC_block, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.batch_size = batch_size
        self.dropout = dropout
        self.fc = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        if fc_zero_init:
            torch.zero_(self.fc.weight.data)
        self.act = Activation(activation)
        if dropout != 0:
            self.drop = nn.Dropout(p=dropout)

        self.dim_hook = dim_hook
        if self.dim_hook is not None:
            self.NI = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
            self.reset_weights()

    def reset_weights(self):
        self.NI.requires_grad = False
        if utils.args.lambda_0:
            torch.nn.init.zeros_(self.NI)
            return
        if utils.args.distribution == 'uniform':
            torch.nn.init.uniform_(self.NI)
        elif utils.args.distribution == 'normal':
            torch.nn.init.normal_(self.NI, mean=0.5, std=1)
            self.NI.clamp_(0, 1)
        elif utils.args.distribution == 'beta':
            dist = Beta(torch.ones_like(self.NI) * 0.5, torch.ones_like(self.NI) * 0.5)
            self.NI.data = dist.sample()

    def forward(self, x, labels, y):
        x = self.fc(x)
        x = self.act(x)
        if self.dropout != 0:
            x = self.drop(x)

        if labels is not None and self.dim_hook is not None:
            neuromodulator_level = expectation(labels).mm(self.NI.view(-1, prod(self.NI.shape[1:]))).view(x.shape)
            x.backward(gradient=local_modulation(neuromodulator_level), retain_graph=True)

        self.out = x.detach()

        return x


class CNN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, activation, dim_hook, batch_size):
        super(CNN_block, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.act = Activation(activation)
        if utils.args.pool == 'Avg':
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.time_counter = 0
        self.batch_size = batch_size
        self.out_channels = out_channels

        self.dim_hook = dim_hook
        if self.dim_hook is not None:
            self.NI = nn.Parameter(torch.Tensor(torch.Size(dim_hook)))
            self.reset_weights()

    def reset_weights(self):
        self.NI.requires_grad = False
        if utils.args.lambda_0:
            torch.nn.init.zeros_(self.NI)
            return
        if utils.args.distribution == 'uniform':
            torch.nn.init.uniform_(self.NI)
        elif utils.args.distribution == 'normal':
            torch.nn.init.normal_(self.NI, mean=0.5, std=1)
            self.NI.clamp_(0, 1)
        elif utils.args.distribution == 'beta':
            dist = Beta(torch.ones_like(self.NI) * 0.5, torch.ones_like(self.NI) * 0.5)
            self.NI.data = dist.sample()

    def forward(self, x, labels, y):
        x = self.conv(x)
        x = self.act(x)
        if labels is not None and self.dim_hook is not None:
            neuromodulator_level = expectation(labels).mm(self.NI.view(-1, prod(self.NI.shape[1:]))).view(x.shape)
            x.backward(gradient=local_modulation(neuromodulator_level), retain_graph=True)
        x = self.pool(x)

        return x


class Activation(nn.Module):
    def __init__(self, activation):
        super(Activation, self).__init__()

        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "none":
            self.act = None
        else:
            raise NameError("=== ERROR: activation " + str(activation) + " not supported")

    def forward(self, x):
        if self.act == None:
            return x
        else:
            return self.act(x)


class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, old_spike):
        ctx.save_for_backward(input, old_spike)
        result = input.gt(spike_args['thresh']).float()
        return result

    @staticmethod
    def backward(ctx, grad_output):  # three pseudo-gradient mode for backpropagation
        input, old_spike = ctx.saved_tensors
        grad_input = grad_output.clone()
        new_spike = input > spike_args['thresh'] - spike_args['lens']
        delta_spike = new_spike.float() - old_spike

        return grad_input * delta_spike, None, None


act_fun = ActFun.apply
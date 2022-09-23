import sys
import torch


class Net(torch.nn.Module):
    def __init__(self, args, inputsize, taskcla, labsize, nhid, nlayers=1):
        super(Net, self).__init__()

        ncha, size, size2 = inputsize
        self.taskcla = taskcla
        self.args = args
        self.nlayers = nlayers

        self.act = torch.nn.Sigmoid()
        self.drop = torch.nn.Dropout(0)
        self.fc1 = torch.nn.Linear(ncha * size * size2, nhid, bias=False)
        if self.nlayers > 1:
            self.fc2 = torch.nn.Linear(nhid, nhid, bias=False)
        if self.nlayers > 2:
            self.fc3 = torch.nn.Linear(nhid, nhid, bias=False)

        self.last = torch.nn.ModuleList()
        for t, n in self.taskcla:
            if args.multi_output:
                self.last.append(torch.nn.Linear(nhid, n))
            else:
                self.last = torch.nn.Linear(nhid, n)
        return

    def forward(self, x, t):
        u = x.view(x.size(0), -1)

        u = self.drop(self.act(self.fc1(u)))
        if self.nlayers > 1:
            u = self.drop(self.act(self.fc2(u)))
        if self.nlayers > 2:
            u = self.drop(self.act(self.fc3(u)))

        if self.args.multi_output:
            y = self.last[t](u)
        else:
            y = self.last(u)

        return y

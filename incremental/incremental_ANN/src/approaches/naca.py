import time
import numpy as np
import torch
from tqdm import tqdm
import utils

torch.autograd.set_detect_anomaly(True)


class Appr(object):
    def __init__(self, model, nlab, nepochs=100, sbatch=16, lr=0.01, lr_min=5e-4, lr_factor=1, lr_patience=5, clipgrad=10000, args=None):
        self.model = model
        self.args = args
        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.nlab = nlab
        self.mse = torch.nn.MSELoss()
        self.optimizer = self._get_optimizer()

        return

    def _get_optimizer(self, lr=None):
        if lr is None: lr = self.lr
        return torch.optim.SGD(self.model.parameters(), lr=lr)

    def train(self, t, xtrain, ytrain, xvalid, yvalid):
        best_loss = np.inf
        best_model = utils.get_model(self.model)
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)

        # Loop epochs
        ytrain = torch.zeros(ytrain.shape[0], self.nlab).cuda().scatter_(1, ytrain.unsqueeze(1).long(), 1.0).cuda()
        yvalid = torch.zeros(yvalid.shape[0], self.nlab).cuda().scatter_(1, yvalid.unsqueeze(1).long(), 1.0).cuda()
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.train_epoch(t, xtrain, ytrain)
            clock1 = time.time()
            train_loss, train_acc = 0, 0
            clock2 = time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e + 1, 1000 * self.sbatch * (clock1 - clock0) / xtrain.size(0), 1000 * self.sbatch * (clock2 - clock1) / xtrain.size(0), train_loss, 100 * train_acc), end='')
            # Valid
            valid_loss, valid_acc = self.eval(t, xvalid, yvalid)
            print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss, 100 * valid_acc), end='')
            # Adapt lr
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = utils.get_model(self.model)
                patience = self.lr_patience
                print(' *', end='')
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    lr = max(lr, self.lr_min)
                    print(' lr={:.1e}'.format(lr), end='')
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()
            ## thomas ,quick training
            if valid_acc > 0.95:
                break
        utils.epoch.append(e)
        # Restore best validation model
        utils.set_model_(self.model, best_model)

    def train_epoch(self, t, x, y):
        self.model.train()

        r = np.arange(x.size(0))
        np.random.shuffle(r)
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in tqdm(range(0, len(r), self.sbatch)):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            images = torch.autograd.Variable(x[b], volatile=False)
            targets = torch.autograd.Variable(y[b], volatile=False)
            task = torch.autograd.Variable(torch.LongTensor([t]).cuda(), volatile=False)
            # Forward
            self.model.forward(task, images, targets)

            # Apply step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)

            self.optimizer.step()
            self.optimizer.zero_grad()

    def eval(self, t, x, y):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0, len(r), self.sbatch):
            if i + self.sbatch <= len(r):
                b = r[i:i + self.sbatch]
            else:
                b = r[i:]
            with torch.no_grad():
                images = torch.autograd.Variable(x[b])
                targets = torch.autograd.Variable(y[b])
                task = torch.autograd.Variable(torch.LongTensor([t]).cuda())

            # Forward
            output, _ = self.model.forward(task, images, None)
            loss = self.criterion(output, targets)
            _, pred = output.max(1)
            targets = targets.max(1)[1]
            hits = (pred == targets).float()

            # Log
            total_loss += loss.data.item() * len(b)
            total_acc += hits.sum().data.item()
            total_num += len(b)

        return total_loss / total_num, total_acc / total_num

    def criterion(self, outputs, targets):
        return self.mse(outputs, targets)

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
            with torch.no_grad():
                images = torch.autograd.Variable(x[b])
                target = torch.autograd.Variable(y[b])
                targets = torch.zeros(images.shape[0], self.nlab).to(target.device).scatter_(1, target.unsqueeze(1).long(), 1.0)
            # Forward
            self.model.forward(t, images, targets)

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
                target = torch.autograd.Variable(y[b])

            # Forward
            outputs, _ = self.model.forward(t, images, None)
            if utils.args.multi_output:
                output = outputs[t]
            else:
                output = outputs
            targets = torch.zeros_like(output).to(target.device).scatter_(1, target.unsqueeze(1).long(), 1.0)
            loss = self.criterion(output, targets)
            _, pred = output.max(1)
            hits = (pred == target).float()

            # Log
            total_loss += loss.data.item() * len(b)
            total_acc += hits.sum().data.item()
            total_num += len(b)

        return total_loss / total_num, total_acc / total_num

    def criterion(self, outputs, targets):
        return self.mse(outputs, targets)

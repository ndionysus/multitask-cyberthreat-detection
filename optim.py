import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Optimizer:
    def get_opt(self, params):
        # Set optimizer
        print("Optimizer:", self.conf["method"])
        # Adam Optimizer
        if self.conf["method"].lower() == "adam":
            optimizer = optim.Adam(params=params,
                                   weight_decay=self.conf["l2_loss"],
                                   lr=self.conf["lr"])
        # RMSPROP
        elif self.conf["method"].lower() == "rmsprop":
            optimizer = optim.RMSprop(params=params,
                                      weight_decay=self.conf["l2_loss"],
                                      lr=self.conf["lr"],
                                      momentum=conf["momentum"])
            # Stochastic Gradient Descent
        elif self.conf["method"].lower() == "sgd":
            optimizer = optim.SGD(params=params,
                                  weight_decay=self.conf["l2_loss"],
                                  lr=self.conf["lr"],
                                  momentum=self.conf["momentum"])
        else:
            raise("UNDEFINED OPTIMIZER", self.conf["method"])

        return optimizer

    def get_state_dict(self):
        return self.opt.state_dict()

    def train_op(self, op_loss):
        # Simple training op
        # Zero the gradient
        self.opt.zero_grad()
        # Backward pass
        op_loss.backward()
        # Update the parameters
        self.opt.step()

    def __init__(self, params, conf):
        self.conf = conf
        self.opt = self.get_opt(params)


class LR_scheduler:
    def get_sch(self, opt, conf):
        # Set scheduler
        if conf["lr_scheduler"] is None:  # Constant lr
            scheduler = None
        elif conf["lr_scheduler"].lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(opt,
                                                  step_size=conf["step"],
                                                  gamma=conf["decay"])
        else:
            raise("UNDEFINED SCHEDULER", conf["lr_scheduler"])

        return scheduler

    def get_state_dict(self):
        if self.sch is not None:
            return self.sch.state_dict()
        else:
            return None

    def step(self):
        if self.sch is not None:
            self.sch.step()

    def __init__(self, opt, conf):
        self.sch = self.get_sch(opt, conf)


class BCE(nn.BCEWithLogitsLoss):
    def __init__(self):
        super(BCE, self).__init__()

    def forward(self, x, y):
        op_loss = super(BCE, self).forward(x, y)
        return op_loss


class WeightedCEL(nn.CrossEntropyLoss):
    def __init__(self, tag_vocab):
        w = torch.ones(len(tag_vocab))
        w[tag_vocab["<p>"]] = 0.0
        w[tag_vocab["<s>"]] = 0.0
        w[tag_vocab["</s>"]] = 0.0
        super(WeightedCEL, self).__init__(weight=w)

    def forward(self, preds, target, mask):
        # to long
        target = target.long()
        # reshape labels to give a flat vector of length batch_size*seq_len
        target = target.view(-1)
        shape = preds.size()
        preds = preds.view(shape[0]*shape[1], -1)

        # cross entropy loss for all non 'PAD' tokens
        loss = super(WeightedCEL, self).forward(preds, target)
        return loss


class CustomLoss(nn.Module):

    def __init__(self, loss_fn):
        super(CustomLoss, self).__init__()
        self.loss_fn = loss_fn

    def forward(self, x, y, mask):
        loss = -(self.loss_fn(x, y, mask=mask))
        return loss

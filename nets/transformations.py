import torch
import torch.nn.functional as F
from torch.autograd import Variable


def get_laststep(model, lens):
    shape = model.size()
    idx = (lens - 1).view(-1, 1).expand(shape[0], model.size(2)).unsqueeze(1)
    model = model.gather(1, Variable(idx)).squeeze(1)
    return model


def pooled_output(model):
    # global average pooling
    avg_pool = torch.mean(model, 1)
    # global max pooling
    max_pool, _ = torch.max(model, 1)
    model = torch.cat((max_pool, avg_pool), 1)
    return model

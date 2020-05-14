import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import nets.transformations as trf


def init_all(m):
    t = type(m)
    # print(t)
    if t in [nn.Linear]:
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif t in [nn.Conv2d]:
        nn.init.kaiming_normal_(m.weight, mode='fan_in')
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif t in [nn.LSTM, nn.GRU]:
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)


class TranEnc(nn.Module):
    def __init__(self, dim, conf):
        super(TranEnc, self).__init__()
        tlayer = nn.TransformerEncoderLayer(d_model=dim, nhead=conf["nhead"],
                                            dim_feedforward=conf["dim_forward"],
                                            dropout=conf["dropout"])
        self.fw = nn.TransformerEncoder(tlayer, conf["n_layers"])

    def forward(self, model):
        return self.fw(model)


class MultiHeadAttn(nn.Module):
    def __init__(self, dim, nhead=8, dropout=0.0, bias=False):
        super(MultiHeadAttn, self).__init__()
        self.dim = dim
        self.nhead = nhead

        # Q, K, V layers
        self.q = nn.Linear(dim, dim*nhead, bias=bias)
        self.k = nn.Linear(dim, dim*nhead, bias=bias)
        self.v = nn.Linear(dim, dim*nhead, bias=bias)

        # output
        self.output = nn.Linear(dim*nhead, dim)

    def forward(self, x):
        # Get shapes
        batch, seq, dim = x.size()
        # Compute weights
        q = self.q(x).view(batch, seq, self.nhead, dim).transpose(1, 2)
        k = self.k(x).view(batch, seq, self.nhead, dim).transpose(1, 2)
        v = self.v(x).view(batch, seq, self.nhead, dim).transpose(1, 2)

        # Fold heads into the batch dimension
        q = q.contiguous().view(batch * self.nhead, seq, dim)
        k = k.contiguous().view(batch * self.nhead, seq, dim)
        v = v.contiguous().view(batch * self.nhead, seq, dim)

        # Scaled dot product
        q = q / (dim ** (1/4))
        k = k / (dim ** (1/4))
        # attn_w -> ( batch*nhead , seq , seq)
        attn_w = torch.bmm(q, k.transpose(1, 2))
        # apply softmax
        attn_w = F.softmax(attn_w, dim=2)

        # apply attn-w and compute final output
        out = torch.bmm(attn_w, v).view(batch, self.nhead, seq, dim)
        out = out.transpose(1, 2).contiguous().view(
            batch, seq, self.nhead * dim)
        out = self.output(out)

        return out


class Embedding(nn.Module):
    '''
        INPUT : (BATCH, SEQ_LEN)
        OUTPUT : (BATCH, SEQ_LEN, DIM)
    '''
    @staticmethod
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    def load_embeddings(self, path):
        with open(path, "r") as f:
            return dict(self.get_coefs(*line.strip().split(' ')) for line in f)

    def __init__(self, conf, vocab, d):
        super(Embedding, self).__init__()
        # TODO: CONVERT NP TO TORCH
        if conf["pretrained"] is not None and d == "w_dim":
            print("Pretrained vecs:", conf["pretrained"])
            # Initialize an empty matrix
            matrix = []
            # Get a Dict key : vec
            vec_dict = self.load_embeddings(conf["pretrained"])
            # Make sure the config dim and the pretrained vecs match
            dim = len(list(vec_dict.values())[0])
            conf[d] = dim
            # Iterate over the vocab and either get a w from the vec_dict or create a random vec
            for w, id in vocab.items():
                if w in vec_dict:
                    matrix.append(vec_dict[w])
                else:
                    vec = np.random.uniform(-0.5, 0.5, dim)
                    matrix.append(vec)
            del vec_dict
            w = torch.FloatTensor(matrix)
            self.emb = nn.Embedding.from_pretrained(w, freeze=False)
        else:
            self.emb = nn.Embedding(len(vocab), conf[d])

    def forward(self, x):
        return self.emb(x)


class word_emb(nn.Module):
    def __init__(self, conf, vocab):
        super(word_emb, self).__init__()
        # WORD EMBEDDINGS AND TWEET-LEVEL BILSTM
        self.w_emb = Embedding(conf, vocab, "w_dim")
        # Input Dropout
        self.w_in_dropout = SpatialDropout(conf["w_in_dropout"])

    def forward(self, word_ids):
        # embedd and apply inital dropout
        word_ids = self.w_emb(word_ids)
        word_ids = self.w_in_dropout(word_ids)
        return word_ids


class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        shape = x.size()
        # If x has 3 dimensions
        if len(shape) == 3:
            x = x.unsqueeze(2)    # (B, T, 1, D)
            x = x.permute(0, 3, 2, 1)  # (B, D 1, T)
            x = super(SpatialDropout, self).forward(x)  # (B, D, 1, T)
            x = x.permute(0, 3, 2, 1)  # (B, T, 1, D)
            x = x.squeeze(2)  # (B, T, D)
        # 4 dimensions
        else:
            x = x.permute(0, 3, 2, 1)  # (B, D, 1, T)
            x = super(SpatialDropout, self).forward(x)  # (B, D, 1, T)
            x = x.permute(0, 3, 2, 1)  # (B, T, 1, D)
        return x


class fcnn(nn.Module):
    '''
        INPUT : (BATCH, DIM)
        OUTPUT : (BATCH, DIM)
    '''

    def __init__(self, config, input):
        super(fcnn, self).__init__()
        self.net = nn.Sequential()
        for l, layer in enumerate(config):
            # xW + b
            self.net.add_module("Linear%d" % l, nn.Linear(input, layer["dim"]))
            nn.init.xavier_uniform_(self.net[-1].weight)
            # non-linear activation
            self.net.add_module("ReLu-{}".format(l), nn.ReLU())
            # dropout
            self.net.add_module("keep_prob-{}".format(l),
                                nn.Dropout(layer["dropout"]))
            if layer["batchnorm"]:
                # batchnorm
                self.net.add_module(
                    "BatchNormalization-{}".format(l), nn.BatchNorm1d(layer["dim"]))
            # update dim
            input = layer["dim"]

    def forward(self, x):
        return self.net(x)


class f_rnn(nn.Module):
    '''
        BUILD:
            conf : an array of dicts, each dict should provide hyperparameters
                   and design variables. See a config json file in ./configs/
            in_feats : expected input size
        FW:
            INPUT : (BATCH, SEQ ,DIM), (BATCH,SEQ_LEN)
            OUTPUT : (BATCH, SEQ, DIM)
    '''

    def __init__(self, config, in_feats):
        super(f_rnn, self).__init__()
        self.rnn_layers = nn.ModuleList()
        for l, layer in enumerate(config):
            if layer["cell"] == "lstm":
                rnn_l = nn.LSTM(in_feats, layer["dim"],
                                num_layers=layer["n_layers"],
                                bidirectional=layer["bidirectional"],
                                batch_first=True)
            else:
                rnn_l = nn.GRU(in_feats, layer["dim"],
                               num_layers=layer["n_layers"],
                               bidirectional=layer["bidirectional"],
                               batch_first=True)
            self.rnn_layers.add_module("RNN-%d" % l, rnn_l)
            in_feats = 2 * \
                layer["dim"] if layer["bidirectional"] else layer["dim"]
        self.output_size = in_feats

    def forward(self, seq, lens):
        # Get shape
        shape = seq.size()
        # sort
        lens, idx = torch.sort(lens, descending=True)
        seq = seq[idx]
        # pack_padded_sequence
        model = pack_padded_sequence(seq, lens, batch_first=True)
        # LSTM
        for layer in self.rnn_layers:
            layer.flatten_parameters()
            model, _ = layer(model)
        model, _ = pad_packed_sequence(model, batch_first=True)
        # resort back
        _, idx = torch.sort(idx, 0)
        model = model[idx]
        lens = lens[idx]
        return model


class char_RNN(nn.Module):
    def __init__(self, conf, vocab):
        super(char_RNN, self).__init__()
        # Character embeddings
        self.c_emb = Embedding(conf, vocab, "c_dim")
        # Sentence Input Dropout
        self.c_in_dropout = SpatialDropout(conf["c_in_dropout"])
        # Word-level Bilstm
        self.c_bilstm = word_RNN(
            conf["c_dim"], conf["c_rnn_out"], conf["c_rnn_layers"])
        self.output_size = self.c_bilstm.output_size

    def forward(self, char_ids, wlen):
        # Embedd
        char_ids = self.c_emb(char_ids)
        # Spatial Dropout
        char_ids = self.c_in_dropout(char_ids)
        # Reshape to send max_w_len to the time dimension
        shape = char_ids.size()
        char_ids = char_ids.view(shape[0]*shape[1], shape[2], -1)
        wlen = wlen.view(-1)
        # Model
        char_ids = self.c_bilstm(char_ids, wlen)
        char_ids = char_ids.view(shape[0], shape[1], -1)
        return char_ids


class word_RNN(nn.Module):
    def __init__(self, in_shape, output_shape, conf):
        super(word_RNN, self).__init__()
        self.output = output_shape
        self.conf = conf
        if self.output == "maxpool":  # (B,2*D)
            self.w_bilstm = f_rnn(conf, in_shape)
            self.output_size = self.w_bilstm.output_size*2
        elif self.output == "laststep":  # (B,D)
            self.w_bilstm = f_rnn(conf, in_shape)
            self.output_size = self.w_bilstm.output_size
        else:  # (B,T,D) allsteps
            self.w_bilstm = f_rnn(conf, in_shape)
            self.output_size = self.w_bilstm.output_size

    def forward(self, ids, lens):
        model = self.w_bilstm(ids, lens)
        if self.output == "maxpool":  # (B,2*D)
            return trf.pooled_output(model)
        elif self.output == "laststep":  # (B,D)
            return trf.get_laststep(model, lens)
        else:  # (B,T,D) allsteps
            return model


class char_CNN(nn.Module):
    def __init__(self, conf, vocab, dev="cpu"):
        super(char_CNN, self).__init__()
        # Device
        self.dev = dev
        # Character embeddings
        self.emb = Embedding(conf, vocab, "c_dim")
        # Input Dropout
        self.c_in_dropout = SpatialDropout(conf["c_in_dropout"])
        # Word-level Bilstm
        self.c_cnn_layers = word_CNN(conf["c_cnn_layers"], conf["c_dim"], 1)
        # output shape
        self.output_size = sum([l["filters"] for l in conf["c_cnn_layers"]])

    def forward(self, char_ids):
        # Embedd
        char_ids = self.emb(char_ids)
        # Spatial Dropout
        char_ids = self.c_in_dropout(char_ids)
        # Reshape to send max_w_len to the time dimension
        shape = char_ids.size()
        char_ids = char_ids.view(shape[0]*shape[1], 1, shape[2], shape[3])
        # Model
        char_ids = self.c_cnn_layers(char_ids, self.dev)
        char_ids = char_ids.view(shape[0], shape[1], -1)
        return char_ids


class word_CNN(nn.Module):
    '''
        INPUT : (BATCH, CHANNELS, SEQ_LEN, DIM)
        OUTPUT : (BATCH, SUM_OF_FILTERS)
    '''

    def __init__(self, conf, in_feats, in_channels, keep_dims=False):
        super(word_CNN, self).__init__()
        self.cnn_layers = nn.ModuleList()
        self.keep_dims = keep_dims
        for l, layer in enumerate(conf):
            if self.keep_dims:
                pad = layer["height"] - 1
            else:
                pad = 0
            self.cnn_layers.add_module("Conv-%d" % l, nn.Conv2d(in_channels, layer["filters"],
                                                                [layer["height"],
                                                                    in_feats],
                                                                padding=(pad//2, 0)))
        self.output_size = sum([l["filters"] for l in conf])

    def forward(self, x, dev):
        # CNN
        model = torch.Tensor([]).to(dev)
        # batch dim
        b = x.size(0)
        for i, layer in enumerate(self.cnn_layers):

            if self.keep_dims:
                h_f = layer.kernel_size[0]
                pad = h_f - 1
                if pad % 2 == 0:
                    pad_x = x
                else:
                    pad_x = F.pad(x, (0, 0, 0, 1))
            else:
                pad_x = x

            layer_output = layer(pad_x)

            if self.keep_dims:
                layer_output = layer_output.squeeze(-1)
            else:
                layer_output = F.max_pool2d(
                    layer_output, (layer_output.size(2), 1))
                layer_output = layer_output.view(b, -1)

            model = torch.cat([model, layer_output], 1)

        model = F.relu(model)
        return model

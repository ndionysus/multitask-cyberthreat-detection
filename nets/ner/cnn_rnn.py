import torch
import torch.nn as nn
from torchcrf import CRF
import numpy as np

from nets.base import NN
from nets.modules import word_emb, char_CNN, char_RNN, word_RNN, word_CNN, init_all


class RNN(NN):

    def __init__(self, conf, vocab, char_vocab, tag_vocab):
        super(RNN, self).__init__()
        # Word embedding and initial dropout
        self.w_input = word_emb(conf, vocab)

        # CHARACTER EMBEDDINGS AND WORD-LEVEL BILSTM
        self.use_chars = conf["use_chars"]
        if self.use_chars:
            self.char_RNN = char_RNN(conf, char_vocab)
            in_shape = self.char_RNN.output_size + conf["w_dim"]
        else:
            in_shape = conf["w_dim"]

        # Main BiLSTM
        self.word_RNN = word_RNN(in_shape, conf["w_rnn_out"], conf["w_rnn_layers"])
        output_size = self.word_RNN.output_size

        # Output Layer
        self.mid_dropout = nn.Dropout(conf["mid_dropout"])
        self.n_tags = len(tag_vocab)
        self.output = nn.Linear(output_size, self.n_tags)

        # CRF Layer
        self.use_crf = conf["use_crf"]
        if self.use_crf:
            self.crf = CRF(self.n_tags, batch_first=True)

        # Initialize weights
        # self.apply(init_all)

        # Maybe move to GPU
        self.to(self.device)

    def forward(self, word_ids, char_ids, slen, wlen, mask):
        # Get batch
        # word_ids = batch, max_seq_len, ids
        # char_ids = batch, max_seq_len, max_w_len ,ids
        # slen = batch, len
        # wlen = batch, max_seq_len, len
        # mask = batch, max_seq_len

        # embedd and apply inital dropout
        word_ids = self.w_input(word_ids)

        # Embedd and apply inital dropout - Chars
        if self.use_chars:
            char_ids = self.char_RNN(char_ids, wlen)
            word_ids = torch.cat([word_ids, char_ids], -1)

        model = self.word_RNN(word_ids, slen)
        model = self.mid_dropout(model)
        model = self.output(model)

        if self.use_crf and (not self.training):  # viterbi decoding
            model = self.crf.decode(model, mask)
        elif not self.training:  # non-crf inference
            model = torch.argmax(model, dim=-1)
            model = model.cpu().detach().tolist()

        return model


class RNN_CNN(NN):
    def __init__(self, conf, vocab, char_vocab, tag_vocab):
        super(RNN_CNN, self).__init__()
        # Word embedding and initial dropout
        self.w_input = word_emb(conf, vocab)

        # CHARACTER EMBEDDINGS AND WORD-LEVEL BILSTM
        self.use_chars = conf["use_chars"]
        if self.use_chars:
            self.char_RNN = char_RNN(conf, char_vocab)
            in_shape = self.char_RNN.output_size + conf["w_dim"]
        else:
            in_shape = conf["w_dim"]

        # CNN
        self.w_cnn_layers = word_CNN(conf["w_cnn_layers"], in_shape, 1, keep_dims=True)
        output_size = sum([l["filters"] for l in conf["w_cnn_layers"]])

        # Output Layer
        self.mid_dropout = nn.Dropout(conf["mid_dropout"])
        self.n_tags = len(tag_vocab)
        self.output = nn.Linear(output_size, self.n_tags)

        # CRF Layer
        self.use_crf = conf["use_crf"]
        if self.use_crf:
            self.crf = CRF(self.n_tags, batch_first=True)

        # Maybe move to GPU
        self.to(self.device)

    def forward(self, word_ids, char_ids, slen, wlen, mask):
        # Get batch
        # word_ids = batch, max_seq_len, ids
        # char_ids = batch, max_seq_len, max_w_len ,ids
        # slen = batch, len
        # wlen = batch, max_seq_len, len
        # mask = batch, max_seq_len

        # embedd and apply inital dropout
        word_ids = self.w_input(word_ids)

        # Embedd and apply inital dropout - Chars
        if self.use_chars:
            char_ids = self.char_RNN(char_ids, wlen)
            word_ids = torch.cat([word_ids, char_ids], -1)

        # SAVE SOME DIMENSIONS
        shape = word_ids.size()
        word_ids = word_ids.view(shape[0], 1, shape[1], shape[2])

        # CNN
        model = self.w_cnn_layers(word_ids, self.device)

        # SAVE SOME DIMENSIONS
        s = model.size()
        model = model.permute(0, 2, 1)
        model = self.mid_dropout(model)

        model = self.output(model)

        if self.use_crf and (not self.training):  # viterbi decoding
            model = self.crf.decode(model, mask)
        elif not self.training:  # non-crf inference
            model = torch.argmax(model, dim=-1)
            model = model.cpu().detach().tolist()

        return model


class CNN(NN):

    def __init__(self, conf, vocab, char_vocab, tag_vocab):
        super(CNN, self).__init__()
        # Word embedding and initial dropout
        self.w_input = word_emb(conf, vocab)

        # CHARACTER EMBEDDINGS AND WORD-LEVEL BILSTM
        self.use_chars = conf["use_chars"]
        if self.use_chars:
            self.char_cnn = char_CNN(conf, char_vocab, self.device)
            in_shape = self.char_cnn.output_size + conf["w_dim"]
        else:
            in_shape = conf["w_dim"]

        # CNN
        self.w_cnn_layers = word_CNN(conf["w_cnn_layers"], in_shape, 1, keep_dims=True)
        output_size = self.w_cnn_layers.output_size

        # Output Layer
        #self.mid_norm = nn.BatchNorm1d(output_size)
        self.mid_dropout = nn.Dropout(conf["mid_dropout"])
        self.n_tags = len(tag_vocab)
        self.output = nn.Linear(output_size, self.n_tags)

        # CRF Layer
        self.use_crf = conf["use_crf"]
        if self.use_crf:
            self.crf = CRF(self.n_tags, batch_first=True)

        # Maybe move to GPU
        self.to(self.device)

    def forward(self, word_ids, char_ids, slen, wlen, mask):
        # Get batch
        # word_ids = batch, max_seq_len, ids
        # char_ids = batch, max_seq_len, max_w_len ,ids
        # slen = batch, len
        # wlen = batch, max_seq_len, len
        # mask = batch, max_seq_len

        # embedd and apply inital dropout
        word_ids = self.w_input(word_ids)

        # Embedd and apply inital dropout - Chars
        if self.use_chars:
            char_ids = self.char_cnn(char_ids)
            word_ids = torch.cat([word_ids, char_ids], -1)

        # SAVE SOME DIMENSIONS
        shape = word_ids.size()
        word_ids = word_ids.view(shape[0], 1, shape[1], shape[2])

        # CNN
        model = self.w_cnn_layers(word_ids, self.device)

        # SAVE SOME DIMENSIONS
        s = model.size()
        model = model.permute(0, 2, 1)
        model = self.mid_dropout(model)

        model = self.output(model)

        if self.use_crf and (not self.training):  # viterbi decoding
            model = self.crf.decode(model, mask)
        elif not self.training:  # non-crf inference
            model = torch.argmax(model, dim=-1)
            model = model.cpu().detach().tolist()

        return model


class C_CNN(NN):

    def __init__(self, conf, vocab, char_vocab, tag_vocab):
        super(C_CNN, self).__init__()
        # CHARACTER EMBEDDINGS AND WORD-LEVEL BILSTM
        self.char_cnn = char_CNN(conf, char_vocab, self.device)
        in_shape = self.char_cnn.output_size

        # CNN
        self.w_cnn_layers = word_CNN(conf["w_cnn_layers"], in_shape, 1, keep_dims=True)
        output_size = self.w_cnn_layers.output_size

        # Output Layer
        self.mid_dropout = nn.Dropout(conf["mid_dropout"])
        self.n_tags = len(tag_vocab)
        self.output = nn.Linear(output_size, self.n_tags)

        # CRF Layer
        self.use_crf = conf["use_crf"]
        if self.use_crf:
            self.crf = CRF(self.n_tags, batch_first=True)

        # reset parameters to initializer
        # self.apply(init_all)

        # Maybe move to GPU
        self.to(self.device)

    def forward(self, word_ids, char_ids, slen, wlen, mask):
        # Get batch
        # word_ids = batch, max_seq_len, ids
        # char_ids = batch, max_seq_len, max_w_len ,ids
        # slen = batch, len
        # wlen = batch, max_seq_len, len
        # mask = batch, max_seq_len
        # Embedd and apply inital dropout - Chars
        word_ids = self.char_cnn(char_ids)
        # SAVE SOME DIMENSIONS
        shape = word_ids.size()
        word_ids = word_ids.view(shape[0], 1, shape[1], shape[2])

        # CNN
        model = self.w_cnn_layers(word_ids, self.device)
        s = model.size()
        model = model.permute(0, 2, 1)
        model = self.mid_dropout(model)

        model = self.output(model)

        if self.use_crf and (not self.training):  # viterbi decoding
            model = self.crf.decode(model, mask)
        elif not self.training:  # non-crf inference
            model = torch.argmax(model, dim=-1)
            model = model.cpu().detach().tolist()

        return model


class C_RNN(NN):

    def __init__(self, conf, vocab, char_vocab, tag_vocab):
        super(C_RNN, self).__init__()
        # CHARACTER EMBEDDINGS AND WORD-LEVEL BILSTM
        self.char_RNN = char_RNN(conf, char_vocab)
        in_shape = self.char_RNN.output_size

        # Main BiLSTM
        self.word_RNN = word_RNN(in_shape, conf["w_rnn_out"], conf["w_rnn_layers"])
        output_size = self.word_RNN.output_size

        # Output Layer
        self.mid_dropout = nn.Dropout(conf["mid_dropout"])
        self.n_tags = len(tag_vocab)
        self.output = nn.Linear(output_size, self.n_tags)

        # CRF Layer
        self.use_crf = conf["use_crf"]
        if self.use_crf:
            self.crf = CRF(self.n_tags, batch_first=True)

        # reset parameters to initializer
        # self.apply(init_all)

        # Maybe move to GPU
        self.to(self.device)

    def forward(self, word_ids, char_ids, slen, wlen, mask):
        # Get batch
        # word_ids = batch, max_seq_len, ids
        # char_ids = batch, max_seq_len, max_w_len ,ids
        # slen = batch, len
        # wlen = batch, max_seq_len, len
        # mask = batch, max_seq_len
        # Embedd and apply inital dropout - Chars
        word_ids = self.char_RNN(char_ids, wlen)

        model = self.word_RNN(word_ids, slen)
        model = self.mid_dropout(model)
        model = self.output(model)

        if self.use_crf and (not self.training):  # viterbi decoding
            model = self.crf.decode(model, mask)
        elif not self.training:  # non-crf inference
            model = torch.argmax(model, dim=-1)
            model = model.cpu().detach().tolist()

        return model


class CNN_RNN(NN):
    def __init__(self, conf, vocab, char_vocab, tag_vocab):
        super(CNN_RNN, self).__init__()
        # Word embedding and initial dropout
        self.w_input = word_emb(conf, vocab)

        # CHARACTER EMBEDDINGS AND WORD-LEVEL BILSTM
        self.use_chars = conf["use_chars"]
        if self.use_chars:
            self.char_cnn = char_CNN(conf, char_vocab, self.device)
            in_shape = self.char_cnn.output_size + conf["w_dim"]
        else:
            in_shape = conf["w_dim"]

        # Main BiLSTM
        self.word_RNN = word_RNN(in_shape, conf["w_rnn_out"], conf["w_rnn_layers"])
        output_size = self.word_RNN.output_size

        # Output Layer
        self.mid_dropout = nn.Dropout(conf["mid_dropout"])
        self.n_tags = len(tag_vocab)
        self.output = nn.Linear(output_size, self.n_tags)

        # CRF Layer
        self.use_crf = conf["use_crf"]
        if self.use_crf:
            self.crf = CRF(self.n_tags, batch_first=True)

        # Maybe move to GPU
        self.to(self.device)

    def forward(self, word_ids, char_ids, slen, wlen, mask):
        # Get batch
        # word_ids = batch, max_seq_len, ids
        # char_ids = batch, max_seq_len, max_w_len ,ids
        # slen = batch, len
        # wlen = batch, max_seq_len, len
        # mask = batch, max_seq_len

        # embedd and apply inital dropout
        word_ids = self.w_input(word_ids)

        # Embedd and apply inital dropout - Chars
        if self.use_chars:
            char_ids = self.char_cnn(char_ids)
            word_ids = torch.cat([word_ids, char_ids], -1)

        model = self.word_RNN(word_ids, slen)
        model = self.mid_dropout(model)
        model = self.output(model)

        if self.use_crf and (not self.training):  # viterbi decoding
            model = self.crf.decode(model, mask)
        elif not self.training:  # non-crf inference
            model = torch.argmax(model, dim=-1)
            model = model.cpu().detach().tolist()

        return model

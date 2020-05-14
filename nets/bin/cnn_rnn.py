import torch
import torch.nn as nn
import numpy as np

from nets.base import NN
from nets.modules import word_emb, char_CNN, char_RNN, word_RNN, word_CNN, init_all


class C_CNN(NN):

    def __init__(self, conf, vocab, char_vocab):
        super(C_CNN, self).__init__()
        # CHARACTER EMBEDDINGS AND WORD-LEVEL BILSTM
        self.char_cnn = char_CNN(conf, char_vocab, self.device)
        in_shape = self.char_cnn.output_size

        # CNN
        self.w_cnn_layers = word_CNN(conf["w_cnn_layers"], in_shape, 1)
        output_size = self.w_cnn_layers.output_size

        # Output Layer
        self.mid_dropout = nn.Dropout(conf["mid_dropout"])
        self.output = nn.Linear(output_size, 1)

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
        model = self.mid_dropout(model)
        model = self.output(model)
        return model


class C_RNN(NN):

    def __init__(self, conf, vocab, char_vocab):
        super(C_RNN, self).__init__()
        # CHARACTER EMBEDDINGS AND WORD-LEVEL BILSTM
        self.char_RNN = char_RNN(conf, char_vocab)
        in_shape = self.char_RNN.output_size

        # Main BiLSTM
        self.word_RNN = word_RNN(in_shape, conf["w_rnn_out"], conf["w_rnn_layers"])
        output_size = self.word_RNN.output_size

        # Output Layer
        self.mid_dropout = nn.Dropout(conf["mid_dropout"])
        self.output = nn.Linear(output_size, 1)

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
        char_ids = self.char_RNN(char_ids, wlen)
        model = self.word_RNN(char_ids, slen)
        model = self.mid_dropout(model)
        model = self.output(model)
        return model


class RNN(NN):

    def __init__(self, conf, vocab, char_vocab):
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
        self.output = nn.Linear(output_size, 1)

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

        word_ids = self.w_input(word_ids)

        # Embedd and apply inital dropout - Chars
        if self.use_chars:
            char_ids = self.char_RNN(char_ids, wlen)
            word_ids = torch.cat([word_ids, char_ids], -1)

        model = self.word_RNN(word_ids, slen)
        model = self.mid_dropout(model)
        model = self.output(model)
        return model


class CNN(NN):
    def __init__(self, conf, vocab, char_vocab):
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
        self.w_cnn_layers = word_CNN(conf["w_cnn_layers"], in_shape, 1)
        output_size = self.w_cnn_layers.output_size

        # Output Layer
        # self.mid_norm = nn.BatchNorm1d(output_size)
        self.mid_dropout = nn.Dropout(conf["mid_dropout"])
        self.output = nn.Linear(output_size, 1)

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

        # model = self.mid_norm(model)
        model = self.mid_dropout(model)
        model = self.output(model)
        return model


class CNN_RNN(NN):
    def __init__(self, conf, vocab, char_vocab):
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
        #self.mid_norm = nn.BatchNorm1d(output_size)
        self.mid_dropout = nn.Dropout(conf["mid_dropout"])
        self.output = nn.Linear(output_size, 1)

        # Maybe move to GPU
        self.to(self.device)

    def forward(self, word_ids, char_ids, slen, wlen, mask):
        # Get batch
        # word_ids = batch, max_seq_len, ids
        # char_ids = batch, max_seq_len, max_w_len ,ids
        # slen = batch, len
        # wlen = batch, max_seq_len, len
        # mask = batch, max_seq_len

        word_ids = self.w_input(word_ids)

        # Embedd and apply inital dropout - Chars
        if self.use_chars:
            char_ids = self.char_cnn(char_ids)
            word_ids = torch.cat([word_ids, char_ids], -1)

        # RNN
        model = self.word_RNN(word_ids, slen)

        #model = self.mid_norm(model)
        model = self.mid_dropout(model)
        model = self.output(model)
        return model


class RNN_CNN(NN):
    def __init__(self, conf, vocab, char_vocab):
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
        self.w_cnn_layers = word_CNN(conf["w_cnn_layers"], in_shape, 1)
        output_size = self.w_cnn_layers.output_size

        # Output Layer
        #self.mid_norm = nn.BatchNorm1d(output_size)
        self.mid_dropout = nn.Dropout(conf["mid_dropout"])
        self.output = nn.Linear(output_size, 1)

        # Maybe move to GPU
        self.to(self.device)

    def forward(self, word_ids, char_ids, slen, wlen, mask):
        # Get batch
        # word_ids = batch, max_seq_len, ids
        # char_ids = batch, max_seq_len, max_w_len ,ids
        # slen = batch, len
        # wlen = batch, max_seq_len, len
        # mask = batch, max_seq_len

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

        #model = self.mid_norm(model)
        model = self.mid_dropout(model)
        model = self.output(model)
        return model

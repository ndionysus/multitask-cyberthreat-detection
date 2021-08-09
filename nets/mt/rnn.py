import torch
import torch.nn as nn
from torchcrf import CRF
import numpy as np

import nets.transformations as trf

from nets.base import NN
from nets.modules import Embedding, SpatialDropout, char_RNN, word_RNN

'''
INPUT -> CHAR EMB -> CHAR BILSTM +             - MaxPool + Global Avg - > Sigmoid(1)
      -> WORD EMB -> ----------- + WORD BILSTM |
                                               - CRF(N-tags)
'''


class RNN(NN):

    def __init__(self, conf, vocab, char_vocab, tag_vocab):
        super(RNN, self).__init__()
        # WORD EMBEDDINGS AND TWEET-LEVEL BILSTM
        # ======================================================================
        self.w_emb = Embedding(conf, vocab, "w_dim")
        # Sentence Input Dropout
        self.w_in_dropout = SpatialDropout(conf["w_in_dropout"])

        # CHARACTER EMBEDDINGS AND WORD-LEVEL BILSTM
        # ======================================================================
        self.use_chars = conf["use_chars"]
        if self.use_chars:
            self.char_RNN = char_RNN(conf, char_vocab)
            in_shape = self.char_RNN.output_size + conf["w_dim"]
        else:
            in_shape = conf["w_dim"]

        # Main BiLSTM
        # ======================================================================
        self.word_RNN = word_RNN(in_shape, None, conf["w_rnn_layers"])
        output_size = self.word_RNN.output_size
        self.mid_dropout = nn.Dropout(conf["mid_dropout"])
        self.w_bin_out = conf["w_bin_out"]
        out_bin_size = 2*output_size if self.w_bin_out == "maxpool" else output_size

        # Output Layers
        # ======================================================================
        # Binary
        self.bin_out = nn.Linear(out_bin_size, 1)

        # NER
        self.n_tags = len(tag_vocab)
        self.ner_out = nn.Linear(output_size, self.n_tags)

        # CRF Layer
        self.use_crf = conf["use_crf"]
        if self.use_crf:
            self.crf = CRF(self.n_tags, batch_first=True)
        # ======================================================================

        # Maybe move to GPU
        self.to(self.device)

    def shared(self, word_ids, char_ids, slen, wlen):
        # Get batch
        # word_ids = batch, max_seq_len, ids
        # char_ids = batch, max_seq_len, max_w_len ,ids
        # slen = batch, len
        # wlen = batch, max_seq_len, len

        # embedd and apply inital dropout
        word_ids = self.w_emb(word_ids)
        word_ids = self.w_in_dropout(word_ids)

        # Embedd and apply inital dropout - Chars
        if self.use_chars:
            char_ids = self.char_RNN(char_ids, wlen)
            word_ids = torch.cat([word_ids, char_ids], -1)

        model = self.word_RNN(word_ids, slen)
        model = self.mid_dropout(model)
        return model

    def _bin_fw(self, model, lens):
        '''Binary task specific layer(s) forward pass'''
        if self.w_bin_out == "maxpool":
            bin_out = trf.pooled_output(model)
        else:
            bin_out = trf.get_laststep(model, lens)
        bin_out = self.bin_out(bin_out)
        return bin_out

    def _ner_fw(self, model, mask):
        '''NER task specific layer(s) forward pass'''
        ner_out = self.ner_out(model)
        if self.use_crf and (not self.training):  # viterbi decoding
            ner_out = self.crf.decode(ner_out, mask)
        elif not self.training:  # non-crf inference
            ner_out = torch.argmax(ner_out, dim=-1)
            ner_out = ner_out.cpu().detach().tolist()
        return ner_out

    def bin_fw(self, word_ids, char_ids, slen, wlen, mask):
        '''TRAINING OR EVALUATION'''
        # Pass through the initial shared layers
        model = self.shared(word_ids, char_ids, slen, wlen)
        # Bin pass
        return self._bin_fw(model, slen)

    def ner_fw(self, word_ids, char_ids, slen, wlen, mask):
        '''TRAINING OR EVALUATION'''
        # Pass through the initial shared layers
        model = self.shared(word_ids, char_ids, slen, wlen)
        # NER pass
        return self._ner_fw(model, mask)

    def forward(self, word_ids, char_ids, slen, wlen, mask):
        '''INFERENCE'''
        # Pass through the initial shared layers
        model = self.shared(word_ids, char_ids, slen, wlen)
        # Bin output
        bin_out = self._bin_fw(model, slen)
        # NER output
        ner_out = self._ner_fw(model, mask)
        return bin_out, ner_out

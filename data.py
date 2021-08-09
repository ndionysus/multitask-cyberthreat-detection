import torch
import numpy as np
from torch.utils import data

import pandas as pd
from sklearn.model_selection import train_test_split, KFold

from time import time


class Dataset:

    def tag2tok(self, tags):
        if pd.isnull(tags):
            return np.nan
        tok_tags = [self.tag_vocab["<s>"]]
        for t in tags.split(" "):
            tok_tags.append(self.tag_vocab[t])
        tok_tags.append(self.tag_vocab["</s>"])
        return tok_tags

    def str2tok(self, s):
        sentence = [self.vocab["<s>"]]
        words = [[self.char_vocab["<s>"]]]
        for w in s.split(" "):
            chars = []
            for c in w:
                if c in self.char_vocab:
                    chars.append(self.char_vocab[c])
                else:
                    chars.append(self.char_vocab["<unk>"])
            words.append(chars)
            if w in self.vocab:
                sentence.append(self.vocab[w])
            else:
                sentence.append(self.vocab["<unk>"])

        sentence.append(self.vocab["</s>"])
        words.append([self.char_vocab["</s>"]])

        return sentence, words

    def fit(self, df):
        tok_tweet, tok_chars, slen, wlen, tok_tags = [], [], [], [], []
        for w, t in zip(df["clean_tweet"], df["entities"]):
            tk, tc = self.str2tok(w)
            tt = self.tag2tok(t)

            tok_tweet.append(tk)
            tok_chars.append(tc)
            tok_tags.append(tt)
            slen.append(len(tk))
            wlen.append(max([len(w) for w in tc]))

        df["tok_tweet"] = tok_tweet
        df["tok_chars"] = tok_chars
        df["tok_tags"] = tok_tags
        df["slen"], df["wlen"] = slen, wlen
        return df

    def build_vocabs(self, set):
        # Initialize vocabulary
        self.vocab = {"<p>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        self.char_vocab = {"<p>": 0, "<unk>": 1, "<s>": 2, "</s>": 3}
        self.tag_vocab = {"<p>": 0, "<s>": 1, "</s>": 2}

        # fill a dict with word : count
        w_count = dict()
        for n, line in enumerate(set["clean_tweet"]):
            for w in line.split(" "):
                w_count[w] = 1 if w not in w_count else w_count[w]+1

        # add words to the vocab if they are above threshold
        for w, c in w_count.items():
            if (c >= self.threshold):
                self.vocab[w] = len(self.vocab)
                for char in w:
                    if not char in self.char_vocab:
                        self.char_vocab[char] = len(self.char_vocab)
        del w_count

        # tags
        for s in set[set["relevant"] == 1]["entities"]:
            for w in s.split():
                if not w in self.tag_vocab:
                    self.tag_vocab[w] = len(self.tag_vocab)

        # Save a inverted vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.inv_tags = {v: k for k, v in self.tag_vocab.items()}

    def __init__(self, batch_size=256, threshold=100, tokenizer=None):
        # Variables
        # threshold = min occurences to keep a word in the vocab
        self.threshold = threshold
        self.batch_size = batch_size
        self.tokenizer = tokenizer

        # Load Dataset
        dset = pd.read_csv("./datasets/all_data.csv", sep="\t")
        dset["timestamp"] = pd.to_datetime(
            dset["timestamp"]).dt.tz_convert(None)

        # Timestamps
        train = (pd.Timestamp(2016, 11, 21), pd.Timestamp(2017, 4, 1))
        test = (pd.Timestamp(2018, 6, 1), pd.Timestamp(2018, 9, 2))

        # Time Mask
        train = (dset["timestamp"] >= train[0]) & (
            dset["timestamp"] < train[1])
        test = (dset["timestamp"] >= test[0]) & (
            dset["timestamp"] < test[1])

        # Apply mask with dset[mask] and make the train, val and test sets
        self.train, self.val = train_test_split(dset[train].copy())
        self.test = dset[test].copy()
        self.pad = 0

        if self.tokenizer is None:
            # Build Vocabulary
            self.build_vocabs(self.train)

            # Fit datasets
            self.train = self.fit(self.train)
            self.val = self.fit(self.val)
            self.test = self.fit(self.test)

    def build_batches(self, y="relevant"):

        def _get_pos(x):
            return x[x["relevant"] == 1]

        # Make Pytorch Dataset
        if self.tokenizer is None:
            if y == "tok_tags":
                train = Set(_get_pos(self.train), y=y)
                val = Set(_get_pos(self.val), y=y)
                test = Set(_get_pos(self.test), y=y)
            else:
                train = Set(self.train)
                val = Set(self.val)
                test = Set(self.test)
            collate_fn = PadSequence
        else:
            pass  # TODO

        # Batches
        train_batch = data.DataLoader(train, batch_size=self.batch_size,
                                      shuffle=True, collate_fn=collate_fn(y, self.pad))
        val_batch = data.DataLoader(val, batch_size=self.batch_size,
                                    collate_fn=collate_fn(y, self.pad))
        test_batch = data.DataLoader(test, batch_size=self.batch_size,
                                     collate_fn=collate_fn(y, self.pad))
        return train_batch, val_batch, test_batch


class DatasetUtil:

    def pad(self, batch):
        # unzip batch
        tok_tweet, tok_chars, slen, wlen = batch

        # maximum lengths
        max_s_len = max(slen)
        max_w_len = max(wlen)

        # Initialize padded array
        batch_len = len(tok_tweet)
        padded_tweet = np.zeros([batch_len, max_s_len])
        padded_chars = np.zeros([batch_len, max_s_len, max_w_len])
        padded_wlen = np.ones([batch_len, max_s_len])
        mask = np.zeros([batch_len, max_s_len])

        # Pad
        for i, line in enumerate(tok_tweet):
            limit = min(len(line), max_s_len)
            padded_tweet[i][0: limit] = line[0: limit]
            mask[i][0:limit] = 1

            # Pad characters
            for j, word in enumerate(tok_chars[i]):
                padded_wlen[i][j] = max(len(word), 1)
                limit = min(len(word), max_w_len)
                padded_chars[i][j][0: limit] = word[0: limit]

        return {"word_ids": padded_tweet, "char_ids": padded_chars, "slen": slen, "wlen": padded_wlen, "mask": mask}

    def str2tok(self, s):
        sentence = [self.vocab["<s>"]]
        words = [[self.char_vocab["<s>"]]]
        for w in s.split(" "):
            chars = []
            for c in w:
                if c in self.char_vocab:
                    chars.append(self.char_vocab[c])
                else:
                    chars.append(self.char_vocab["<unk>"])
            words.append(chars)
            if w in self.vocab:
                sentence.append(self.vocab[w])
            else:
                sentence.append(self.vocab["<unk>"])

        sentence.append(self.vocab["</s>"])
        words.append([self.char_vocab["</s>"]])

        return sentence, words

    def fit(self, line):
        tok_tweet, tok_chars, slen, wlen = [], [], [], []
        tk, tc = self.str2tok(line)

        tok_tweet.append(tk)
        tok_chars.append(tc)
        slen.append(len(tk))
        wlen.append(max([len(w) for w in tc]))

        padded_input = self.pad((tok_tweet, tok_chars, slen, wlen))

        return padded_input

    def decode_tags(self, tags):
        return [self.inv_tags[tag] for tag in tags]

    def __init__(self, word_vocab, char_vocab, tag_vocab):
        self.vocab = word_vocab
        self.char_vocab = char_vocab
        self.tag_vocab = tag_vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.inv_tags = {v: k for k, v in self.tag_vocab.items()}


class Set(data.Dataset):
    def __init__(self, df, y="relevant"):
        self.df = df
        self.y = y

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Select a single sample
        x = self.df.iloc[index]
        tok_tweet = x["tok_tweet"]
        tok_chars = x["tok_chars"]
        slen = x["slen"]
        wlen = x["wlen"]
        y = x[self.y]
        return tok_tweet, tok_chars, slen, wlen, y


class PadSequence:
    def __init__(self, y="relevant", pad=0):
        self.y = y

    def __call__(self, batch):

        # unzip batch
        tok_tweet, tok_chars, slen, wlen, y = zip(*batch)

        # maximum lengths
        max_s_len = max(slen)
        max_w_len = max(wlen)

        # Initialize padded array
        batch_len = len(tok_tweet)
        padded_tweet = np.zeros([batch_len, max_s_len])
        padded_chars = np.zeros([batch_len, max_s_len, max_w_len])
        padded_wlen = np.ones([batch_len, max_s_len])
        mask = np.zeros([batch_len, max_s_len])

        if self.y == "tok_tags":
            padded_y = np.zeros([batch_len, max_s_len])
        else:
            padded_y = y

        # Pad
        for i, (line, tags) in enumerate(zip(tok_tweet, y)):
            limit = min(len(line), max_s_len)
            padded_tweet[i][0: limit] = line[0: limit]
            mask[i][0:limit] = 1
            # Pad tags
            if self.y == "tok_tags":
                padded_y[i][0: limit] = tags[0: limit]
            # Pad characters
            for j, word in enumerate(tok_chars[i]):
                padded_wlen[i][j] = max(len(word), 1)
                limit = min(len(word), max_w_len)
                padded_chars[i][j][0: limit] = word[0: limit]

        return {"word_ids": padded_tweet, "char_ids": padded_chars, "slen": slen, "wlen": padded_wlen, "mask": mask}, padded_y

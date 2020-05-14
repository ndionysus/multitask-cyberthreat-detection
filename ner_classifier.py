import numpy as np
import pandas as pd
import torch
import json
import argparse


import utils
from data import Dataset
from optim import Optimizer, LR_scheduler, WeightedCEL, CustomLoss

from nets.ner.cnn_rnn import CNN_RNN, RNN_CNN, RNN, CNN, C_CNN, C_RNN


class Main:

    def build_model(self, conf, vocab, char_vocab, tag_vocab):
        if conf["arch"] == "rnn":
            return RNN(conf, vocab, char_vocab, tag_vocab)
        elif conf["arch"] == "cnn":
            return CNN(conf, vocab, char_vocab, tag_vocab)
        elif conf["arch"] == "cnn_rnn":
            return CNN_RNN(conf, vocab, char_vocab, tag_vocab)
        elif conf["arch"] == "rnn_cnn":
            return RNN_CNN(conf, vocab, char_vocab, tag_vocab)
        elif conf["arch"] == "char_cnn":
            return C_CNN(conf, vocab, char_vocab, tag_vocab)
        elif conf["arch"] == "char_rnn":
            return C_RNN(conf, vocab, char_vocab, tag_vocab)
        else:
            raise Exception("INVALID ARCH")

    def fw_pass(self, model, batch):
        # input, labels
        padded_input, y = batch
        # convert into tensors
        for key, val in padded_input.items():
            if key == "mask":
                padded_input[key] = model.Tensor(val).byte()
            else:
                padded_input[key] = model.Tensor(val).long()
        # Make y a tensor
        y = model.Tensor(y).long()
        # get predictions
        preds = model(**padded_input)

        return preds, y, padded_input["mask"]

    def __init__(self, dset, conf, save=False):
        # Set batches
        train, val, test = dset.build_batches("tok_tags")

        # Build model
        model = self.build_model(
            conf["model"], dset.vocab, dset.char_vocab, dset.tag_vocab)
        opt = Optimizer(model.parameters(), conf["optim"])
        if conf["model"]["use_crf"]:
            loss = CustomLoss(model.crf)
        else:
            loss = WeightedCEL(dset.tag_vocab)
        lr_sch = LR_scheduler(opt.opt, conf["optim"])

        # To track early stopping
        self.best = {"val": {"f1": 0}, "test": {}}
        step, stop = 0, 0

        # Tags to ignore in metrics
        ign_tok = [dset.tag_vocab["<p>"], dset.tag_vocab["<s>"],
                   dset.tag_vocab["</s>"]]

        for ep in range(conf["train"]["max_epochs"]):
            print("\n\tEpoch %d" % ep)
            for batch in train:
                # set the in training mode.
                model.train()
                # advance step
                step += 1
                # forward pass
                x, y, mask = self.fw_pass(model, batch)
                # measure error
                fw_loss = loss(x, y, mask)
                # backward pass
                opt.train_op(fw_loss)

                # validation
                if step % conf["train"]["val_steps"] == 0:
                    # Set the in testing mode
                    model.eval()
                    # Eval on val set
                    val_metrics = utils.ner_fw_eval(model, self.fw_pass,
                                                    val, step, ign_tok)
                    if val_metrics["f1"] > self.best["val"]["f1"]:
                        # reset Early stop
                        stop = 0
                        # Eval on test set
                        test_metrics = utils.ner_fw_eval(model, self.fw_pass, test,
                                                         step, ign_tok)
                        self.best = {"val": val_metrics, "test": test_metrics}
                        if save:
                            model.save(step, conf, self.best,
                                       opt, lr_sch, "ner")
                    else:
                        if stop == conf["train"]["patience"]:
                            return
                        stop += 1
                # maybe update lr
                lr_sch.step()


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(
        description='CyberThreat Tool <-> Named Entity Recognition')
    parser.add_argument('-load', metavar='load', default=None,
                        help='Checkpoint location.')
    parser.add_argument('-save', metavar='save', type=utils.str2bool,
                        help='Boolean', default=False)
    parser.add_argument('-conf', metavar='config', default="./conf/ner/rnn.json",
                        help='model configuration. JSON files defined in ./configs/')
    parser.add_argument('-ckpt', metavar='ckpt', help='Checkpoint location')
    args = parser.parse_args()

    # Set Seed for reproducibility
    utils.set_seed()

    if args.load is None:
        # Load Config file
        conf = json.load(open(args.conf, "r"))
        # Load Dataset object
        dset = Dataset(batch_size=conf["train"]["batch_size"])
        # Main training loop
        Main(dset, conf, save=args.save)

    else:
        # Load Config file
        # Load Model
        pass

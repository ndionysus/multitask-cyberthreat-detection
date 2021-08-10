import enum
import utils
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm

from data import Dataset, DatasetUtil
from optim import *

from nets.mt.rnn import RNN
from nets.mt.cnn import CNN
from nets.mt.cnn_rnn import CNN_RNN
from nets.mt.rnn_cnn import RNN_CNN
from nets.mt.char_cnn import C_CNN
from nets.mt.char_rnn import C_RNN


def build_model(conf, vocab, char_vocab, tag_vocab):
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


class Main:

    def bin_fw_pass(self, model, batch):
        # input, labels
        padded_input, y = batch
        # convert into tensors
        for key, val in padded_input.items():
            if key == "mask":
                padded_input[key] = model.Tensor(val).byte()
            else:
                padded_input[key] = model.Tensor(val).long()
        y = model.Tensor(y).float().view(-1, 1)
        # get predictions
        preds = model.bin_fw(**padded_input)
        return preds, y, padded_input["mask"]

    def ner_fw_pass(self, model, batch):
        # input, labels
        padded_input, y = batch
        # convert into tensors
        for key, val in padded_input.items():
            padded_input[key] = model.Tensor(val).long()
        padded_input["mask"] = padded_input["word_ids"].data.gt(0).byte()
        y = model.Tensor(y).long()
        # get predictions
        preds = model.ner_fw(**padded_input)
        return preds, y, padded_input["mask"]

    def fw_pass(self, model, task):
        # input, labels
        batch = self.get_batch(task)
        if task == "bin":
            return self.bin_fw_pass(model, batch)
        elif task == "ner":
            return self.ner_fw_pass(model, batch)

    def get_batch(self, task):
        if task == "bin":
            # Binary task input
            try:
                batch = next(self.iter_bin)
            except StopIteration:
                self.bin_ep += 1
                self.iter_bin = iter(self.bin_train)
                batch = next(self.iter_bin)
        elif task == "ner":
            # NER task input
            try:
                batch = next(self.iter_ner)
            except StopIteration:
                self.ner_ep += 1
                self.iter_ner = iter(self.ner_train)
                batch = next(self.iter_ner)
        else:
            raise Exception(task)
        return batch

    def __init__(self, dset, conf, save=False):
        # Set batches
        self.bin_train, self.bin_val, self.bin_test = dset.build_batches(
            "relevant")
        self.ner_train, self.ner_val, self.ner_test = dset.build_batches(
            "tok_tags")

        # Create iterators. Because we cannot just iterate through 2 Dataloaders
        self.iter_bin = iter(self.bin_train)
        self.iter_ner = iter(self.ner_train)

        # Build model
        model = build_model(conf["model"], dset.vocab,
                            dset.char_vocab, dset.tag_vocab)
        # Optimizer(s)
        bin_opt = Optimizer(model.parameters(), conf["optim"]["bin"])
        bin_loss = BCE()
        bin_lr_sch = LR_scheduler(bin_opt.opt, conf["optim"]["bin"])
        # Binary
        # NER
        ner_opt = Optimizer(model.parameters(), conf["optim"]["ner"])
        if conf["model"]["use_crf"]:
            ner_loss = CustomLoss(model.crf)
        else:
            ner_loss = WeightedCEL(dset.tag_vocab)
        ner_lr_sch = LR_scheduler(ner_opt.opt, conf["optim"]["ner"])

        # To track early stopping
        self.best = {"val": {"f1": 0}, "test": {}}
        self.bin_ep, self.ner_ep, stop = 0, 0, 0

        # Tags to ignore in metrics
        ign_tok = [dset.tag_vocab["<p>"],
                   dset.tag_vocab["<s>"], dset.tag_vocab["</s>"]]

        # Train loop
        for step in range(conf["train"]["max_steps"]):
            # Set training mode
            model.train()

            for task in ["bin", "ner"]:
                # Train
                x, y, mask = self.fw_pass(model, task)

                if task == "bin":
                    fw_loss = bin_loss(x, y)
                    bin_opt.train_op(fw_loss)
                else:
                    fw_loss = ner_loss(x, y, mask)
                    ner_opt.train_op(fw_loss)

            # validation
            if step % conf["train"]["val_steps"] == 0:
                # Set the module in testing mode
                model.eval()

                # Eval on val set
                print("\n\tVALIDATION\n")
                val_bin = utils.bin_fw_eval(
                    model, self.bin_fw_pass, self.bin_val, step)
                val_ner = utils.ner_fw_eval(
                    model, self.ner_fw_pass, self.ner_val, step, ign_tok)
                f1 = (2*val_bin["f1"]*val_ner["f1"]) / \
                    (val_bin["f1"]+val_ner["f1"])
                print("\n\t\tF1:\t%.5f" % (f1))

                if f1 > self.best["val"]["f1"]:
                    # reset Early stop
                    stop = 0
                    # Eval on test set
                    print("\n\tTEST\n")
                    test_bin = utils.bin_fw_eval(
                        model, self.bin_fw_pass, self.bin_test, step)
                    test_ner = utils.ner_fw_eval(
                        model, self.ner_fw_pass, self.ner_test, step, ign_tok)
                    test_f1 = (2*test_bin["f1"]*test_ner["f1"]
                               ) / (test_bin["f1"]+test_ner["f1"])
                    print("\n\t\tF1:\t%.5f" % (test_f1))
                    # save best
                    self.best = {"val": {"f1": f1, "bin": val_bin, "ner": val_ner},
                                 "test": {"f1": test_f1, "bin": test_bin, "ner": test_ner}}

                    if save:
                        model.save(step, conf, self.best,
                                   (bin_opt, ner_opt), (bin_lr_sch, ner_lr_sch),
                                   dset.vocab, dset.char_vocab, dset.tag_vocab,
                                   task="mt")
                else:
                    if stop == conf["train"]["patience"]:
                        return
                    stop += 1

            # maybe update lr
            ner_lr_sch.step()
            bin_lr_sch.step()


class Synapse:

    def fw_pass(self, model, padded_input):
        # convert into tensors
        for key, val in padded_input.items():
            padded_input[key] = model.Tensor(val).long()
        padded_input["mask"] = padded_input["word_ids"].data.gt(0).byte()
        # get predictions
        bin_out, ner_out = model(**padded_input)
        return bin_out, ner_out

    def __init__(self, conf, ckpt_path, synapse, output="output.json"):
        ckpt = torch.load(ckpt_path)
        print(ckpt.keys())

        # Build DatasetUtil - General functionality to transform data
        data_util = DatasetUtil(ckpt["word_vocab"],
                                ckpt["char_vocab"],
                                ckpt["tag_vocab"])

        # Build model
        model = build_model(conf["model"],
                            ckpt["word_vocab"],
                            ckpt["char_vocab"],
                            ckpt["tag_vocab"])
        print("Loaded model from ", ckpt_path)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Read Synapse json
        synpase_file = open(synapse, "r").read()
        lines = synpase_file.split("}{")

        with open(output, 'w', encoding='utf-8') as output_file:
            for i, l in enumerate(tqdm(lines)):
                # Dumb parsing, due to the odd {cluster}{cluster} format
                # TODO: Check if this is a known format or standard.
                # Any line that isnt the first will have a missing { a the start
                if i != 0:
                    l = "{" + l
                # Any line that isnt the last one will have a missing } at the end
                if i != len(lines)-1:
                    l += "}"

                json_line = json.loads(l)

                tweets = json_line["_source"]["tweets"]

                for (i, tweet) in enumerate(tweets):
                    # Preprocess
                    clean_tweet = utils.clean(tweet["text"])
                    batch = data_util.fit(clean_tweet)
                    # Prediction
                    bin_out, ner_out = self.fw_pass(model, batch)

                    # Binary pred with value between [0:1]
                    bin_out = torch.sigmoid(
                        bin_out).detach().cpu().numpy()[0][0]

                    # NER with sequence of tags
                    ner_out = ner_out[0]
                    # Convert ID to corresponding String
                    ner_out = data_util.decode_tags(ner_out)[1:-1]
                    ner_out_text = " ".join(ner_out)

                    # Summarize
                    entities = {"Company": "",
                                "Asset": "",
                                "Threat": "",
                                "IDs": ""}
                    for (word, entity) in zip(clean_tweet.split(" "), ner_out):
                        if("PRO" in entity) and word not in entities["Asset"]:
                            entities["Asset"] = f"{entities['Asset']} {word}"
                        elif("VUL" in entity) and word not in entities["Threat"]:
                            entities["Threat"] = f"{entities['Threat']} {word}"
                        elif("ID" in entity) and word not in entities["IDs"]:
                            entities["IDs"] = f"{entities['IDs']} {word}"
                        elif("ORG" in entity) and word not in entities["Company"]:
                            entities["Company"] = f"{entities['Company']} {word}"

                    entities["Asset"] = entities["Asset"][1:]
                    entities["Threat"] = entities["Threat"][1:]
                    entities["IDs"] = entities["IDs"][1:]
                    entities["Company"] = entities["Company"][1:]

                    tweet["clean_text"] = clean_tweet
                    tweet["tags"] = ner_out_text
                    tweet["entities"] = entities
                    tweet["binary_pred_confidence"] = str(bin_out)
                    tweet["binary_pred"] = int(round(bin_out))

                    json_line["_source"]["tweets"][i] = tweet

                json.dump(json_line, output_file,
                          ensure_ascii=False, indent=4)


class ExampleClass:

    def fw_pass(self, model, padded_input):
        # convert into tensors
        for key, val in padded_input.items():
            padded_input[key] = model.Tensor(val).long()
        padded_input["mask"] = padded_input["word_ids"].data.gt(0).byte()
        # get predictions
        bin_out, ner_out = model(**padded_input)
        return bin_out, ner_out

    def __init__(self, conf, ckpt_path, input_file, output="output.json"):
        ckpt = torch.load(ckpt_path)

        # Build DatasetUtil - General functionality to transform data
        data_util = DatasetUtil(ckpt["word_vocab"],
                                ckpt["char_vocab"],
                                ckpt["tag_vocab"])

        # Build model
        model = build_model(conf["model"],
                            ckpt["word_vocab"],
                            ckpt["char_vocab"],
                            ckpt["tag_vocab"])
        print("Loaded model from ", ckpt_path)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        # Read Input file
        input_file = open(input_file, "r").read()
        # Parse file as needed ...

        with open(output, 'w', encoding='utf-8') as output_file:
            for i, l in enumerate(tqdm(input_file)):
                '''
                    Example workflow
                '''
                # Preprocess
                clean_tweet = utils.clean()  # pass text here
                batch = data_util.fit(clean_tweet)
                # Prediction
                bin_out, ner_out = self.fw_pass(model, batch)

                # Binary pred with value between [0:1]
                bin_out = torch.sigmoid(
                    bin_out).detach().cpu().numpy()[0][0]

                # NER with sequence of tags
                ner_out = ner_out[0]
                # Convert ID to corresponding String
                ner_out = data_util.decode_tags(ner_out)[1:-1]
                ner_out_text = " ".join(ner_out)


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(
        description='CyberThreat Tool <-> Multi-Task Classifier')
    parser.add_argument('-load', metavar='load', default=None,
                        help='Checkpoint location.')
    parser.add_argument('-save', metavar='save', type=utils.str2bool,
                        help='Boolean', default=False)
    parser.add_argument('-conf', metavar='conf', default="./conf/mt/rnn.json",
                        help='model configuration. JSON files defined in ./conf/')
    parser.add_argument('-input', metavar='input_file', default=None,
                        help='Pass an input file for inference')

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
        conf = json.load(open(args.conf, "r"))
        # Switch this to your Example class
        Synapse(conf, args.load, args.input)

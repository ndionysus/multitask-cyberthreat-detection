from bin_classifier import Main as Bin_Main
from ner_classifier import Main as Ner_Main
from mt_classifier import Main as Mt_Main

import utils
import json
import os
import argparse
from data import Dataset

'''
    BAREBONES GRID-SEARCH SCRIPT
'''

# Lists
vecs = [None]  # Check
dropouts = [0.0, 0.3, 0.5]
use_chars = [True]
c_dims = [50, 100]
c_cnn_layers = [
    [{"height": 2, "filters": 128, "dilation": 1}],
    [{"height": 2, "filters": 256, "dilation": 1}],
    [{"height": 3, "filters": 128, "dilation": 1}],
    [{"height": 3, "filters": 256, "dilation": 1}]
]
w_dims = [100, 200, 300]
w_cnn_layers = [
    [{"height": 2, "filters": 128, "dilation": 1}],
    [{"height": 3, "filters": 128, "dilation": 1}],
    [{"height": 2, "filters": 256, "dilation": 1}],
    [{"height": 3, "filters": 256, "dilation": 1}]
]
c_rnn_out = ["laststep", "maxpool"]
c_rnn_layers = [
    [{"cell": "lstm", "dim": 128, "n_layers": 1, "bidirectional": True}],
    [{"cell": "lstm", "dim": 256, "n_layers": 1, "bidirectional": True}]
]
w_rnn_out = ["laststep", "maxpool"]
w_rnn_layers = [
    [{"cell": "lstm", "dim": 128, "n_layers": 1, "bidirectional": True}],
    [{"cell": "lstm", "dim": 256, "n_layers": 1, "bidirectional": True}]
]
use_crf = [True]


def check_dir(d):
    if not os.path.isdir(d):
        os.mkdir(d)


def save_conf(conf, task, arch):
    dir = "results/"
    check_dir(dir)
    dir += task + "/"
    check_dir(dir)
    json.dump(conf, open(dir+arch+".json", "w"), indent=4)


def bin_cnn():
    counter = 0
    # Load Config file
    conf = json.load(open("conf/bin/cnn.json", "r"))
    conf["model"]["pretrained"] = None
    best = conf
    best["metrics"] = {"val": {"f1": 0}, "test": {}}
    # Load Dataset object
    dset = Dataset(batch_size=conf["train"]["batch_size"])
    for w in w_dims:
        conf["model"]["w_dim"] = w
        for wlayer in w_cnn_layers:
            conf["model"]["w_cnn_layers"] = wlayer
            conf["model"]["w_in_dropout"] = 0.0
            for mdrop in dropouts:
                conf["model"]["mid_dropout"] = mdrop
                conf["model"]["use_chars"] = True
                for c in c_dims:
                    conf["model"]["c_dim"] = c
                    for clayer in c_cnn_layers:
                        conf["model"]["c_cnn_layers"] = clayer
                        conf["model"]["c_in_dropout"] = 0.0
                        # Main training loop
                        results = Bin_Main(dset, conf).best
                        print(counter)
                        counter += 1
                        if results["val"]["f1"] > best["metrics"]["val"]["f1"]:
                            conf["metrics"] = results
                            best = conf
                            save_conf(conf, "bin", "cnn")
                            print(best)


def bin_rnn():
    counter = 0
    # Load Config file
    conf = json.load(open("conf/bin/rnn.json", "r"))
    conf["model"]["pretrained"] = None
    best = conf
    best["metrics"] = {"val": {"f1": 0}, "test": {}}
    dset = Dataset(batch_size=conf["train"]["batch_size"])
    for w in w_dims:
        conf["model"]["w_dim"] = w
        for wlayer in w_rnn_layers:
            conf["model"]["w_rnn_layers"] = wlayer
            for wout in w_rnn_out:
                conf["model"]["w_rnn_out"] = wout
                conf["model"]["w_in_dropout"] = 0.0
                for mdrop in dropouts:
                    conf["model"]["mid_dropout"] = mdrop
                    conf["model"]["use_chars"] = True
                    for cout in c_rnn_out:
                        conf["model"]["c_rnn_out"] = cout
                        for c in c_dims:
                            conf["model"]["c_dim"] = c
                            for clayer in c_rnn_layers:
                                conf["model"]["c_rnn_layers"] = clayer
                                conf["model"]["c_in_dropout"] = 0.0
                                # Main training loop
                                results = Bin_Main(dset, conf).best
                                print(counter)
                                counter += 1
                                if results["val"]["f1"] > best["metrics"]["val"]["f1"]:
                                    conf["metrics"] = results
                                    best = conf
                                    save_conf(conf, "bin", "rnn")
                                    print(best)


def bin_cnn_rnn():
    counter = 0
    # Load Config file
    conf = json.load(open("conf/bin/cnn_rnn.json", "r"))
    conf["model"]["pretrained"] = None
    best = conf
    best["metrics"] = {"val": {"f1": 0}, "test": {}}
    dset = Dataset(batch_size=conf["train"]["batch_size"])
    for w in w_dims:
        conf["model"]["w_dim"] = w
        for wlayer in w_rnn_layers:
            conf["model"]["w_rnn_layers"] = wlayer
            for wout in w_rnn_out:
                conf["model"]["w_rnn_out"] = wout
                conf["model"]["w_in_dropout"] = 0.0
                for mdrop in dropouts:
                    conf["model"]["mid_dropout"] = mdrop
                    conf["model"]["use_chars"] = True
                    for c in c_dims:
                        conf["model"]["c_dim"] = c
                        for clayer in c_cnn_layers:
                            conf["model"]["c_cnn_layers"] = clayer
                            conf["model"]["c_in_dropout"] = 0.0
                            # Main training loop
                            #results = Bin_Main(dset, conf).best
                            print(counter)
                            counter += 1
                            # if results["val"]["f1"] > best["metrics"]["val"]["f1"]:
                            #    conf["metrics"] = results
                            #    best = conf
                            #    save_conf(conf, "bin", "cnn_rnn")
                            #    print(best)


def bin_rnn_cnn():
    counter = 0
    # Load Config file
    conf = json.load(open("conf/bin/rnn_cnn.json", "r"))
    conf["model"]["pretrained"] = None
    best = conf
    best["metrics"] = {"val": {"f1": 0}, "test": {}}
    dset = Dataset(batch_size=conf["train"]["batch_size"])
    for w in w_dims:
        conf["model"]["w_dim"] = w
        for wlayer in w_cnn_layers:
            conf["model"]["w_cnn_layers"] = wlayer
            conf["model"]["w_in_dropout"] = 0.0
            for mdrop in dropouts:
                conf["model"]["mid_dropout"] = mdrop
                conf["model"]["use_chars"] = True
                for cout in c_rnn_out:
                    conf["model"]["c_rnn_out"] = cout
                    for c in c_dims:
                        conf["model"]["c_dim"] = c
                        for clayer in c_rnn_layers:
                            conf["model"]["c_rnn_layers"] = clayer
                            conf["model"]["c_in_dropout"] = 0.0
                            # Main training loop
                            results = Bin_Main(dset, conf).best
                            print(counter)
                            counter += 1
                            if results["val"]["f1"] > best["metrics"]["val"]["f1"]:
                                conf["metrics"] = results
                                best = conf
                                save_conf(conf, "bin", "rnn_cnn")
                                print(best)


def ner_rnn():
    counter = 0
    # Load Config file
    conf = json.load(open("conf/ner/rnn.json", "r"))
    conf["model"]["pretrained"] = None
    best = conf
    best["metrics"] = {"val": {"f1": 0}, "test": {}}
    dset = Dataset(batch_size=conf["train"]["batch_size"])
    for w in w_dims:
        conf["model"]["w_dim"] = w
        for wlayer in w_rnn_layers:
            conf["model"]["w_rnn_layers"] = wlayer
            conf["model"]["w_rnn_out"] = None
            conf["model"]["w_in_dropout"] = 0.0
            for mdrop in dropouts:
                conf["model"]["mid_dropout"] = mdrop
                conf["model"]["use_chars"] = True
                conf["model"]["use_crf"] = True
                for cout in c_rnn_out:
                    conf["model"]["c_rnn_out"] = cout
                    for c in c_dims:
                        conf["model"]["c_dim"] = c
                        for clayer in c_rnn_layers:
                            conf["model"]["c_rnn_layers"] = clayer
                            conf["model"]["c_in_dropout"] = 0.0
                            # Main training loop
                            results = Ner_Main(dset, conf).best
                            print(counter)
                            counter += 1
                            if results["val"]["f1"] > best["metrics"]["val"]["f1"]:
                                conf["metrics"] = results
                                best = conf
                                save_conf(conf, "ner", "rnn")
                                print(best)


def ner_cnn():
    # Load Config file
    conf = json.load(open("conf/ner/cnn.json", "r"))
    conf["model"]["pretrained"] = None
    best = conf
    best["metrics"] = {"val": {"f1": 0}, "test": {}}
    dset = Dataset(batch_size=conf["train"]["batch_size"])
    for w in w_dims:
        conf["model"]["w_dim"] = w
        for wlayer in w_cnn_layers:
            conf["model"]["w_cnn_layers"] = wlayer
            conf["model"]["w_in_dropout"] = 0.0
            for mdrop in dropouts:
                conf["model"]["mid_dropout"] = mdrop
                conf["model"]["use_chars"] = True
                conf["model"]["use_crf"] = True
                for c in c_dims:
                    conf["model"]["c_dim"] = c
                    for clayer in c_cnn_layers:
                        conf["model"]["c_cnn_layers"] = clayer
                        conf["model"]["c_in_dropout"] = 0.0
                        # Main training loop
                        results = Ner_Main(dset, conf).best
                        if results["val"]["f1"] > best["metrics"]["val"]["f1"]:
                            conf["metrics"] = results
                            best = conf
                            save_conf(conf, "ner", "cnn")
                            print(best)


def ner_cnn_rnn():
    # Load Config file
    conf = json.load(open("conf/ner/cnn_rnn.json", "r"))
    conf["model"]["pretrained"] = None
    best = conf
    best["metrics"] = {"val": {"f1": 0}, "test": {}}
    dset = Dataset(batch_size=conf["train"]["batch_size"])
    for w in w_dims:
        conf["model"]["w_dim"] = w
        for wlayer in w_rnn_layers:
            conf["model"]["w_rnn_layers"] = wlayer
            conf["model"]["w_rnn_out"] = None
            conf["model"]["w_in_dropout"] = 0.0
            for mdrop in dropouts:
                conf["model"]["mid_dropout"] = mdrop
                conf["model"]["use_chars"] = True
                conf["model"]["use_crf"] = True
                for c in c_dims:
                    conf["model"]["c_dim"] = c
                    for clayer in c_cnn_layers:
                        conf["model"]["c_cnn_layers"] = clayer
                        conf["model"]["c_in_dropout"] = 0.0
                        # Main training loop
                        results = Ner_Main(dset, conf).best
                        if results["val"]["f1"] > best["metrics"]["val"]["f1"]:
                            conf["metrics"] = results
                            best = conf
                            save_conf(conf, "ner", "cnn_rnn")
                            print(best)


def ner_rnn_cnn():
    # Load Config file
    conf = json.load(open("conf/ner/rnn_cnn.json", "r"))
    conf["model"]["pretrained"] = None
    best = conf
    best["metrics"] = {"val": {"f1": 0}, "test": {}}
    dset = Dataset(batch_size=conf["train"]["batch_size"])
    for w in w_dims:
        conf["model"]["w_dim"] = w
        for wlayer in w_cnn_layers:
            conf["model"]["w_cnn_layers"] = wlayer
            conf["model"]["w_in_dropout"] = 0.0
            for mdrop in dropouts:
                conf["model"]["mid_dropout"] = mdrop
                conf["model"]["use_chars"] = True
                conf["model"]["use_crf"] = True
                for c in c_dims:
                    conf["model"]["c_dim"] = c
                    for clayer in c_rnn_layers:
                        conf["model"]["c_rnn_layers"] = clayer
                        for cout in c_rnn_out:
                            conf["model"]["c_rnn_out"] = cout
                            conf["model"]["c_in_dropout"] = 0.0
                            # Main training loop
                            results = Ner_Main(dset, conf).best
                            if results["val"]["f1"] > best["metrics"]["val"]["f1"]:
                                conf["metrics"] = results
                                best = conf
                                save_conf(conf, "ner", "rnn_cnn")
                                print(best)


def mt_rnn():
    # Load Config file
    conf = json.load(open("conf/mt/rnn.json", "r"))
    conf["model"]["pretrained"] = None
    conf["model"]["use_chars"] = True
    conf["model"]["w_in_dropout"] = 0.0
    conf["model"]["use_crf"] = True
    conf["model"]["w_rnn_out"] = None
    conf["model"]["c_in_dropout"] = 0.0
    best = conf
    best["metrics"] = {"val": {"f1": 0}, "test": {}}
    dset = Dataset(batch_size=conf["train"]["batch_size"])
    for w in w_dims:
        conf["model"]["w_dim"] = w
        for wlayer in w_rnn_layers:
            conf["model"]["w_rnn_layers"] = wlayer
            for wout in w_rnn_out:
                conf["model"]["w_bin_out"] = wout
                for mdrop in dropouts:
                    conf["model"]["mid_dropout"] = mdrop
                    for cout in c_rnn_out:
                        conf["model"]["c_rnn_out"] = cout
                        for c in c_dims:
                            conf["model"]["c_dim"] = c
                            for clayer in c_rnn_layers:
                                conf["model"]["c_rnn_layers"] = clayer
                                # Main training loop
                                results = Mt_Main(dset, conf).best
                                if results["val"]["f1"] > best["metrics"]["val"]["f1"]:
                                    conf["metrics"] = results
                                    best = conf
                                    save_conf(conf, "mt", "rnn")
                                    print(best)


def mt_cnn():
    # Load Config file
    conf = json.load(open("conf/mt/cnn.json", "r"))
    conf["model"]["pretrained"] = None
    best = conf
    best["metrics"] = {"val": {"f1": 0}, "test": {}}
    dset = Dataset(batch_size=conf["train"]["batch_size"])
    for w in w_dims:
        conf["model"]["w_dim"] = w
        for wlayer in w_cnn_layers:
            conf["model"]["w_cnn_layers"] = wlayer
            conf["model"]["w_in_dropout"] = 0.0
            for wout in w_rnn_out:
                conf["model"]["w_bin_out"] = wout
                for mdrop in dropouts:
                    conf["model"]["mid_dropout"] = mdrop
                    conf["model"]["use_chars"] = True
                    conf["model"]["use_crf"] = True
                    for c in c_dims:
                        conf["model"]["c_dim"] = c
                        for clayer in c_cnn_layers:
                            conf["model"]["c_cnn_layers"] = clayer
                            conf["model"]["c_in_dropout"] = 0.0
                            # Main training loop
                            results = Mt_Main(dset, conf).best
                            if results["val"]["f1"] > best["metrics"]["val"]["f1"]:
                                conf["metrics"] = results
                                best = conf
                                save_conf(conf, "mt", "cnn")
                                print(best)


def mt_cnn_rnn():
    # Load Config file
    conf = json.load(open("conf/mt/cnn_rnn.json", "r"))
    conf["model"]["pretrained"] = None
    conf["model"]["w_in_dropout"] = 0.0
    conf["model"]["use_chars"] = True
    conf["model"]["use_crf"] = True
    conf["model"]["c_in_dropout"] = 0.0
    best = conf
    best["metrics"] = {"val": {"f1": 0}, "test": {}}
    dset = Dataset(batch_size=conf["train"]["batch_size"])
    for w in w_dims:
        conf["model"]["w_dim"] = w
        for wlayer in w_rnn_layers:
            conf["model"]["w_rnn_layers"] = wlayer
            for wout in w_rnn_out:
                conf["model"]["w_bin_out"] = wout
                for mdrop in dropouts:
                    conf["model"]["mid_dropout"] = mdrop
                    for c in c_dims:
                        conf["model"]["c_dim"] = c
                        for clayer in c_cnn_layers:
                            conf["model"]["c_cnn_layers"] = clayer
                            # Main training loop
                            results = Mt_Main(dset, conf).best
                            if results["val"]["f1"] > best["metrics"]["val"]["f1"]:
                                conf["metrics"] = results
                                best = conf
                                save_conf(conf, "mt", "cnn_rnn")
                                print(best)


def mt_rnn_cnn():
    # Load Config file
    conf = json.load(open("conf/mt/rnn_cnn.json", "r"))
    conf["model"]["pretrained"] = None
    conf["model"]["w_in_dropout"] = 0.0
    conf["model"]["use_chars"] = True
    conf["model"]["use_crf"] = True
    conf["model"]["c_in_dropout"] = 0.0
    best = conf
    best["metrics"] = {"val": {"f1": 0}, "test": {}}
    dset = Dataset(batch_size=conf["train"]["batch_size"])
    for w in w_dims:
        conf["model"]["w_dim"] = w
        for wlayer in w_cnn_layers:
            conf["model"]["w_cnn_layers"] = wlayer
            for wout in w_rnn_out:
                conf["model"]["w_bin_out"] = wout
                for mdrop in dropouts:
                    conf["model"]["mid_dropout"] = mdrop
                    for c in c_dims:
                        conf["model"]["c_dim"] = c
                        for clayer in c_rnn_layers:
                            conf["model"]["c_rnn_layers"] = clayer
                            # Main training loop
                            results = Mt_Main(dset, conf).best
                            if results["val"]["f1"] > best["metrics"]["val"]["f1"]:
                                conf["metrics"] = results
                                best = conf
                                save_conf(conf, "mt", "rnn_cnn", inf)
                                print(best)


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(
        description='CyberThreat Tool <-> Grid Search')
    parser.add_argument('-arch', metavar='arch', help='Net', default="cnn")
    parser.add_argument('-task', metavar='task', help='Task', default="bin")

    args = parser.parse_args()

    # Set Seed for reproducibility
    utils.set_seed()

    if args.task == "bin":
        if args.arch == "cnn":
            bin_cnn()
        elif args.arch == "rnn":
            bin_rnn()
        elif args.arch == "cnn_rnn":
            bin_cnn_rnn()
        elif args.arch == "rnn_cnn":
            bin_rnn_cnn()
        else:
            raise Exception("Undefined %s" % args.arch)

    elif args.task == "ner":
        if args.arch == "rnn":
            ner_rnn()
        elif args.arch == "cnn":
            ner_cnn()
        elif args.arch == "cnn_rnn":
            ner_cnn_rnn()
        elif args.arch == "rnn_cnn":
            ner_rnn_cnn()
        else:
            raise Exception("Undefined %s" % args.arch)

    elif args.task == "mt":
        if args.arch == "rnn":
            mt_rnn()
        elif args.arch == "cnn":
            mt_cnn()
        elif args.arch == "cnn_rnn":
            mt_cnn_rnn()
        elif args.arch == "rnn_cnn":
            mt_rnn_cnn()
        else:
            raise Exception("Undefined %s" % args.arch)

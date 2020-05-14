import torch
import numpy as np
import pandas as pd
import re

from sklearn import metrics

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)


def set_seed(seed=2008):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise Exception('Boolean value expected.')


def clean(string):
    def rm_space(string):
        exit = False
        while(not exit):
            if "  " in string:
                string = string.replace("  ", " ")
            else:
                exit = True
        if string.startswith(" "):
            string = string[1:]
        return string

    string = string.strip().lower()
    # other cases where special chars that we want to keep appear in odd places
    string = string.replace("...", " ")
    string = string.replace(" . ", " ")
    string = string.replace(". ", " ")
    string = string.replace(" - ", " ")
    string = string.replace(" -", " ")
    string = string.replace("- ", " ")
    string = string.replace(" : ", " ")
    string = string.replace(": ", " ")
    string = string.replace(" :", " ")
    string = string.replace(" _ ", " ")
    string = string.replace("_ ", " ")
    string = string.replace(" _", " ")

    new_string = []
    string = string.split(" ")
    for word in string:
        if not ("https://" in word) and (not "http://" in word):
            word = re.sub(r"[^A-Za-z0-9.\-:_]", " ", word)
            new_string.append(word)
    string = rm_space(" ".join(new_string))
    return string


# ======= #
# BINARY
# ======= #
def bin_fw_eval(model, fw, set, step):
    with torch.no_grad():
        # Store predictions
        val_x, val_y = np.asarray([]), np.asarray([])
        # iterate through the val batch
        for batch in set:
            # forward pass
            x, y, mask = fw(model, batch)
            # apply a sigmoid function and flatten predictions
            x = torch.sigmoid(x).cpu().detach().numpy()
            x = np.reshape(x, (-1))
            y = np.reshape(y.cpu(), (-1))
            # append results
            val_x = np.concatenate((val_x, x), -1)
            val_y = np.concatenate((val_y, y), -1)
        # Compute metrics
        acc, tpr, tnr, f1 = eval_binary(val_x, val_y)
        output = {"acc": acc, "tpr": tpr, "tnr": tnr, "f1": f1}

        print("\t\t" + "_"*80 + "\n")
        # Print Metrics
        print("\t\tSTEP:", step,
              "\tACC:", format(acc, ".5f"),
              "\tTPR:", format(tpr, ".5f"),
              "\tTNR:", format(tnr, ".5f"),
              "\tF1: ", format(f1, ".5f"))

    return output


def eval_binary(scores, target):
    # Round the predictions
    scores = np.around(scores)
    # Confusion Matrix
    tn, fp, fn, tp = metrics.confusion_matrix(target, scores).ravel()
    acc = (tp+tn)/(tn + fp + fn + tp)
    tpr = tp / (tp+fn)
    tnr = tn / (tn+fp)
    #pre = tp / (tp+fp)
    f1 = (2*tpr*tnr)/(tpr+tnr)
    return acc, tpr, tnr, f1


# ======= #
# NER
# ======= #

def ner_fw_eval(model, fw, data, step, tags):
    with torch.no_grad():
        # Store predictions
        val_x, val_y = np.asarray([]), np.asarray([])

        for batch in data:
            # forward pass
            x, y, mask = fw(model, batch)
            mask = mask.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            for i, (p, r, m) in enumerate(zip(x, y, mask)):
                r = r[:sum(m)]
                p = p[:sum(m)]
                val_x = np.concatenate((val_x, p), -1)
                val_y = np.concatenate((val_y, r), -1)

        # Compute metrics
        acc, p, r, f1 = eval_ner(val_x, val_y, tags)
        output = {"acc": acc, "p": p, "r": r, "f1": f1}
        # Print Metrics
        print("\t\t" + "_"*80 + "\n")
        print("\t\tSTEP:", step,
              "\tACC:", format(acc, ".5f"),
              "\tPRE:", format(p, ".5f"),
              "\tREC:", format(r, ".5f"),
              "\tF1: ", format(f1, ".5f"))

    return output


def eval_ner(scores, target, ign_ids):
    mask = np.asarray([t in ign_ids for t in target], dtype=bool)
    target = target[~mask]
    scores = scores[~mask]
    acc = metrics.accuracy_score(scores, target)
    p = metrics.precision_score(scores, target, average="weighted")
    r = metrics.recall_score(scores, target, average="weighted")
    f1 = (2*p*r)/(p+r)
    return acc, p, r, f1

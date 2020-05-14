import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
'''
GENERIC NN CLASS THAT HOLDS THE COMMON COMPONENTS ACROSS NNs
'''


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        # GPU
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.ts = datetime.now().strftime("%Y%m%d%H%M%S")
        print("\nDEVICE(s) : ", self.device)

    def Tensor(self, data):
        # Helper function to turn an array into a tensor
        return torch.Tensor(data).to(self.device)

    def save(self, step, conf, metrics, opt, lr_sch, task, path="./ckpts"):
        # Check if folders exists
        if not os.path.isdir(path):
            os.makedirs(path)
        path += "/%s" % task
        if not os.path.isdir(path):
            os.makedirs(path)
        self.dir = "{}/{}_{}/".format(path, conf["model"]["arch"], self.ts)
        if not os.path.isdir(self.dir):
            os.makedirs(self.dir)
        ckpt = self.dir+"ckpt_step-{0}.pth".format(step)

        # Build save dictionary
        save_dict = dict()
        save_dict["step"] = step
        save_dict["model_state"] = self.state_dict()
        # If Multi-task, we have two Optimizers and two Learning Sch
        if task == "mt":
            save_dict["bin_optimizer_state"] = opt[0].get_state_dict()
            save_dict["bin_lr_scheduler"] = lr_sch[0].get_state_dict()
            save_dict["ner_optimizer_state"] = opt[1].get_state_dict()
            save_dict["ner_lr_scheduler"] = lr_sch[1].get_state_dict()
        else:
            save_dict["optimizer_state"] = opt.get_state_dict()
            save_dict["lr_scheduler"] = lr_sch.get_state_dict()
        torch.save(save_dict, ckpt)
        # Delete last ckpt
        try:
            os.remove(self.ckpt)
            self.ckpt = ckpt
        except AttributeError:
            # First checkpoint
            self.ckpt = ckpt
            # Save the config.json containing the model defined and the variables to use
            json.dump(conf, open(self.dir+"conf.json", "w"), indent=2)
        json.dump(metrics, open(self.dir+"results.json", "w"), indent=2)

    def load(self, path, opt=None):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if opt is not None:
            opt.load_state_dict(checkpoint['optimizer_state_dict'])

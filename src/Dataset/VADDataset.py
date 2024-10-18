import os, glob
import torch
import numpy as np

# Dataset for VVAD-LRS3
class VADDataset(torch.utils.data.Dataset):
    def __init__(self, hp, is_train=True):
        self.is_train  = is_train
        self.root_speech = hp.data.speech
        self.root_noise = hp.data.noise
                                              
    def __getitem__(self, index):
        # Determine label

        # if speech, mix

        # scaling

        # else only noise

        data = {}
        data["input"] = noisy
        data["target"] = clean
        return data

    def __len__(self):
        if self.is_train : 
            return len(self.dataset["y_train"])
        else : 
            return len(self.dataset["y_test"])



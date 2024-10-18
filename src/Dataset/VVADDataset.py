import os, glob
import torch
import h5py
import cv2
import numpy as np
import torchvision.transforms as transforms

# Dataset for VVAD-LRS3
class VVADDataset(torch.utils.data.Dataset):
    def __init__(self, hp, is_train=True):
        self.dataset = h5py.File(hp.data.VVAD, "r")
        self.is_train  = is_train

        self.T = 38

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 랜덤 좌우 반전
            transforms.RandomRotation(30),      # 랜덤 회전 (각도 범위)
            transforms.ToTensor(),              # 텐서 변환
        ])

    def __getitem__(self, index):
        # (38, 96, 96, 3)
        if self.is_train : 
            input= self.dataset["x_train"][index]
            label = self.dataset["y_train"][index]
        else :
            input = self.dataset["x_test"][index]
            label = self.dataset["y_test"][index]

        # to gray scale
        plate = np.zeros(input.shape[:3])
        labels = np.zeros((self.T))
        labels = labels + label

        for i in range(plate.shape[0]) : 
            plate[i] = cv2.cvtColor(input[i], cv2.COLOR_BGR2GRAY)
        plate = plate/255.0
        plate = torch.tensor(plate).float()
        labels = torch.tensor(labels).float()

        data = {}
        data["input"] = plate
        data["target"] = labels

        return data

    def __len__(self):
        if self.is_train : 
            return len(self.dataset["y_train"])
        else : 
            return len(self.dataset["y_test"])



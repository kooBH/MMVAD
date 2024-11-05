import os, glob
import torch
import h5py
import cv2
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.v2 as v2
from PIL import Image
import random

def video_transform(frames, p = 0.5):
    # 0.5 확률로 영상 전체에 대한 플립 여부 결정
    if random.random() > p:
        # 모든 프레임에 동일한 좌우 반전 적용
        return [transforms.functional.hflip(frame) for frame in frames]
    else:
        # 좌우 반전 미적용
        return frames

def video_rotate(frames, p = 0.5,angle = 30):
    if random.random() > p : 
        # 회전 각도 설정 (예: -30도에서 30도 사이)
        angle = random.uniform(-angle, angle)
        
        # 모든 프레임에 동일한 각도로 회전 적용
        rotated_frames = [transforms.functional.rotate(frame, angle) for frame in frames]
        return rotated_frames
    else :
        return frames

# Dataset for VVAD-LRS3
class VVADDataset(torch.utils.data.Dataset):
    def __init__(self, hp, is_train=True):

        self.dataset = h5py.File(hp.data.VVAD, "r")

        self.is_train  = is_train

        self.hp = hp

        self.T = 38
        self.Gray = transforms.Grayscale(num_output_channels=1)
        self.PIL2Tensor = transforms.ToTensor()
        self.gaussian_noise = transforms.v2.GaussianNoise(0.1)
        self.sharpen = v2.RandomAdjustSharpness(sharpness_factor=2,p = hp.aug.sharpen_p)
        self.contrast = v2.RandomAutocontrast(p = hp.aug.contrast_p)
        self.equalize = v2.RandomEqualize(p = hp.aug.equalize_p)
        self.blur = v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.))
        self.perspective = v2.RandomPerspective(distortion_scale=0.6, p=hp.aug.perspective_p)

    def PreProcess(self, frames, is_train=True):
        PIL_frames = [Image.fromarray(frame) for frame in frames]

        # Require PIL image
        if is_train : 
            PIL_frames = video_transform(PIL_frames,
                                         p =self.hp.aug.flip_p)
            PIL_frames = video_rotate(PIL_frames, 
                                    p = self.hp.aug.rotate_p,
                                    angle = self.hp.aug.rotate_angle)
        PIL_frames = [self.Gray(frame) for frame in PIL_frames]
        frames = [self.PIL2Tensor(frame) for frame in PIL_frames]
        frames = torch.stack(frames, dim=0)

        # Do not support PIL images
        if is_train : 
            if random.random() > self.hp.aug.gaussian_p:
                frames = self.gaussian_noise(frames)
            frames = self.sharpen(frames)
            frames = self.contrast(frames)
            frames = self.equalize(frames)
            if random.random() > self.hp.aug.blur_p:
                frames = self.blur(frames)
            frames = self.perspective(frames)

        frames = torch.squeeze(frames,1)
        return frames

    def __getitem__(self, index):
        # (38, 96, 96, 3)
        if self.is_train : 
            input= self.dataset["x_train"][index]
            label = self.dataset["y_train"][index]
            input = self.PreProcess(input)
        else :
            input = self.dataset["x_test"][index]
            label = self.dataset["y_test"][index]
            input = self.PreProcess(input,is_train=False)

        labels = np.zeros((self.T))
        labels = labels + label
        labels = torch.tensor(labels).float()

        data = {}
        data["input"] = input
        data["target"] = labels

        return data

    def __len__(self):
        if self.is_train : 
            return len(self.dataset["y_train"])
        else : 
            return len(self.dataset["y_test"])



import os, glob
import torch
import numpy as np
import librosa as rs
import random

from Dataset.KoreaSpeechDataset import KoreaSpeechDataset
from Dataset.AVA_Speech_Dataset import AVASpeechDataset
from torch.utils.data import ConcatDataset
from os.path import join

def get_list(item,format) : 
    list_item = []
    if type(item) is str :
        list_item = glob.glob(join(item,"**",format),recursive=True)
    elif type(item) is list :
        for i in item : 
            list_item += glob.glob(join(i,"**",format),recursive=True)
    return list_item

class AVADDataset(torch.utils.data.Dataset):
    def __init__(self, hp, is_train=True):
        self.hp = hp.data
        self.is_train  = is_train
        self.sr = hp.audio.sr
        self.max_len = hp.data.max_len
        self.hop_length = hp.data.hop_length
        self.len_data = hp.data.len_data

        if is_train : 
            self.Dataset_AVA = AVASpeechDataset(hp.data.AVA_train,sample_rate = self.sr,max_len = self.max_len,hop_length = self.hop_length)
            self.Dataset_Korea = KoreaSpeechDataset(hp.data.Korea,sample_rate = self.sr,max_len = self.max_len,hop_length = self.hop_length)
            self.prob_speech = hp.data.prob_speech

            # https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.ConcatDataset
            self.combined_clean = ConcatDataset([self.Dataset_AVA, self.Dataset_Korea])
            self.combined_noise = get_list(hp.data.noise,"*.wav")

        # AVA-Speech validation set with 4 videos
        else : 
            self.combined_noisy = AVASpeechDataset(hp.data.AVA_dev,sample_rate = self.sr,max_len = self.max_len,hop_length = self.hop_length)

    def __getitem__(self, index):
        if not self.is_train :
            audio, label, is_clean = self.combined_noisy[index]
            if audio.dim() ==2:
                audio = audio.squeeze(0)
            return audio,label
        else :
            # speech
            if np.random.rand() < self.prob_speech : 
                audio, label, is_clean = self.combined_clean[index]

                # clean -> noisy 
                if is_clean : 
                    audio = self.mix_noisy(audio)
            # noise only
            else :
                audio, label = self.get_noise()

            if audio.dim() ==2:
                audio = audio.squeeze(0)


            # scaling
            if self.hp.scale.use:
                peak_val = torch.max(torch.abs(audio))
                audio = audio / (peak_val + 1e-8)

                scale = torch.empty(1, device=audio.device).uniform_(self.hp.scale.min, self.hp.scale.max)
                audio = audio * scale
                

            # Augmentation
            return audio,label

    def __len__(self):
        if not self.is_train :
            return len(self.combined_noisy)
        else :
            return len(self.combined_clean)
    
    def get_noise(self):
        path_audio = random.choice(self.combined_noise)
        audio, _ = rs.load(path_audio,sr=self.sr,mono=True)
        audio, _ = self.match_length(audio)
        label = np.zeros(self.max_len // self.hop_length,dtype=np.float32)
        return torch.from_numpy(audio), torch.from_numpy(label)
    
    def mix_noisy(self,clean):

        path_noise = np.random.choice(self.combined_noise)
        noise = rs.load(path_noise,sr=self.sr,mono=True)[0]

        noise, _ = self.match_length(noise)
        noise = torch.from_numpy(noise).float()

        SNR = np.random.uniform(self.hp.SNR[0],self.hp.SNR[1])

        rms_clean = torch.sqrt(torch.mean(clean**2))
        rms_noise = torch.sqrt(torch.mean(noise**2))
        rms_noise_target = rms_clean / (10**(SNR/20))
        noise = noise * (rms_noise_target / (rms_noise + 1e-8))
        noisy = clean + noise

        return noisy

    def match_length(self,wav,idx_start=None) :
        if len(wav) > self.max_len :
            left = len(wav) - self.max_len
            if idx_start is None :
                idx_start = np.random.randint(left)
            wav = wav[idx_start:idx_start+self.max_len]
        elif len(wav) < self.max_len :
            shortage = self.max_len - len(wav) 
            wav = np.pad(wav,(0,shortage),mode="wrap")
        return wav, idx_start
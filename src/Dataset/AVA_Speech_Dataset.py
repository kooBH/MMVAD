# AVA-Speech Dataset
# https://research.google.com/ava/download.html#ava_speech_download

import os
import torch
import torchaudio
from torch.utils.data import Dataset

class AVASpeechDataset(Dataset):
    """
    AVA-Speech Dataset for VAD task.
    Labels are assigned based on directory names.
    0: No Speech, 1: Speech (Clean, with Noise, or with Music)
    """
    def __init__(self, root_dir, sample_rate=16000, max_len=32000, hop_length = 128, transform=None):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.hop_length = hop_length
        self.transform = transform
        
        self.file_list = []
        self.labels = []
        
        # Mapping directory to binary VAD label
        # 0: Absent, 1: Present
        self.label_map = {
            'NO_SPEECH': 0,
            'CLEAN_SPEECH': 1,
            'SPEECH_WITH_NOISE': 1,
            'SPEECH_WITH_MUSIC': 1
        }
        self.is_clean = False
        
        self._prepare_dataset()
        
        print(f"AVASpeechDataset : {len(self.file_list)}")

    def _prepare_dataset(self):
        """Scan directories and build file list with labels."""
        for folder_name, label in self.label_map.items():
            folder_path = os.path.join(self.root_dir, folder_name)
            if not os.path.exists(folder_path):
                continue
                
            for file in os.listdir(folder_path):
                if file.endswith(('.wav', '.flac', '.mp3')):
                    self.file_list.append(os.path.join(folder_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]

        label_pt = torch.zeros(self.max_len // self.hop_length, dtype=torch.float32)
        if label == 1:
            label_pt[:] = 1.0
        
        # Load audio file
        waveform, sr = torchaudio.load(file_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Standardize length (Padding or Cropping)
        if waveform.shape[1] > self.max_len:
            waveform = waveform[:, :self.max_len]
        elif waveform.shape[1] < self.max_len:
            padding = self.max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        if self.transform:
            waveform = self.transform(waveform)
            
        return waveform, label_pt, self.is_clean

# Implementation Note:
# 1. Labels are simplified to 0 (No speech) and 1 (Speech).
# 2. Audio duration is fixed to ensure batch processing.
# 3. Use return_convergence_delta=True if verifying with IG/SHAP later.
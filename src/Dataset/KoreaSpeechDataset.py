# AI-Hub Korea Speech Dataset

import torch
import librosa
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class KoreaSpeechDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, max_len=32000, hop_length=128, top_db=30):
        """
        Args:
            root_dir (str): Root directory for wav files.
            sample_rate (int): Target sampling rate.
            max_len (int): Total audio samples to load (fixed length).
            hop_length (int): Number of samples between successive frames for labeling.
            top_db (int): Energy threshold for librosa.effects.split.
        """
        self.root_dir = root_dir
        self.sr = sample_rate
        self.max_len = max_len
        self.hop_length = hop_length
        self.top_db = top_db
        
        # Calculate expected label length
        self.label_len = self.max_len // self.hop_length
        
        # Recursive search
        self.file_list = list(Path(root_dir).rglob("*.wav"))
        
        if not self.file_list:
            raise FileNotFoundError(f"No wav files found in {root_dir}")
        
        self.is_clean = True
        print(f"KoreaSpeechDataset : {len(self.file_list)}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = str(self.file_list[idx])
        
        # 1. Load audio with fixed length
        # Using duration to limit loading for efficiency
        audio, _ = librosa.load(file_path, sr=self.sr, mono=True)
        
        # 2. Pad or Truncate audio to max_len
        if len(audio) > self.max_len:
            audio = audio[:self.max_len]
        else:
            audio = np.pad(audio, (0, self.max_len - len(audio)), mode='wrap')
            
        # 3. Detect voice intervals from the fixed-length audio
        # intervals contains [start_sample, end_sample]
        intervals = librosa.effects.split(audio, top_db=self.top_db)
        
        # 4. Create Frame-level Labels
        # Initialize labels with 0
        label = np.zeros(self.label_len, dtype=np.float32)
        
        for start_sample, end_sample in intervals:
            # Convert sample indices to frame indices
            start_frame = start_sample // self.hop_length
            end_frame = end_sample // self.hop_length
            
            # Ensure indices do not exceed label_len
            start_frame = min(start_frame, self.label_len)
            end_frame = min(end_frame, self.label_len)
            
            label[start_frame:end_frame] = 1.0
            
        return torch.from_numpy(audio), torch.from_numpy(label), self.is_clean

# Example Usage
if __name__ == "__main__":
    # Parameters for 16kHz audio, 2 seconds long, 8ms hop size (128 samples)
    # dataset = KoreaSpeechDataset("./data", sample_rate=16000, max_len=32000, hop_length=128)
    # audio, label = dataset[0]
    # print(f"Audio shape: {audio.shape}") # Expected: [32000]
    # print(f"Label shape: {label.shape}") # Expected: [250] (32000 / 128)
    pass
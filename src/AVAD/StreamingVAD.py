import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import numpy as np

class StreamingConvBlock(nn.Module):
    """
    CNN Block optimized for frame-wise feature extraction.
    """
    def __init__(self, cin: int, cout: int, kernel_size: Union[int, tuple] = 3):
        super().__init__()
        # Use (1, K) kernel to avoid temporal look-ahead in CNN
        # If input is a single frame, height(time) becomes 1.
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=kernel_size, padding=(0, 1), bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class StreamingVAD(nn.Module):
    def __init__(
        self, 
        input_dim: int = 257, 
        feature_dim : int = 32,
        output_dim: int = 1, 
        hidden_dim: int = 256,
        num_layers: int = 2,
        compression : float = 0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.compression = compression

        self.linear = nn.Linear(input_dim, feature_dim)

        # 2D CNN to extract frequency-domain features per frame
        # Pooling is only applied to the frequency axis (dim 3)
        self.features = nn.Sequential(
            StreamingConvBlock(1, 32, kernel_size=(1, 3)),
            nn.MaxPool2d(kernel_size=(1, 2)), 
            StreamingConvBlock(32, 64, kernel_size=(1, 3)),
            nn.MaxPool2d(kernel_size=(1, 2)),
            StreamingConvBlock(64, 128, kernel_size=(1, 3)),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.1)
        )

        # Calculate RNN input dimension (Frequency axis reduction)
        # 32 -> 16 -> 8 -> 4 (after 3 maxpools)
        reduced_freq = feature_dim // 8
        self.rnn_input_dim = 128 * reduced_freq

        # Unidirectional GRU for Streaming
        self.gru = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False # Essential for streaming
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, f,_ = x.shape

        x = torch.sqrt(x[...,0]**2 + x[...,1]**2 + 1e-7)
        x = x.pow_(self.compression)
        x = x.unsqueeze(1)  # [B,1,T,F]
        


        # [B,C,T,input_dim] -> [B,C,T,feature_dim]
        x = self.linear(x) 
        
        # 1. Feature Extraction
        # CNN works on each frame independently in the time axis
        x = self.features(x) # [B, C', T, F']
        
        # 2. Reshape for RNN
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, t, -1) # [B, T, C' * F']
        
        # 3. RNN step
        # h_prev maintains the memory of past frames
        x, h_new = self.gru(x, h_prev)
        
        # 4. Output
        decision = torch.sigmoid(self.output_layer(x))
        decision = decision.squeeze(-1)  # [B, T]
        
        return decision, h_new

    def init_state(self, batch_size: int, device: torch.device):
        """Initialize hidden state with zeros."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
    
class StreamingVAD_helper(nn.Module):
    def __init__(self, hp):
        super().__init__()

        self.frame_size = hp.audio.n_fft
        self.hop_size = hp.audio.n_hop

        n_freq = self.frame_size // 2 + 1

        self.model = StreamingVAD(
            input_dim=n_freq,
            feature_dim = hp.model.feature_dim,
            output_dim=1,
            hidden_dim=hp.model.hidden_dim,
            num_layers=hp.model.num_layers
        )

        window = torch.zeros(self.frame_size)
        if self.frame_size // self.hop_size == 4 : 
            n = torch.arange(self.frame_size)
            window = 0.5 * (1.0 - torch.cos(2.0 * np.pi * n / self.frame_size))
            
            energy_sum = torch.sum(window ** 2) / self.hop_size
            window /= torch.sqrt(energy_sum)
        elif self.frame_size // self.hop_size == 2 :
            n = torch.arange(self.frame_size)
            window = torch.sin(np.pi * (n + 0.5) / self.frame_size)
            energy_sum = torch.sum(window ** 2) / self.hop_size
            window /= torch.sqrt(energy_sum)
        else :
            raise RuntimeError(f"Not supported frame_size // hop_size {self.frame_size}//{self.hop_size}")
        
        self.register_buffer('window', window)

    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        x = torch.stft(
            x,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            win_length=self.frame_size,
            window=self.window,
            return_complex=True
        )

        # x : complex [B, F, T] -> real[B, T,F,2]
        x = torch.stack((x.real, x.imag), dim=-1).permute(0,2,1,3)

        return self.model(x, h_prev)
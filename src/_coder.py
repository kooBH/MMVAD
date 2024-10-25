import torch
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self,
        in_channels = 1, 
        out_channels = 32,
        kernel_size = (3,3,1),
        stride = (1,1,1),
        padding = (0,0,0),
        **kwargs
        ):
        super(encoder,self).__init__()

        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.ReLU()
    
    def forward(self,x) : 
        x = self.conv_1(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
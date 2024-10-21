import torch
import torch.nn as nn

class encoder(nn.Module):
    def __init__(self,
        in_channels = 1, 
        out_channels = 32,
        kernel_size = (3,3,1),
        stride = (1,1,1),
        padding = (0,0,0)
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

class TGRU(nn.Module):
    def __init__(self,
        n_feature,
        dropout=0.0
        ) :
        super(TGRU,self).__init__()

        self.gru = nn.GRU(n_feature,n_feature,bidirectional=True,batch_first = True,dropout=dropout)
        self.f_glu = nn.GLU()
        self.f_norm = nn.InstanceNorm1d(n_feature)
        self.b_glu = nn.GLU()  
        self.b_norm = nn.InstanceNorm1d(n_feature)

    def forward(self,x) : 
        y = torch.permute(x,(0,2,1))
        y,_ = self.gru(y)
        y = torch.permute(y,(0,2,1))

        fv = y[:,:y.shape[1]//2,:]
        bv = y[:,y.shape[1]//2:,:]

        fv = nn.functional.relu(self.f_norm(fv))
        bv = nn.functional.relu(self.b_norm(bv))
        y = x + fv + bv

        return y

class Labeler(nn.Module):
    def __init__(self,
        n_feature) : 
        super(Labeler,self).__init__()

        self.fc = nn.Linear(n_feature,1)
        self.activation = nn.Sigmoid()

    def forward(self,x) : 
        y = torch.permute(x,(0,2,1))
        y = self.fc(y)
        y = self.activation(y)
        return y

class VVAD(nn.Module):
    def __init__(self) : 
        super(VVAD,self).__init__()

        # encoder
        self.enc = []
        module = encoder(1,32, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,0,0))
        self.add_module("enc{}".format(1),module)
        self.enc.append(module)

        module = encoder(32,64, kernel_size=(3,3,3), stride=(2,2,2), padding=(0,0,0))
        self.add_module("enc{}".format(2),module)
        self.enc.append(module)

        module = encoder(64,128, kernel_size=(3,3,1), stride=(2,2,1), padding=(0,0,0))
        self.add_module("enc{}".format(3),module)
        self.enc.append(module)

        module = encoder(128,256, kernel_size=(3,3,1), stride=(2,2,1), padding=(0,0,0))
        self.add_module("enc{}".format(4),module)
        self.enc.append(module)

        module = encoder(256,256, kernel_size=(3,3,1), stride=(2,2,1), padding=(1,1,0))
        self.add_module("enc{}".format(5),module)
        self.enc.append(module)

        module = encoder(256,256, kernel_size=(3,3,1), stride=(1,1,1), padding=(0,0,0))
        self.add_module("enc{}".format(5),module)
        self.enc.append(module)

        # Bottleneck
        ## Temporal
        self.tgru = TGRU(256,dropout=0.2)

        ## Feature

        ## Channel

        ## Label
        self.labeler = Labeler(256)

        self.enc = nn.ModuleList(self.enc)

    def forward(self,x) : 
        """
            x : (B, 1, 96,96, T)
            40ms per each timestep
        """

        # encoder
        for enc in self.enc : 
            x = enc(x)
            #print("enc : {} | {}".format(x.shape, x.shape[1]*x.shape[2]*x.shape[3]))

        x = torch.squeeze(x,2)
        x = torch.squeeze(x,2)

        #print("Bottleneck : {}".format(x.shape))
        # bottlneck

        x = self.tgru(x)

        y = self.labeler(x)
        y = torch.permute(y,(0,2,1))
        return y

class VVAD_helper(nn.Module) : 
    def __init__(self,hp) : 
        super(VVAD_helper,self).__init__()
        self.m = VVAD()

    def forward(self,x,timestep=38) : 
        x = torch.permute(x,(0,2,3,1))
        x = torch.unsqueeze(x,1)
        y = self.m(x)
        y = nn.functional.interpolate(y,(timestep),mode='linear')
        y = torch.squeeze(y,1)
        return y


if __name__ == "__main__" : 
    model = VVAD_helper()
    x = torch.randn(1,1,96,96,38)
    print(x.shape)
    y = model(x,timestep=38)
    print(y.shape)
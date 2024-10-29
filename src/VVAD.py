import torch
import torch.nn as nn
import _coder

class TGRU(nn.Module):
    def __init__(self,
        n_feature,
        dropout=0.0,
        **kwargs
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

class FSA(nn.Module):
    def __init__(self,n_channels, num_heads=4,dropout=0.0) :
        super(FSA,self).__init__()

        self.SA = nn.MultiheadAttention(n_channels, n_channels, batch_first = True,dropout=dropout)
        self.bnsa = nn.BatchNorm1d(n_channels)
        self.relu = nn.ReLU()


    def forward(self,x) :
        # x : [B,C,T] -> [B,T,C]
        x_ = torch.permute(x,(0,2,1))

        ysa,h = self.SA(x_,x_,x_)
        ysa = torch.permute(ysa,(0,2,1))
        ysa = self.bnsa(ysa)
        ysa = self.relu(ysa)

        output = x + ysa
        return output

class Labeler(nn.Module):
    def __init__(self,
        n_feature,
        **kwargs) : 
        super(Labeler,self).__init__()

        self.fc = nn.Linear(n_feature,1)
        self.activation = nn.Sigmoid()

    def forward(self,x) : 
        y = torch.permute(x,(0,2,1))
        y = self.fc(y)
        y = self.activation(y)
        return y

class VVAD(nn.Module):
    def __init__(self, hp) : 
        super(VVAD,self).__init__()

        arch = hp.model.architecture

        encoder = getattr(_coder,hp.model.encoder)

        # encoder
        self.enc = []

        for i in range(len(arch["encoder"])) : 
            module = encoder(**arch["encoder"][f"enc{i+1}"])
            self.add_module(f"enc{i+1}",module)
            self.enc.append(module)

        # Bottleneck
        ## Temporal
        self.TemporalBottleneck= TGRU(**arch["TB"])

        ## Feature
        #self.FrequencialBottleneck = FSA(256, num_heads=4, dropout=0.0)

        ## Channel

        ## Label
        self.labeler = Labeler(**arch["labeler"])

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

        x = self.TemporalBottleneck(x)

        #x = self.FrequencialBottleneck(x)

        y = self.labeler(x)
        y = torch.permute(y,(0,2,1))
        return y

class VVAD_helper(nn.Module) : 
    def __init__(self,hp) : 
        super(VVAD_helper,self).__init__()
        self.m = VVAD(hp)

    def forward(self,x,timestep=38) : 
        # x : (B, T, 96,96)
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
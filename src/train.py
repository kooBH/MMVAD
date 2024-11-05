import torch
import argparse
import torchaudio
import os
import numpy as np

from tensorboardX import SummaryWriter

from utils.hparams import HParam
from utils.writer import MyWriter

from common import run,get_model, evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, default=None,
                        help="default configuration")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--step','-s',type=int,required=False,default=0)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    parser.add_argument('--epoch','-e',type=int,required=False,default=None)
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)
    

    #hp = HParam(args.config,args.default,merge_except=["architecture"])
    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration : "+args.config)

    device = args.device
    version = args.version_name
    task = hp.task
    torch.cuda.set_device(device)

    batch_size = hp.train.batch_size
    if args.epoch is None : 
        num_epochs = hp.train.epoch
    else :
        num_epochs = args.epoch
    num_workers = hp.train.num_workers

    best_loss = 1e7

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version
    log_dir = hp.log.root+'/'+'log'+'/'+version

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    writer = MyWriter(log_dir)

    if task == "VVAD" : 
        from Dataset.VVADDataset import VVADDataset
        train_dataset = VVADDataset(hp,is_train=True)
        test_dataset= VVADDataset(hp,is_train=False)
    elif task == "VAD" :
        from Dataset.VADDataset import VADDataset
        train_dataset = VADDataset(hp,is_train=True)
        test_dataset= VADDataset(hp,is_train=False)
        raise NotImplementedError("task==VAD is not implemented yet")
    else : 
        raise Exception("ERROR::Unsupported task : {}".format(task))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    model = get_model(hp,device=device)

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    if hp.loss.type == "MSELoss":
        criterion = torch.nn.MSELoss()
    elif hp.loss.type == "BCELoss":
        criterion = torch.nn.BCELoss()
    else :
        raise Exception("ERROR::Unsupported criterion : {}".format(hp.loss.type))

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.train.adam)

    if hp.scheduler.type == 'Plateau': 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            mode=hp.scheduler.Plateau.mode,
            factor=hp.scheduler.Plateau.factor,
            patience=hp.scheduler.Plateau.patience,
            min_lr=hp.scheduler.Plateau.min_lr)
    elif hp.scheduler.type == 'oneCycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                max_lr = hp.scheduler.oneCycle.max_lr,
                epochs=hp.train.epoch,
                steps_per_epoch = len(train_loader)
                )
    elif hp.scheduler.type == "CosineAnnealingLR" : 
       scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.scheduler.CosineAnnealingLR.T_max, eta_min=hp.scheduler.CosineAnnealingLR.eta_min)
    else :
        raise Exception("ERROR::Unsupported sceduler type : {}".format(hp.scheduler.type))

    step = args.step

    for epoch in range(num_epochs):
        print("Epoch : {}/{}".format(epoch+1,num_epochs))
        ### TRAIN ####
        model.train()
        train_loss=0
        for i, (batch_data) in enumerate(train_loader):
            step +=1
            loss = run(batch_data,model,criterion,hp,device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
           

            if step %  hp.train.summary_interval == 0:
                print('TRAIN::{} : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(version,epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
                writer.log_value(loss,step,'train loss : '+hp.loss.type)

        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pt')
            
        #### EVAL ####
        model.eval()
        with torch.no_grad():
            test_loss =0.
            for j, (batch_data) in enumerate(test_loader):
                loss = run(batch_data,model,criterion,hp,device=device)
                test_loss += loss.item()

                test_loss +=loss.item()

            test_loss = test_loss/len(test_loader)
            print('TEST::{} :  Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(version, epoch+1, num_epochs, j+1, len(test_loader), test_loss))
            scheduler.step(test_loss)
            
            writer.log_value(test_loss,step,'test loss : ' + hp.loss.type)

            if best_loss > test_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = test_loss

    writer.close()


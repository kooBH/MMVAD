import torch
import torch.nn as nn
from utils.metric import run_metric

def get_model(hp,device="cpu"):

    if hp.model.type == "VVAD" :
        from VVAD import VVAD_helper
        model = VVAD_helper(hp).to(device)
    elif hp.model.type == "StreamingVAD" :
        from AVAD.StreamingVAD import StreamingVAD_helper
        model = StreamingVAD_helper(hp).to(device)
    else :
        return NotImplementedError("ERROR::Unsupported model type : {}".format(hp.model.type))

    return model

def run(data,model,criterion,hp,device="cuda:0"): 

    if hp.task == "AVAD" :
        return run_AVAD(data,model,criterion,hp,device)

    # TODO for VVAD
    input = data['input'].to(device)
    target = data['target'].to(device)
    output = model(input)

    if torch.isnan(target).any() or torch.isinf(target).any():
        import pdb; pdb.set_trace()
    try : 
        loss = criterion(output,target).to(device)
    except RuntimeError as e :
        print("ERROR::{}".format(e))
        import pdb; pdb.set_trace()

    return loss
    
def run_AVAD(data,model,criterion,hp,device="cuda:0"):
    audio,label = data

    audio = audio.to(device)
    label = label.to(device)
    output,h = model(audio)

    if output.shape[-1] != label.shape[-1] :
        min_len = min(output.shape[-1],label.shape[-1])
        output = output[:,:min_len]
        label = label[:,:min_len]

    loss = criterion(output,label).to(device)

    return loss


def evaluate(hp, model,list_data,device="cuda:0"):
    #### EVAL ####
    model.eval()
    with torch.no_grad():
        ## Metric
        metric = {}
        for m in hp.log.eval : 
            metric["{}".format(m)] = 0.0

        for pair_data in list_data : 
            path_noisy = pair_data[0]
            path_clean = pair_data[1]
            noisy = rs.load(path_noisy,sr=hp.data.sr)[0]
            noisy = torch.unsqueeze(torch.from_numpy(noisy),0).to(device)
            estim = model(noisy).cpu().detach().numpy()[0]
            clean = rs.load(path_clean,sr=hp.data.sr)[0]

            if len(clean) > len(estim) :
                clean = clean[:len(estim)]
            else :
                estim = estim[:len(clean)]
            for m in hp.log.eval : 
                val= run_metric(estim,clean,m) 
                metric["{}".format(m)] += val
            
        for m in hp.log.eval : 
            key = "{}".format(m)
            metric[key] /= len(list_data)
    return metric
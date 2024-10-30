import torch
import argparse
import numpy as np
import torchaudio
import os
import sys
import glob
from utils.hparams import HParam
from AMI_label import AMI_label
from common import run,get_model
from PIL import Image
import torch
from torchvision import transforms

from tqdm.auto import tqdm

def counter(word):
    word = word.split("/")[-1]
    return int(word.split("_")[2])

def load_n_process(model,faces,transform,device="cuda:1"): 
    images = []
    for path in faces:
        image = Image.open(path)
        image = transform(image)
        images.append(image)
    images = torch.cat(images,0)
    images= images.unsqueeze(0)
    images = images.float()
    images = images.to(device)
    #print(images.shape)

    label = model(images,timestep=images.shape[1])[0]

    return label

def slide_label(label, t_label, idx, slide) : 
    if label is None :
        #print(f"slide label init {t_label.shape}")
        return t_label
    else :
        #print(f"slide_label slide {label.shape} {t_label.shape} {idx} {slide}")
        if len(t_label) > slide :
            t_label[:slide] = t_label[:slide] + label[idx:idx + slide]
            t_label[:slide] = t_label[:slide]/2
            label = torch.cat((label[:-slide],t_label),0)
        else : 
            label = torch.cat((label,t_label),0)

        return label

# Read Arbitary Data and Process
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str,required=True)
    parser.add_argument('-d','--default',type=str,required=True)
    parser.add_argument('-m','--model',type=str,required=True)
    parser.add_argument('-i','--input_dir',type=str,required=True)
    parser.add_argument('-l','--label_dir',type=str,required=True)
    parser.add_argument('-o','--output_dir',type=str,required=True)
    args = parser.parse_args()

    device = "cuda:1"
    model_name="resnet50"
    unit_segment = 50
    shift_segment = 25

    ## Parameters 
    hp = HParam(args.config,args.default)
    print('NOTE::Loading configuration :: ' + args.config)

    torch.cuda.set_device(device)

    num_epochs = 1
    batch_size = 1

    os.makedirs(args.output_dir,exist_ok=True)

    ## Model
    model = get_model(hp,device=device)
    if not args.model== None : 
        print('NOTE::Loading pre-trained model : '+ args.model)
        model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    transform = transforms.Compose([
          transforms.Grayscale(num_output_channels=1),
          transforms.ToTensor()
    ])

    # Run for AMI dataset
    AMI = glob.glob(os.path.join(args.input_dir,"amicorpus","**",'*Closeup*.avi'),recursive=True)

    GT = AMI_label(args.label_dir)

    print("AMI : {}".format(len(AMI)))

    for path in tqdm(AMI) :  
        # path : /home/data/kbh/AMI_IITP/amicorpus/ES2002a/video/ES2002a.Closeup1.avi 

        # IS1006d.Closeup4.avi 
        # ES2004a.Closeup3
        # Parse name
        name = path.split('/')[-1]
        meeting_id = name.split('.')[0]
        speaker_order = name.split('.')[-2][-1]
        speaker_order = str(int(speaker_order))
        #print(f"{meeting_id} {label[meeting_id]}")
        speaker_id = GT[meeting_id]["Label"][str(int(speaker_order)-1)]["ID"]
        sec_audio = GT[meeting_id]["GTD"]
        frame_audio = int(sec_audio*25)

        if os.path.exists(os.path.join(args.output_dir,f"{meeting_id}_{speaker_order}_{speaker_id}.npy")) :
            continue

        # dir_face : /home/data/kbh/AMI_IITP/faces/ES2002a_1
        # face jpgs :  ES2002a_1_1626_1_1.jpg
        # {meeting}_{speaker_order}_{frame:0 ~}_{face_order: 1}_{num faces}.jpg
        dir_face = os.path.join(args.input_dir,"faces",f"{meeting_id}_{speaker_order}")

        list_faces = glob.glob(os.path.join(dir_face,"*.jpg"))
        list_faces.sort(key=counter)

        # single face only
        start = 0
        end = 0
        prev_frame = 0
        prev_face = 0
        cur_frame = 0
        cur_face = 0
        on = False
        single_talk = []
        for face in list_faces : 
            cur_frame = counter(face)
            if cur_frame == prev_frame : 
                continue

            tmp = face.split("/")[-1]
            tmp = tmp .split(".")[0]
            cur_face = int(tmp.split("_")[4])

            #print(f"{on}| {prev_frame} {cur_frame} {prev_face} {cur_face} | {face}")

            # End of continuous face
            if cur_frame - prev_frame > 1 or cur_face != prev_face :
                # end of segment
                if prev_face == 1:
                    end = prev_frame
                    duration = (end-start)/25.0
                    #print(f"{start} {end} {duration}")
                    on = False

                    if duration > 1.0 : 
                        single_talk.append((start,end,items))

                # start of segment
                if cur_face == 1:
                    start = cur_frame
                    on = True
                    items = []
            if on :
                items.append(face)

            prev_face = cur_face
            prev_frame = cur_frame

        estim = -np.ones(frame_audio)

        # Run VVAD for segment
        with torch.no_grad() : 
            for idx, (start,end,faces) in enumerate(single_talk): 

                if start > frame_audio :
                    break
                if end > frame_audio :
                    #print(f"edge frame {end - start} > {frame_audio - start} | {len(faces)}")
                    faces = faces[:frame_audio-start]
                    end = frame_audio

                length = end - start+1
                #print(f"{start} ~ {end} | length {length} faces {len(faces)} | frame {frame_audio} {frame_audio - start}")

                # need to slide
                if length > unit_segment :
                    idx = 0
                    label = None
                    while idx < length : 
                        t_slide = shift_segment
                        t_faces = faces[idx:idx+unit_segment]
                        if len(t_faces) < 25 :
                            break
                        t_label = load_n_process(model,t_faces,transform)
                        label = slide_label(label,t_label,idx,shift_segment)
                        #print(f"{start} {end} {idx} {t_label.mean()}")
                        idx += shift_segment
                        if len(label) >= length :
                            break
                # single batch
                else : 
                    label = load_n_process(model,faces,transform)
                    #print(f"{start} {end} {label.mean()}")
                #print(f"{label.shape}")

                try :
                    estim[start:end+1] = label.cpu().numpy()
                except ValueError as e : 
                    print(f"ERROR:: {e}")
                    print(f"{path} : {start} {end} {label.shape} {estim.shape}")
                    continue
                except RuntimeError as e : 
                    print(f"ERROR:: {e}")
                    print(f"{path} : {start} {end} {label.shape} {estim.shape}")
                    continue

                # Write result
            np.save(os.path.join(args.output_dir,f"{meeting_id}_{speaker_order}_{speaker_id}"),estim)
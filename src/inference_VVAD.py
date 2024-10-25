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

from tqdm import tqdm

"""
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ..
"""
from ibug.face_detection import RetinaFacePredictor

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

    device = "cuda:0"
    model_name="resnet50"

    face_detector = RetinaFacePredictor(
            device=device,
            threshold=0.8,
            model=RetinaFacePredictor.get_model(model_name),
    )

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

    # Run for AMI dataset
    AMI = glob.glob(os.path.join(args.input_dir,"**",'*Closeup*.avi'),recursive=True)

    print("AMI : {}".format(len(AMI)))

    for path in AMI :  
        # Load Video
        name = path.split('/')[-1]
        meeting_id = name.split('.')[0]
        speaker_order = name.split('.')[-2][-1]
        speaker_order = str(int(speaker_order)-1)
        speaker_id = meeting_info[meeting_id][speaker_order]

        # iterative for video
        cap = cv2.VideoCapture(path)

        while True:
            ret, frame = cap.read()

            detected_faces = self.face_detector(frame, rgb=False)

            import pdb
            pdb.set_trace()

            if not ret : break

        # extract frace

        print(f'path : {path}')
        exit()


            
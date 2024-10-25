"""
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ..
"""
from ibug.face_detection import RetinaFacePredictor
import cv2
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
import os,glob

device = "cuda:1"
model_name="resnet50"

#dir_in = "/home/data/kbh/AMI_IITP/amicorpus/ES2002a/video"
#path = "/home/data/kbh/AMI_IITP/amicorpus/ES2002a/video/ES2002a.Closeup1.avi"

list_input = glob.glob("/home/data/kbh/AMI_IITP/amicorpus/*/video")

def extract(dir_in, face_detector) : 
    # iterative for video
    ID_meeting = dir_in.split("/")[-2]

    for i_spk in range(4) : 
        cnt = 0;
        path  = dir_in + f"/{ID_meeting}.Closeup{i_spk+1}.avi"
        cap = cv2.VideoCapture(path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #print(f"{path} : {frame_count}")

        if i_spk == 0 : 
            min_frame_count = frame_count
            counter = np.zeros((4,frame_count))
        else : 
            if frame_count < min_frame_count : 
                #print(f"Size mismatch : {min_frame_count} -> {frame_count}, resizing")
                min_frame_count = frame_count
                counter = np.resize(counter,(4,min_frame_count))
            elif frame_count > min_frame_count : 
                pass
                #print(f"Size mismatch : {min_frame_count} != {frame_count}, skip")

        #progress_bar = tqdm(total=min_frame_count)
        while True:
            ret, frame = cap.read()
            if not ret : break

            if cnt >= min_frame_count : break

            detected_faces = face_detector(frame, rgb=False)
            #print(f'{cnt} {detected_faces.shape}')
            num_faces = detected_faces.shape[0]
            counter[i_spk,cnt] = num_faces
            cnt += 1

            #progress_bar.update(1)
        #progress_bar.close()

    all_sit = np.all(counter == 1, axis=0)

    prev = False
    FPS = 25
    unit = 1.0/FPS

    with open(f"{ID_meeting}_all_sit.txt","w") as f :
        f.write(f"{min_frame_count*unit}\n")
        # calcuate all seated segments
        for i in range(len(all_sit)) : 
            cur = all_sit[i]
            if cur != prev : 
                # on speech
                if cur : 
                    start = i*unit
                # off speech
                else : 
                    end = i*unit
                    dur = end - start
                    if dur > 1.0 :
                        f.write(f"{start:.2f} {end:.2f}\n")
            prev = cur

def run(idx):
    dir_in = list_input[idx]
    print(dir_in)
    face_detector = RetinaFacePredictor(
            device=device,
            threshold=0.8,
            model=RetinaFacePredictor.get_model(model_name),
    )
    extract(dir_in, face_detector)

def extract_sample(dir_in):
    print(dir_in)
    face_detector = RetinaFacePredictor(
            device=device,
            threshold=0.8,
            model=RetinaFacePredictor.get_model(model_name),
    )
    extract(dir_in, face_detector)

if __name__=='__main__': 
    #cpu_num = int(cpu_count()/2)
    cpu_num = 8

    dir_in = "/home/data/kbh/AMI_IITP/amicorpus/ES2002d/video"
    extract_sample(dir_in)
    exit()

    arr = list(range(len(list_input)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(run, arr), total=len(arr),ascii=True,desc='processing'))
"""
데모 영상을 읽어서 
Face Detection을 수행
1. Bounding Box가 쳐져있는 영상을 저장
2. 개별 Face 별 영상을 저장

데모 영상에는 4명의 화자가 고정된 위치에 있다.
insta360 X3를 사용하여 촬용하였으며
웹캠모드를 사용하여 180도 영상이 위아래로 붙어있다.

해상도는 1280, 720이며 30fps로 촬영하였다.
"""

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

def process(path,face_detector) : 
    # path : blahblah/name.avi
    dir_in = os.path.dirname(path)
    name = os.path.basename(path).split(".")[0]
    name_bbox = f"{name}_bbox"

    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 코덱 설정
    vid_bbox = cv2.VideoWriter(os.path.join(dir_in,f"{name_bbox}.mp4"), fourcc, 30.0, (1280, 720)) 

    blank = np.zeros((96, 96, 3), dtype=np.uint8)

    vid_pid = []
    for i in range(4) : 
        vid_pid.append(cv2.VideoWriter(os.path.join(dir_in,f"{name}_pid_{i+1}.mp4"), fourcc, 30.0, (96, 96)))

    pbar = tqdm(total = frame_count)
    idx = 0

    while True : 
        ret, frame = cap.read()
        if not ret : break
        
        detected_faces = face_detector(frame, rgb=False)
        num_faces = detected_faces.shape[0]

        # For found faces
        found = [False,False,False,False]
        for i_faces in range(num_faces) :
            face_box = detected_faces[i_faces]
            x_left   = int(min(face_box[0], face_box[2]))
            y_top    = int(min(face_box[1], face_box[3]))
            x_right  = int(max(face_box[2], face_box[0]))
            y_bottom = int(max(face_box[3], face_box[1]))

            # exception handling
            if x_left < 0 : x_left = 0
            if y_top < 0 : y_top = 0
            if x_right > frame.shape[1] : x_right = frame.shape[1]
            if y_bottom > frame.shape[0] : y_bottom = frame.shape[0]

            # allocate position
            if x_left < 640 : 
                if y_top < 360 : 
                    pid = 1
                    found[0] = True
                    color = (0,255,0)
                else : 
                    pid = 3
                    found[2] = True
                    color = (0,0,255)
            else : 
                if y_top < 360 : 
                    pid = 2
                    found[1] = True
                    color = (255,0,0)
                else : 
                    pid = 4
                    found[3] = True
                    color = (255,255,0)

            # face
            face = frame[y_top:y_bottom, x_left:x_right]
            face = cv2.resize(face, (96, 96))

            # draw bounding box
            cv2.rectangle(frame, (x_left, y_top), (x_right, y_bottom), color, 2)
            cv2.putText(frame, f"Speaker {pid}", (x_left, y_top), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            vid_pid[pid-1].write(face)


        # save bbox
        vid_bbox.write(frame)

        # Write blank for non-face frame
        for i in range(4) : 
            if not found[i] : 
                vid_pid[i].write(blank)
        pbar.update(1)
    pbar.close()

    cap.release()
    vid_bbox.release()
    for i in range(4) : 
        vid_pid[i].release()



if __name__ == "__main__" : 
    face_detector = RetinaFacePredictor(
                device=device,
                threshold=0.8,
                model=RetinaFacePredictor.get_model(model_name),
        )
    dir_in = "/home/data/kbh/IITP" 
    list_path = glob.glob(dir_in + "/*.avi")
    for path in list_path :
        print(path)
        process(path, face_detector)

    
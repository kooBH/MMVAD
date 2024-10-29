import numpy as np
from AMI_label import AMI_label
import argparse
import os,glob


if __name__ == "__main__" : 
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_dir',type=str,required=True)
    parser.add_argument('-l','--label_dir',type=str,required=True)
    parser.add_argument('-o','--output_dir',type=str,required=True)
    args = parser.parse_args()

    list_input = glob.glob(args.input_dir + "/*.npy")
    GT = AMI_label(args.label_dir)
    
    os.makedirs(args.output_dir,exist_ok=True)

    cand_thr = [0.99, 0.8]
    FPS = 25


    for path in list_input :
        # path : /home/kbh/work/1_Active/MMVAD/output_VVAD/v8/ES2002a_1_MEE006.npy
        estim = np.load(path)
        name = path.split("/")[-1]
        meeting_id = name.split("_")[0]
        speaker_order = str(int(name.split("_")[1])-1)

        gt = GT[meeting_id]["Label"][speaker_order]["segs"]
        GTD = GT[meeting_id]["GTD"]

        # make GT array
        GT_array = np.zeros(len(estim))
        for seg in gt :
            start = int(seg[0]) * FPS
            end = int(seg[1]) * FPS
            GT_array[start:end] = 1

            if start >= len(estim) : 
                break

            if end >= len(estim) : 
                end = len(estim)

        FA = np.zeros(len(cand_thr)) # False Alarm
        MD = np.zeros(len(cand_thr)) # Miss Detection
        cnt_face = 0

        print(path)

        # evaluate
        for i in range(len(estim)) : 
            if estim[i] == -1:
                continue
            cnt_face += 1

            for t, thr in enumerate(cand_thr) : 
                if estim[i] > thr :
                    # True Positive
                    if GT_array[i] == 1 : 
                        pass
                    # False positive
                    else : 
                        FA[t] += 1
                else :
                    # False Negative
                    if GT_array[i] == 1 : 
                        MD[t] += 1
                    # True Negative
                    else :
                        pass
        
        for t, thr in enumerate(cand_thr) : 
            print(f"thr {thr} | FA {FA[t]/cnt_face} | MD {MD[t]/cnt_face} | face {cnt_face}/{len(estim)}")

        exit()








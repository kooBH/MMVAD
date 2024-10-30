import numpy as np
from AMI_label import AMI_label
import argparse
import os,glob
from tqdm.auto import tqdm


if __name__ == "__main__" : 
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_dir',type=str,required=True)
    parser.add_argument('-l','--label_dir',type=str,required=True)
    parser.add_argument('-o','--output_dir',type=str,required=True)
    parser.add_argument('-v','--version',type=str,required=True)
    args = parser.parse_args()

    list_input = glob.glob(args.input_dir + "/*.npy")
    GT = AMI_label(args.label_dir,target = "word_and_vocalsounds" )

    
    os.makedirs(args.output_dir,exist_ok=True)

    cand_thr = [0.99, 0.9, 0.5, 0.1]
    FPS = 25

    max_FA = np.zeros(len(cand_thr))
    min_FA = np.zeros(len(cand_thr))
    sum_FA = np.zeros(len(cand_thr))
    avg_FA = np.zeros(len(cand_thr))

    max_MD = np.zeros(len(cand_thr))
    min_MD = np.zeros(len(cand_thr))
    sum_MD = np.zeros(len(cand_thr))
    avg_MD = np.zeros(len(cand_thr))

    sum_GTD = 0

    for path in tqdm(list_input) :
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
        
        sum_GTD += cnt_face
        for t, thr in enumerate(cand_thr) : 
            sum_FA[t] +=FA[t]
            sum_MD[t] +=MD[t]

    avg_FA = sum_FA / sum_GTD
    avg_MD = sum_MD / sum_GTD

    with open(os.path.join(args.output_dir, args.version+".csv"),"w") as f :
        f.write("thr,FA,MD, ERR\n")
        for t, thr in enumerate(cand_thr) : 
            f.write(f"{thr},{avg_FA[t]:.2f},{avg_MD[t]:.2f},{(avg_FA[t]+avg_MD[t])/2:.2f}\n")
    








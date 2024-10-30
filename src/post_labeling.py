import glob,os
import argparse
from tqdm.auto import tqdm
from AMI_label import AMI_label
import numpy as np
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('-v','--vvad_dir',type=str,required=True)
parser.add_argument('-a','--avad_dir',type=str,required=True)
parser.add_argument('-l','--label_dir',type=str,required=True)
parser.add_argument('-o','--output_dir',type=str,required=True)
parser.add_argument('--version',type=str,required=True)
args = parser.parse_args()

FPS = 25
THR_UP = 0.8
THR_FACE = 0.9
THR_DOWN = 0.1
GT = AMI_label(args.label_dir)

def sort_by_order(word) : 
    word = word.split("/")[-1]
    return int(word.split("_")[1])

"""
첫 프레임은 0초가 중심입니다.
그래서 shift = 1s, win_len = 1s 둘다
즉 앞에 0.5초 패딩하고 뽑았습니다
"""
def post(path) : 
    # path = args.css_dir + "/ES2002d_avad.npy"
    name = path.split("/")[-1]
    meeting_id = name.split("_")[0]
    avad = np.load(path)

    avad = np.repeat(avad, FPS,axis=1)

    # load VVAD label
    # path_vvad = args.vvad_dir/ES2002d_1_MEE006.npy
    list_vvad = glob.glob(os.path.join(args.vvad_dir,meeting_id+"*.npy"))

    # merge vvad
    vvad = None
    list_vvad.sort(key=sort_by_order)
    for i_vvad, path_vvad in enumerate(list_vvad) : 
        t_vvad = np.load(path_vvad)
        #print(path_vvad)

        if vvad is None : 
            vvad = np.zeros((4,t_vvad.shape[0]))
        vvad[i_vvad] = t_vvad

    # Cut length
    # NOTE : vvad length is aligned at the inference routine
    len_avad = avad.shape[1]
    len_vvad = vvad.shape[1]

    len_vad = min(len_avad,len_vvad)

    # find optimal order in permutations
    permutations = list(itertools.permutations([0,1,2,3]))
    score = 0
    order = [0,1,2,3]
    # for each permutation
    for perm in permutations : 
        c_score = 0
        n_frame = 0
        correct = 0
        # for 4 speakers
        for i_vvad, i_avad in enumerate(perm) :
            # for each frame
            for i in range(len_vad) : 
                # active video only
                if vvad[i_vvad][i]!= -1 :
                    val = vvad[i_vvad][i] > THR_FACE
                    # count for 
                    if not val :
                        continue
                    correct += val == avad[i_avad][i]
                    n_frame += 1

        c_score = correct / n_frame
        if c_score > score : 
            score = c_score
            order = perm
    print(f"Optimal order : {order} {score:.3f}")
    # into optimal order
    avad = avad[order, :]


    Label = GT[meeting_id]["Label"]
    #print(Label)

    DER1, result1 = GT.measure(meeting_id,avad,unit=0.04,skip_overlap=False)
    DER2, result2 = GT.measure(meeting_id,avad,unit=0.04,skip_overlap=True)

    print(f"w   OL | FA : {result1['false alarm']:.0f}, MD : {result1['missed detection']:.0f}, DER : {result1['diarization error rate']:.4f}")
    print(f"w/o OL | FA : {result2['false alarm']:.0f}, MD : {result2['missed detection']:.0f}, DER : {result2['diarization error rate']:.4f}")

    # post processing for FA(False Alarm) only
    for s in range(4) : 
        for i in range(len_vad) : 
            # Check face exists
            if vvad[s][i] == -1 :
                continue
            
            if avad[s][i] == 1 :  
                vprob_other = 0 
                aprob_other = 0
                for j in range(4):
                    if j==s :
                        continue
                    vprob_other += vvad[j][i]
                    aprob_other += avad[j][i]
                if vprob_other + aprob_other > 1.5 :
                    avad[s][i] = 0
            
    # Eval

    print("---- post processed ----")

    DER1, result1 = GT.measure(meeting_id,avad,unit=0.04,skip_overlap=False)
    DER2, result2 = GT.measure(meeting_id,avad,unit=0.04,skip_overlap=True)

    print(f"w   OL | FA : {result1['false alarm']:.0f}, MD : {result1['missed detection']:.0f}, DER : {result1['diarization error rate']:.4f}")
    print(f"w/o OL | FA : {result2['false alarm']:.0f}, MD : {result2['missed detection']:.0f}, DER : {result2['diarization error rate']:.4f}")



if __name__ == "__main__" : 
    # Run for estimation by audio
    list_avad = glob.glob(args.avad_dir+ "/*.npy")

    GT = AMI_label(args.label_dir)

    for path in tqdm(list_avad) : 
        post(path)

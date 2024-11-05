import glob,os
import argparse
from tqdm.auto import tqdm
from AMI_label import AMI_label
import numpy as np
import itertools
from multiprocessing import Pool, cpu_count
import csv
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('-v','--vvad_dir',type=str,required=True)
parser.add_argument('-a','--avad_dir',type=str,required=True)
parser.add_argument('-l','--label_dir',type=str,required=True)
parser.add_argument('-o','--output_dir',type=str,required=True)
parser.add_argument('-m','--mmvad_dir',type=str,required=True)
parser.add_argument('--version',type=str,required=True)
args = parser.parse_args()

FPS = 25
THR_UP = 0.8
THR_FACE = 0.9
THR_DOWN = 0.1
UNIT = 0.6
GT = AMI_label(args.label_dir, target = "word_and_vocalsounds")

def sort_by_order(word) : 
    word = word.split("/")[-1]
    return int(word.split("_")[1])

"""
AVAD_1
첫 프레임은 0초가 중심입니다.
그래서 shift = 1s, win_len = 1s 둘다
즉 앞에 0.5초 패딩하고 뽑았습니다

AVAD_2
형식은 지난번하고 약간 다르게 4명 한파일로 묶었고, CSS에서 분리 스트림 별로 아니고 그냥 최종 Diarize 결과이고, 
upsample안해둬서 1세그먼트당 0.6초 간격입니다
e.g. /AVAD_2/ES2011a.npy
"""
def post(path, THR_SUM = 1.0) : 
    name = path.split("/")[-1]
    meeting_id = name.split(".")[0]

    avad = np.load(path)

    # load VVAD label
    # path_vvad = args.vvad_dir/ES2002d_1_MEE006.npy
    list_vvad = glob.glob(os.path.join(args.vvad_dir,meeting_id+"*.npy"))

    if len(list_vvad) == 0 :
        print(f"VVAD for {meeting_id} is not found")
        return 0,0,0,0,0,0,0

    # merge vvad
    vvad = None
    list_vvad.sort(key=sort_by_order)
    for i_vvad, path_vvad in enumerate(list_vvad) : 
        t_vvad = np.load(path_vvad)
        #print(path_vvad)

        if vvad is None : 
            vvad = np.zeros((4,t_vvad.shape[0]))
        vvad[i_vvad] = t_vvad

    # matching length
    orig_shape = vvad.shape
    avad = torch.tensor(avad, dtype=torch.float)
    vvad = torch.tensor(vvad)

    # add extra dimension to use torch function
    vvad = vvad.unsqueeze(0)
    vvad = vvad.unsqueeze(0)

    vvad = nn.functional.interpolate(vvad, size=avad.shape,mode='nearest')

    # squeezing
    vvad = vvad.squeeze(0)
    vvad = vvad.squeeze(0)

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
    #print(f"Optimal order : {order} {score:.3f}")
    # into optimal order
    avad = avad[order, :]

    Label = GT[meeting_id]["Label"]
    #print(Label)

    DER11, result1 = GT.measure(meeting_id,avad,unit=UNIT,skip_overlap=True)

    FA1 = result1["false alarm"]
    MD1 = result1["missed detection"]
    SC1 = result1["confusion"]
    Total = result1["total"]

    #print(f"w   OL | FA : {result1['false alarm']:.0f}, MD : {result1['missed detection']:.0f}, DER : {result1['diarization error rate']:.4f}")

    # post processing for FA(False Alarm) only
    for s in range(4) : 
        for i in range(len_vad) : 
            # Check face exists
            if vvad[s][i] == -1 :
                continue
            
            if avad[s][i] == 1 :  
                # It might be other speaker's voice
                vprob_other = 0 
                aprob_other = 0
                for j in range(4):
                    if j==s :
                        continue
                    vprob_other += vvad[j][i]
                    aprob_other += avad[j][i]
                if vprob_other + aprob_other > THR_SUM :
                    avad[s][i] = 0

                # It might be silence
                #if vvad[s][i] < 0.005 :
                #    avad[s][i] = 0

            # It is definitely talking face
            # if avad[s][i] == 0 :
            #   if vvad[s][i] > 0.9 :
            #        avad[s][i] = 1

            # It doesn't seem to be talking face
            #if avad[s][i] == 1 :
            #    if vvad[s][i] < 0.1 :
            #        avad[s][i] = 0

    # Eval

    #print("---- post processed ----")

    DER21, result2 = GT.measure(meeting_id,avad,unit=UNIT, skip_overlap=True)

    FA2 = result2["false alarm"]
    MD2 = result2["missed detection"]
    SC2 = result2["confusion"]

    # save vad
    os.makedirs(os.path.join(args.mmvad_dir),exist_ok=True)
    np.save(os.path.join(args.mmvad_dir,f"{meeting_id}.npy"),avad)

    return FA1,MD1,SC1, FA2,MD2,SC2, Total

list_avad = glob.glob(args.avad_dir+ "/*")
list_thr = [0.7, 1.0, 1.2, 1.3, 2.0]

def process(idx) : 
    path = list_avad[idx]
    name = path.split("/")[-1]
    name = name.split(".")[0]
    for thr in list_thr :
        with open(os.path.join(args.output_dir,"tmp",str(thr),f"{name}.csv"), "w") as f :
            FA1,MD1,SC1,FA2,MD2,SC2, Total = post(path,THR_SUM=thr)
            f.write(f"{FA1}, {MD1}, {SC1}, {FA2}, {MD2}, {SC2}, {Total}\n")

if __name__ == "__main__" : 
    # Run for estimation by audio
    #list_avad = glob.glob(args.avad_dir+ "/*")

    GT = AMI_label(args.label_dir)
    os.makedirs(args.output_dir,exist_ok=True)
    os.makedirs(args.output_dir+"/tmp",exist_ok=True)

    cpu_num = int(cpu_count()/2)
#    for path in list_avad :
#        post(path)
#        exit()

    for thr in list_thr :
        os.makedirs(os.path.join(args.output_dir,"tmp", str(thr)),exist_ok=True)

    # Parallel processing
    arr = list(range(len(list_avad)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(process, arr), total=len(arr),ascii=True,desc='processing'))

    print("Version : ",args.version)
    # Merge results
    with open(os.path.join(args.output_dir, args.version+".csv"),"w") as f :
        f.write("thr,DER1,DER2,Delta,Total,FA1,FA2, MD1,MD2,SC1,SC2\n")
        for thr in list_thr : 
            list_csv = glob.glob(os.path.join(args.output_dir,"tmp",str(thr),"*.csv"))

            FA1=0
            MD1=0
            SC1=0
            FA2=0
            MD2=0
            SC2=0
            Total=0
            for path in list_csv : 
                with open(path,"r") as f2: 
                    reader = csv.reader(f2)
                    row = next(reader)
                    FA1 += float(row[0])
                    MD1 += float(row[1])
                    SC1 += float(row[2])
                    FA2 += float(row[3])
                    MD2 += float(row[4])
                    SC2 += float(row[5])
                    Total += float(row[6])
            DER1 = (FA1+MD1+SC1)/Total
            DER2 = (FA2+MD2+SC2)/Total
            f.write(f"{thr}, {DER1:.4f}, {DER2:.4f}, {(DER2 - DER1):.4f},{Total}, {FA1:.2f}, {FA2:.2f}, {MD1:.2f}, {MD2:.2f}, {SC1:.2f}, {SC2:.2f}\n")
    # Clear temporary files
    os.system(f"rm -r {args.output_dir}/tmp")

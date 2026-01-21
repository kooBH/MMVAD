import os,glob,shutil



list_files = glob.glob(os.path.join("/home/data/VAD/AVA-Speech/train_58","**","*.wav"),recursive=True)

dir_out = "/home/data/VAD/AVA-Speechmoved/dev_4"

os.makedirs(os.path.join(dir_out,"CLEAN_SPEECH"),exist_ok=True)
os.makedirs(os.path.join(dir_out,"NO_SPEECH"),exist_ok=True)
os.makedirs(os.path.join(dir_out,"SPEECH_WITH_NOISE"),exist_ok=True)
os.makedirs(os.path.join(dir_out,"SPEECH_WITH_MUSIC"),exist_ok=True)


count = {"CLEAN_SPEECH":0,"NO_SPEECH":0,"SPEECH_WITH_NOISE":0,"SPEECH_WITH_MUSIC":0}


for path in list_files :
    # e.g. K_SpqDJnlps_1394.43_1395.14_CLEAN_SPEECH.wav
    filename = os.path.basename(path)
    id = filename[:11]

    if id in ["Di1MG6auDYo","HKjR70GCRPE","skiZueh4lfY","vfjywN5CN0Y"] : 
        if "CLEAN_SPEECH" in filename :
            shutil.move(path,os.path.join(dir_out,"CLEAN_SPEECH",filename))
            count["CLEAN_SPEECH"] += 1
        elif "NO_SPEECH" in filename :
            shutil.move(path,os.path.join(dir_out,"NO_SPEECH",filename))
            count["NO_SPEECH"] += 1
        elif "SPEECH_WITH_NOISE" in filename :
            shutil.move(path,os.path.join(dir_out,"SPEECH_WITH_NOISE",filename))
            count["SPEECH_WITH_NOISE"] += 1
        elif "SPEECH_WITH_MUSIC" in filename :
            shutil.move(path,os.path.join(dir_out,"SPEECH_WITH_MUSIC",filename))
            count["SPEECH_WITH_MUSIC"] += 1
    else :
        continue

print(count)
print({sum(count.values())})
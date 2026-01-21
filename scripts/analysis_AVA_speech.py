import os,glob



data = {}


list_files = glob.glob(os.path.join("/home/data/VAD/AVA-Speech","**","*.wav"),recursive=True)



for path in list_files :
    # e.g. K_SpqDJnlps_1394.43_1395.14_CLEAN_SPEECH.wav
    filename = os.path.basename(path)
    id = filename[:11]

    if id not in data :
        data[id] = {"CLEAN_SPEECH":0,"NO_SPEECH":0,"SPEECH_WITH_NOISE":0,"SPEECH_WITH_MUSIC":0}

    if "CLEAN_SPEECH" in filename :
        data[id]["CLEAN_SPEECH"] += 1
    elif "NO_SPEECH" in filename :
        data[id]["NO_SPEECH"] += 1
    elif "SPEECH_WITH_NOISE" in filename :
        data[id]["SPEECH_WITH_NOISE"] += 1
    elif "SPEECH_WITH_MUSIC" in filename :
        data[id]["SPEECH_WITH_MUSIC"] += 1

print("ID, CLEAN_SPEECH, NO_SPEECH, SPEECH_WITH_NOISE, SPEECH_WITH_MUSIC")
for id in data :
    print(f"{id}, {data[id]['CLEAN_SPEECH']}, {data[id]['NO_SPEECH']}, {data[id]['SPEECH_WITH_NOISE']}, {data[id]['SPEECH_WITH_MUSIC']} | Total: {sum(data[id].values())}")

print(f"Total unique IDs: {len(data)}")
    
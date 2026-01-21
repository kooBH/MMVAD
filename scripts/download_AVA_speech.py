from __future__ import unicode_literals
import os
import pandas as pd
import yt_dlp as youtube_dl
import subprocess
import logging
from tqdm import tqdm

# Constants
prefix = 'https://www.youtube.com/watch?v='
temp_root = '/home/data/VAD/AVA-Speech-Temp/'   # Full audio storage
output_root = '/home/data/VAD/AVA-Speech-Segments/' # Final 16kHz segments

class MyLogger(object):
    def __init__(self):
        logging.basicConfig(filename='download.log', level=logging.ERROR)
    def debug(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): logging.error(msg)

def download_full_audio(video_id):
    """Download the best quality audio as a single file."""
    outtmpl = os.path.join(temp_root, f"{video_id}.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': outtmpl,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'postprocessor_args': ['-ar', '16000', '-ac', '1'],
        'quiet': True,
        'ignoreerrors': True,
        'logger': MyLogger(),
    }
    
    target_path = os.path.join(temp_root, f"{video_id}.wav")
    if os.path.exists(target_path):
        return target_path

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        result = ydl.download([prefix + video_id])
        if result == 0: # Success
            return target_path
    return None

def extract_segment(input_path, start_s, end_s, label, video_id):
    """Extract a specific segment from the full audio using ffmpeg directly."""
    duration = float(end_s) - float(start_s)
    clean_label = str(label).replace(' ', '_')
    output_filename = f"{video_id}_{start_s}_{end_s}_{clean_label}.wav"
    output_path = os.path.join(output_root, clean_label, output_filename)

    if os.path.exists(output_path):
        return

    # FFmpeg command for precise and fast seeking
    # -ss before -i is faster; -t is duration
    cmd = [
        'ffmpeg', '-y', '-ss', str(start_s), '-i', input_path,
        '-t', str(duration), '-c', 'copy', output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

if __name__ == '__main__':
    os.makedirs(temp_root, exist_ok=True)
    os.makedirs(output_root, exist_ok=True)
    for list_sub in ['SPEECH_WITH_NOISE','NO_SPEECH','CLEAN_SPEECH','SPEECH_WITH_MUSIC'] :
        os.makedirs(os.path.join(output_root, list_sub), exist_ok=True)

    try:
        ava = pd.read_csv('ava_speech_labels_v1.csv', names=['id', 'start', 'end', 'label'])
        unique_ids = ava['id'].unique()

        print(f"System: Processing {len(unique_ids)} videos to extract {len(ava)} segments.")

        for video_id in tqdm(unique_ids, desc="Videos"):
            # 1. Download full audio if not present
            full_audio_path = download_full_audio(video_id)
            
            if full_audio_path and os.path.exists(full_audio_path):
                # 2. Extract all segments for this video_id
                segments = ava[ava['id'] == video_id]
                for _, row in segments.iterrows():
                    extract_segment(full_audio_path, row['start'], row['end'], row['label'], video_id)
                
                # 3. Optional: Delete full audio to save space after extraction
                # os.remove(full_audio_path) 
            else:
                logging.error(f"Failed to download full audio for ID: {video_id}")

    except Exception as e:
        print(f"Critical Error: {e}")
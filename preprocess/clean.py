import pandas as pd
import wave
import os
from glob import glob
from tqdm import tqdm
df_paths = glob('data/*.parquet')

def save_audio(row, file_name):
    audio_bytes = row['audio']['bytes']
    
    # Convert bytes to wave format and save
    with wave.open(file_name, 'wb') as wave_file:
        wave_file.setnchannels(1) # Mono
        wave_file.setsampwidth(2) # Assuming 16-bit PCM
        wave_file.setframerate(88200) # Assuming a sample rate of 16000 Hz
        wave_file.writeframes(audio_bytes)

voice_counters = {}

for df_path in df_paths:
    df = pd.read_parquet(df_path)
    for index, row in tqdm(df.iterrows()):
        voice = int(row['voice'])
        voice_str = f"p_{voice:03d}"
        if voice_str not in voice_counters:
            voice_counters[voice_str] = 0
        caption = row['caption']
        # mkdir if not exists
        txt_dir = f"txt/{voice_str}"
        os.makedirs(txt_dir, exist_ok=True)
        # save caption
        with open(f"{txt_dir}/{voice_str}_{voice_counters[voice_str]:03d}.txt", 'w') as f:
            f.write(caption)
        # mkdir if not exists
        audio_dir = f"wave/{voice_str}"
        os.makedirs(audio_dir, exist_ok=True)
        # save audio
        save_audio(row, f"{audio_dir}/{voice_str}_{voice_counters[voice_str]:03d}_mic1.wav")
        voice_counters[voice_str] += 1
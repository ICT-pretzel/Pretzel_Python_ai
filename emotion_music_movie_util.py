import librosa
import numpy as np
from moviepy.editor import VideoFileClip
import soundfile as sf
from tqdm import tqdm
from transformers import pipeline

classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", from_pt=True)

def extract_audio_from_video(video_path, audio_path):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='pcm_s16le')
        print("Audio extraction complete.")
    except Exception as e:
        print(f"Error extracting audio: {e}")

def split_audio_file(file_path, chunk_duration=10):
    y, sr = librosa.load(file_path, sr=None)
    chunk_length = int(chunk_duration * sr)
    chunks = [y[i:i + chunk_length] for i in range(0, len(y), chunk_length)]
    return chunks, sr

def is_music_present(y_chunk, sr):
    energy = np.sum(librosa.feature.rms(y=y_chunk))
    return energy > 0.01

def detect_sudden_increase(y_chunk, prev_rms, threshold=2.5):
    current_rms = np.mean(librosa.feature.rms(y=y_chunk))
    if prev_rms is not None and current_rms > prev_rms * threshold:
        return True, current_rms
    return False, current_rms

def analyze_music_mood(audio_file):
    chunks, sr = split_audio_file(audio_file)
    results = []
    prev_rms = None
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        if is_music_present(chunk, sr):
            sudden_increase, prev_rms = detect_sudden_increase(chunk, prev_rms)
            if sudden_increase:
                result = [{"label": "anx", "score": 1.0}]
            else:
                temp_file = "temp_chunk.wav"
                sf.write(temp_file, chunk, sr)
                result = classifier(temp_file)
                # 가장 높은 score를 가진 label만 선택
                max_result = max(result, key=lambda x: x["score"])
            start_time = i * 10
            end_time = (i + 1) * 10
            results.append({
                "timestamp": f"{convert_seconds_to_hms(start_time)}-{convert_seconds_to_hms(end_time)}",
                "result": max_result
            })
    return results

def convert_seconds_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:05.2f}"

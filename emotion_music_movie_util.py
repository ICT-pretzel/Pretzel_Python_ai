import os
import librosa
import numpy as np
import soundfile as sf
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from transformers import pipeline
import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uuid
import json
from datetime import datetime
from urllib.parse import urlparse

router = APIRouter()

class VideoURLWithoutActors(BaseModel):
    url: str

classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", from_pt=True)

def extract_audio_from_video(video_path, audio_path):
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec='pcm_s16le')
        video.close()
        print("Audio extraction complete.")
    except Exception as e:
        print(f"Error extracting audio: {e}")
        raise HTTPException(status_code=500, detail=f"Audio extraction error: {e}")

def split_audio_file(file_path, chunk_duration=10):
    try:
        y, sr = librosa.load(file_path, sr=None)
        chunk_length = int(chunk_duration * sr)
        chunks = [y[i:i + chunk_length] for i in range(0, len(y), chunk_length)]
        return chunks, sr
    except Exception as e:
        print(f"Error splitting audio file: {e}")
        raise HTTPException(status_code=500, detail=f"Audio splitting error: {e}")

def is_music_present(y_chunk, sr):
    try:
        energy = np.sum(librosa.feature.rms(y=y_chunk))
        return energy > 0.01
    except Exception as e:
        print(f"Error checking music presence: {e}")
        raise HTTPException(status_code=500, detail=f"Music presence check error: {e}")

def detect_sudden_increase(y_chunk, prev_rms, threshold=3.0):
    try:
        current_rms = np.mean(librosa.feature.rms(y=y_chunk))
        if prev_rms is not None and current_rms > prev_rms * threshold:
            return True, current_rms
        return False, current_rms
    except Exception as e:
        print(f"Error detecting sudden increase: {e}")
        raise HTTPException(status_code=500, detail=f"Sudden increase detection error: {e}")

def analyze_music_mood(audio_file):
    try:
        chunks, sr = split_audio_file(audio_file)
        results = []
        emotion_counts = {}
        prev_rms = None
        for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            if is_music_present(chunk, sr):
                sudden_increase, current_rms = detect_sudden_increase(chunk, prev_rms, threshold=3.0)
                prev_rms = current_rms
                if sudden_increase:
                    print(f"Sudden increase detected in chunk {i} with current RMS {current_rms}")
                    result = [{"label": "anx", "score": 1.0}]
                    label = "anx"
                else:
                    temp_file = f"temp_chunk_{uuid.uuid4().hex}.wav"
                    sf.write(temp_file, chunk, sr)
                    result = classifier(temp_file)
                    max_result = max(result, key=lambda x: x["score"])
                    label = max_result["label"]
                    print(f"Chunk {i}: {label} with score {max_result['score']}")
                    os.remove(temp_file)
                if label not in emotion_counts:
                    emotion_counts[label] = 0
                emotion_counts[label] += 1
                start_time = i * 10
                end_time = (i + 1) * 10
                results.append({
                    "timestamp": f"{convert_seconds_to_hms(start_time)} --> {convert_seconds_to_hms(end_time)}",
                    "result": result[0]
                })
            else:
                prev_rms = None
        return results, emotion_counts
    except Exception as e:
        print(f"Error analyzing music mood: {e}")
        raise HTTPException(status_code=500, detail=f"Music mood analysis error: {e}")

def convert_seconds_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:05.2f}".replace('.', ',')

def process_emotion_music_movie(video_url):
    try:
        response = requests.get(video_url, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="비디오 파일 다운로드에 실패했습니다")

        temp_video_id = uuid.uuid4().hex
        file_location = f"temp_video_{temp_video_id}.mp4"
        with open(file_location, "wb") as file_object:
            for chunk in response.iter_content(chunk_size=8192):
                file_object.write(chunk)

        audio_file = file_location.replace(".mp4", ".wav")
        extract_audio_from_video(file_location, audio_file)

        mood_results, emotion_counts = analyze_music_mood(audio_file)

        try:
            os.remove(file_location)
        except OSError as e:
            print(f"Error deleting temp video file: {e}")

        try:
            os.remove(audio_file)
        except OSError as e:
            print(f"Error deleting temp audio file: {e}")

        final_results = []
        for result in mood_results:
            start_time = result["timestamp"].split(' --> ')[0]
            end_time = result["timestamp"].split(' --> ')[1]
            label = result["result"]["label"]
            final_results.append({"time": f"{start_time} --> {end_time}", "label": label})

        return final_results, emotion_counts
    except Exception as e:
        print(f"Exception in process_emotion_music_movie: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_filename_from_url(url: str) -> str:
    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path).split('.')[0]

def save_results_as_json(results, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

@router.post("/")
async def emotion_music_movie(video_url: VideoURLWithoutActors):
    print("음악 감정 분석 요청을 받았습니다.")
    try:
        final_results, emotion_counts = process_emotion_music_movie(video_url.url)

        video_filename = get_filename_from_url(video_url.url)

        # 결과를 results 폴더에 저장
        os.makedirs('results', exist_ok=True)
        json_filename = f"{video_filename}_emotion_music_results.json"

        save_results_as_json(final_results, os.path.join('results', json_filename))

        return JSONResponse(content={"message": f"Results saved to {json_filename}"})

    except Exception as e:
        print(f"예외 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Export the router to be included in main.py
emotion_music_movie_router = router
import os
import uvicorn
from fastapi import FastAPI
from actor_face_movie_util import router as actor_face_movie_router
from emotion_music_movie_util import router as emotion_music_movie_router
from worlds_subtitle_movie import router as worlds_subtitle_movie_router

app = FastAPI()

# Include routers from other modules
app.include_router(actor_face_movie_router, prefix="/actor_face_movie")
app.include_router(emotion_music_movie_router, prefix="/emotion_music_movie")
app.include_router(worlds_subtitle_movie_router, prefix="/worlds_subtitle_movie")

if __name__ == "__main__":
    print("서버를 시작합니다...")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Users\\ICT05_04\\Desktop\\finalproject\\movie\\translate-movie-427703-adec2ac5235a.json"
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
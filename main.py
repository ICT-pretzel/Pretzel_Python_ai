import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from datetime import datetime
import json
from actor_face_movie_util import update_actors, load_embeddings, process_video
from emotion_music_movie_util import process_emotion_music_movie
from worlds_subtitle_movie import translate_subtitle
from urllib.parse import unquote, urlparse
from kobert import kobert_eval
from datetime import datetime, timedelta

app = FastAPI()

class VideoURL(BaseModel):
    url: str
    actors: dict

class VideoURLWithoutActors(BaseModel):
    subtitle_url: str
    video_url: str

class SubtitleRequest(BaseModel):
    url: str

embeddings = load_embeddings()
# 시간 변환 함수
def time_to_seconds(time_str):
    """Convert time format HH:MM:SS.ms to total seconds."""
    t = datetime.strptime(time_str, "%H:%M:%S.%f")
    return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond).total_seconds()

# 시간 비교 함수
def overlap_and_same_label(entry1, entry2):
    """Check if two entries overlap and have the same label."""
    start1, end1 = map(time_to_seconds, entry1["time"].split(" --> "))
    start2, end2 = map(time_to_seconds, entry2["time"].split(" --> "))
    
    overlap = max(start1, start2) < min(end1, end2)
    same_label = entry1["label"] == entry2["label"]
    
    return overlap and same_label

def get_filename_from_url(url: str) -> str:
    parsed_url = unquote(urlparse(url).path)
    return os.path.splitext(os.path.basename(parsed_url))[0]

@app.post("/actor_face_movie/")
async def actor_face_movie(video_url: VideoURL):
    print("배우 얼굴 인식 요청을 받았습니다.")
    try:
        update_actors(video_url.actors, embeddings)
        print("배우 정보를 성공적으로 업데이트했습니다.")

        final_results = process_video(video_url.url, embeddings)

        # 결과를 results 폴더에 저장
        os.makedirs('results', exist_ok=True)
        original_filename = get_filename_from_url(video_url.url)
        filename = f"{original_filename}_actor_face_results.json"
        with open(os.path.join('results', filename), 'w', encoding='utf-8') as f:
            json.dump({"actor_timestamps": final_results}, f, ensure_ascii=False, indent=4)

        return JSONResponse(content={"actor_timestamps": final_results})

    except Exception as e:
        print(f"예외 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/emotion_music_movie/")
async def emotion_music_movie(request: VideoURLWithoutActors):
    try:
        print("음악 및 텍스트 감정 분석 요청을 받았습니다.")
        print("KoBert 작동중...")  # 요청 수신 로그 출력
        subtitle_url = "https://storage.googleapis.com/pretzel-movie/"+request.subtitle_url
        response = requests.get(subtitle_url, stream=True)
        response.encoding = 'utf-8'
        subtitle_text = response.text
        subtitle_text = subtitle_text.replace('\r\n', '\n').replace('\r', '\n')
        kobert_results, kobert_counts = kobert_eval(subtitle_text)

        print("emotion_music_movie 작동중...")  # 요청 수신 로그 출력
        music_results, music_counts = process_emotion_music_movie(request.video_url)
        
        final_results = []
        for entry1 in music_results:
            for entry2 in kobert_results:
                if overlap_and_same_label(entry1, entry2):
                    final_results.append(entry1)
                    break
        emotion_counts={        
            "sad": kobert_counts["sad"]*0.4+music_counts["sad"]*0.6,
            "hap": kobert_counts["hap"]*0.4+music_counts["hap"]*0.6,
            "ang": kobert_counts["ang"]*0.4+music_counts["ang"]*0.6,
            "anx": kobert_counts["anx"]*0.4+music_counts["anx"]*0.6
        }
        
        # 결과를 results 폴더에 저장
        os.makedirs('results', exist_ok=True)
        original_filename = get_filename_from_url(request.video_url)
        filename = f"{original_filename}_emotion_music_results.json"
        with open(os.path.join('results', filename), 'w', encoding='utf-8') as f:
            json.dump({"mood_results": final_results, "emotion_counts": emotion_counts}, f, ensure_ascii=False, indent=4)

        return JSONResponse(content={"mood_results": final_results, "emotion_counts": emotion_counts})

    except Exception as e:
        print(f"예외 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/worlds_subtitle_movie/")
async def worlds_subtitle_movie(subtitle_request: SubtitleRequest):
    print("자막 번역 요청을 받았습니다.")
    try:
        translations = translate_subtitle(subtitle_request)
        return JSONResponse(content=translations)
    except Exception as e:
        print(f"예외 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if not os.path.exists("images"):
    os.makedirs("images")

@app.post("/kobert/")
async def kobert(subtitle_url: str):
    try:
        print("KoBert 작동중...")  # 요청 수신 로그 출력
        subtitle_url = "https://storage.googleapis.com/pretzel-movie/"+subtitle_url
        response = requests.get(subtitle_url, stream=True)
        response.encoding = 'utf-8'
        subtitle_text = response.text
        subtitle_text = subtitle_text.replace('\r\n', '\n').replace('\r', '\n')
        kobert_results, kobert_counts = kobert_eval(subtitle_text)
        return kobert_counts

    except Exception as e:
        print(f"Exception: {str(e)}")  # 예외 발생 시 로그 출력
        raise HTTPException(status_code=500, detail=str(e))  # 예외 응답 반환

if __name__ == "__main__":
    print("서버를 시작합니다...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

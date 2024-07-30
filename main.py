import os

# 서비스 계정 키 파일의 경로를 환경 변수로 설정
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ict-pretzel-43373d904ced.json"

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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
from google.cloud import storage

app = FastAPI()

origins = [
    "http://localhost:3010",
    "http://127.0.0.1:3010",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VideoURL(BaseModel):
    url: str
    actors: dict

class VideoURLWithoutActors(BaseModel):
    subtitle_url: str
    video_url: str

class SubtitleRequest(BaseModel):
    url: str

# 프로젝트 ID 및 버킷 이름 설정
project_id = 'ict-pretzel'
bucket_name = 'pretzel-movie'

# 시간 변환 함수
def time_to_seconds(time_str):
    t = datetime.strptime(time_str, "%H:%M:%S.%f")
    return timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond).total_seconds()

# 시간 비교 함수
def overlap_and_same_label(entry1, entry2):
    start1, end1 = map(time_to_seconds, entry1["time"].split(" --> "))
    start2, end2 = map(time_to_seconds, entry2["time"].split(" --> "))
    
    overlap = max(start1, start2) < min(end1, end2)
    same_label = entry1["label"] == entry2["label"]
    
    return overlap and same_label

def get_filename_from_url(url: str) -> str:
    parsed_url = unquote(urlparse(url).path)
    return os.path.splitext(os.path.basename(parsed_url))[0]

@app.post("/actor_face_movie/")
def actor_face_movie(video_url: VideoURL):
    print("배우 얼굴 인식 요청을 받았습니다.")
    try:
        # 배우 사진 임베딩 정보 로드
        embeddings = load_embeddings()
        
        update_actors(video_url.actors, embeddings)
        print("배우 정보를 성공적으로 업데이트했습니다.")

        
        final_results = process_video("https://storage.googleapis.com/pretzel-movie/"+video_url.url, embeddings)
        
        # Storage 클라이언트 생성 (환경 변수를 통해 인증 정보를 자동으로 인식)
        client = storage.Client(project=project_id)

        # 업로드 파일 json화
        json_res = json.dumps(final_results)
        
        # 파일 이름 설정
        original_filename = get_filename_from_url(video_url.url)
        deepface_filename = f"{original_filename}_deepface.json"
        
        # 스토리지 폴더 밀 이름 설정
        deepface_folder = "pretzel-deepfaceAI/"
        deepface_blob_name = deepface_folder + deepface_filename

        # 결과 파일 업로드
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(deepface_blob_name)
        blob.upload_from_string(json_res, content_type='application/json')

        return 1

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
        music_results, music_counts = process_emotion_music_movie("https://storage.googleapis.com/pretzel-movie/"+request.video_url)
        
        final_results = []
        for entry1 in music_results:
            for entry2 in kobert_results:
                if overlap_and_same_label(entry1, entry2):
                    final_results.append(entry1)
                    break
        emotion_counts={        
            "sad": int(kobert_counts["sad"]*0.4+music_counts["sad"]*0.6),
            "hap": int(kobert_counts["hap"]*0.4+music_counts["hap"]*0.6),
            "ang": int(kobert_counts["ang"]*0.4+music_counts["ang"]*0.6),
            "anx": int(kobert_counts["anx"]*0.4+music_counts["anx"]*0.6)
        }
        
        # Storage 클라이언트 생성 (환경 변수를 통해 인증 정보를 자동으로 인식)
        client = storage.Client(project=project_id)

        # 업로드 파일 json화
        json_res = json.dumps(final_results)
        json_count = json.dumps(emotion_counts)
        
        # 파일 이름 설정
        original_filename = get_filename_from_url(request.video_url)
        emotion_res_filename = f"{original_filename}_emotion_res.json"
        emotion_count_filename = f"{original_filename}_emotion_count.json"
        
        # 스토리지 폴더 밀 이름 설정
        res_folder = "pretzel-emotionResAI/"
        count_folder = "pretzel-emotionCountAI/"
        res_blob_name = res_folder + emotion_res_filename
        count_blob_name = count_folder + emotion_count_filename

        # 결과 파일 업로드
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(res_blob_name)
        blob.upload_from_string(json_res, content_type='application/json')
        
        # 카운트 파일 업로드
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(count_blob_name)
        blob.upload_from_string(json_count, content_type='application/json')
        
        return 1
    
    except Exception as e:
        print(f"예외 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/worlds_subtitle_movie/")
def worlds_subtitle_movie(subtitle_request: SubtitleRequest):
    print("자막 번역 요청을 받았습니다.")
    try:
        translations = translate_subtitle(subtitle_request)
        return JSONResponse(content=translations)
    except Exception as e:
        print(f"예외 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("서버를 시작합니다...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

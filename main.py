import uvicorn  # uvicorn을 임포트하여 ASGI 서버로 사용
from fastapi import FastAPI, HTTPException, File, UploadFile  # FastAPI와 HTTP 예외 처리를 위한 모듈 임포트
from fastapi.responses import JSONResponse, FileResponse  # JSON 응답과 파일 응답을 위해 필요한 모듈 임포트
from pydantic import BaseModel  # 데이터 검증을 위한 Pydantic BaseModel 임포트
import requests  # HTTP 요청을 위해 requests 라이브러리 임포트
import os  # 운영 체제 인터페이스를 위한 os 모듈 임포트
import cv2  # OpenCV를 사용하여 비디오 처리
import gc  # 가비지 컬렉션을 위해 gc 모듈 임포트
import json  # JSON 처리를 위한 모듈 임포트
from deepface import DeepFace  # 얼굴 인식 및 분석을 위한 DeepFace 임포트
from scipy.spatial import distance  # 벡터 간의 거리 계산을 위해 scipy.spatial의 distance 임포트
from tqdm import tqdm  # 진행 상태 표시를 위한 tqdm 임포트

# 비디오에서 오디오 추출 및 음악 감정 분석 유틸리티 임포트
from emotion_music_movie_util import extract_audio_from_video, analyze_music_mood
# 얼굴 인식 유틸리티 임포트
from actor_face_movie_util import adjust_brightness_contrast, embeddings, actors, convert_seconds_to_hms

from kobert import extract_str, kobert_eval;

# FastAPI 앱 초기화
app = FastAPI()

# 비디오 URL을 받기 위한 Pydantic 모델 정의
class VideoURL(BaseModel):
    url: str  # URL 문자열 필드

# 결과를 저장할 디렉터리 생성
if not os.path.exists("results"):
    os.makedirs("results")

# 배우 얼굴 인식 엔드포인트 정의
@app.post("/actor_face_movie/")
async def actor_face_movie(video_url: VideoURL):
    try:
        print("Request received for actor face recognition.")  # 요청 수신 로그 출력
        if not os.path.exists("videos"):
            os.makedirs("videos")  # 비디오 저장 폴더가 없으면 생성

        print("Starting download...")  # 다운로드 시작 로그 출력
        response = requests.get(video_url.url, stream=True)  # 비디오 URL에서 스트리밍 다운로드
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download video file")  # 다운로드 실패 시 예외 발생

        file_location = f"videos/{os.path.basename(video_url.url)}"  # 비디오 파일 저장 위치 설정
        with open(file_location, "wb") as file_object:
            for chunk in response.iter_content(chunk_size=8192):
                file_object.write(chunk)  # 비디오 파일을 8192바이트씩 저장
        print("Download complete.")  # 다운로드 완료 로그 출력

        print("Starting face recognition...")  # 얼굴 인식 시작 로그 출력
        cap = cv2.VideoCapture(file_location)  # OpenCV로 비디오 캡처 객체 생성
        fps = cap.get(cv2.CAP_PROP_FPS)  # 비디오의 초당 프레임 수 가져오기
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 비디오의 총 프레임 수 가져오기
        threshold = 0.6  # 유사도 임계값 설정
        interval = 30  # 프레임 간격 설정
        min_interval = 60  # 최소 간격 설정
        timestamps = {actor: [] for actor in actors}  # 각 배우별 타임스탬프를 저장할 딕셔너리 초기화

        with tqdm(total=frame_count // interval) as pbar:  # 진행 상태 표시를 위한 tqdm 사용
            while cap.isOpened():
                frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # 현재 프레임 위치 가져오기
                ret, frame = cap.read()  # 프레임 읽기
                if not ret:
                    break  # 프레임 읽기 실패 시 종료
                if frame_pos % interval == 0:
                    try:
                        adjusted_frame = adjust_brightness_contrast(frame, brightness=30, contrast=30)  # 밝기 및 대비 조정
                        detected_faces = DeepFace.represent(adjusted_frame, model_name="Facenet", enforce_detection=False)  # 얼굴 임베딩 추출
                        if detected_faces is None or len(detected_faces) == 0:
                            continue  # 얼굴이 감지되지 않으면 다음 프레임으로
                        for face in detected_faces:
                            face_embedding = face["embedding"]  # 얼굴 임베딩 추출
                            for actor, actor_embedding in embeddings.items():
                                similarity = distance.cosine(actor_embedding, face_embedding)  # 유사도 계산
                                if similarity < threshold:
                                    timestamp = frame_pos / fps  # 타임스탬프 계산
                                    if not timestamps[actor] or (timestamp - timestamps[actor][-1] > min_interval):
                                        timestamps[actor].append(timestamp)  # 타임스탬프 저장
                                    break
                    except Exception as e:
                        print(f"Error analyzing frame {frame_pos}: {e}")  # 예외 발생 시 로그 출력
                    pbar.update(1)  # 진행 상태 업데이트
                    gc.collect()  # 가비지 컬렉션 실행
        cap.release()  # 비디오 캡처 객체 해제

        final_results = []  # 최종 결과 리스트 초기화
        for actor, ts in timestamps.items():
            if len(ts) > 100:
                continue  # 타임스탬프가 100개를 초과하면 제외
            for t in ts:
                final_results.append({"time": convert_seconds_to_hms(t), "label": actor})  # 결과 저장
        
        print("Face recognition complete.")  # 얼굴 인식 완료 로그 출력

        # 결과를 JSON 파일로 저장
        result_file = "results/actor_face_results.json"
        with open(result_file, "w") as f:
            json.dump({"actor_timestamps": final_results}, f)

        return FileResponse(result_file)  # JSON 파일 응답 반환

    except Exception as e:
        print(f"Exception: {str(e)}")  # 예외 발생 시 로그 출력
        raise HTTPException(status_code=500, detail=str(e))  # 예외 응답 반환

# 음악 감정 분석 엔드포인트 정의
@app.post("/emotion_music_movie/")
async def emotion_music_movie(video_url: VideoURL):
    try:
        print("Request received for music emotion analysis.")  # 요청 수신 로그 출력
        if not os.path.exists("videos"):
            os.makedirs("videos")  # 비디오 저장 폴더가 없으면 생성

        print("Starting download...")  # 다운로드 시작 로그 출력
        response = requests.get(video_url.url, stream=True)  # 비디오 URL에서 스트리밍 다운로드
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download video file")  # 다운로드 실패 시 예외 발생

        file_location = f"videos/{os.path.basename(video_url.url)}"  # 비디오 파일 저장 위치 설정
        with open(file_location, "wb") as file_object:
            for chunk in response.iter_content(chunk_size=8192):
                file_object.write(chunk)  # 비디오 파일을 8192바이트씩 저장
        print("Download complete.")  # 다운로드 완료 로그 출력

        print("Extracting audio from video...")  # 비디오에서 오디오 추출 시작 로그 출력
        audio_file = file_location.replace(".mp4", ".wav")  # 오디오 파일 경로 설정
        extract_audio_from_video(file_location, audio_file)  # 오디오 추출
        print("Audio extraction complete.")  # 오디오 추출 완료 로그 출력

        print("Analyzing music mood...")  # 음악 감정 분석 시작 로그 출력
        mood_results = analyze_music_mood(audio_file)  # 음악 감정 분석 수행
        print("Music mood analysis complete.")  # 음악 감정 분석 완료 로그 출력

        final_results = [{"time": result["timestamp"], "label": result["result"]["label"]} for result in mood_results]  # 최종 결과 리스트 생성

        print("Processing complete.")  # 처리 완료 로그 출력

        # 결과를 JSON 파일로 저장
        result_file = "results/music_mood_results.json"
        with open(result_file, "w") as f:
            json.dump({"mood_results": final_results}, f)

        return FileResponse(result_file)  # JSON 파일 응답 반환

    except Exception as e:
        print(f"Exception: {str(e)}")  # 예외 발생 시 로그 출력
        raise HTTPException(status_code=500, detail=str(e))  # 예외 응답 반환
    
@app.post("/kobert/")
async def kobert(subtitle_url: str):
    try:
        print("Request received for music emotion analysis.")  # 요청 수신 로그 출력 
        print("KoBert 작동중...")  # 요청 수신 로그 출력
        subtitle_url = "https://storage.googleapis.com/pretzel-movie/"+subtitle_url
        response = requests.get(subtitle_url, stream=True)
        response.encoding = 'utf-8'
        subtitle_text = response.text
        subtitle_text = subtitle_text.replace('\r\n', '\n').replace('\r', '\n')
        kobert_eval(subtitle_text)
        return 1

    except Exception as e:
        print(f"Exception: {str(e)}")  # 예외 발생 시 로그 출력
        raise HTTPException(status_code=500, detail=str(e))  # 예외 응답 반환

# 애플리케이션 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # uvicorn을 사용하여 FastAPI 애플리케이션 실행

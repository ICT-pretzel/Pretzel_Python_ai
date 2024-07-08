import os
import cv2
import gc
import pickle
from deepface import DeepFace
from scipy.spatial import distance
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import requests
from tqdm import tqdm
import uuid
import json
from datetime import datetime
from urllib.parse import urlparse

router = APIRouter()

class VideoURL(BaseModel):
    url: str
    actors: dict

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    brightness = max(-127, min(127, brightness))
    contrast = max(-127, min(127, contrast))
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)
    return image

def convert_seconds_to_hms(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:05.2f}"

def update_actors(actors, embeddings):
    if not os.path.exists("images"):
        os.makedirs("images")
    for actor, img_url in actors.items():
        try:
            response = requests.get(img_url, stream=True)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Failed to download image for actor {actor}: {e}")

        img_path = os.path.join("images", f"{actor.replace(' ', '_')}.jpg")
        with open(img_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        try:
            embedding = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=True)[0]["embedding"]
            embeddings[actor] = embedding
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Failed to generate embedding for actor {actor}: {e}")

    with open("actor_embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)

def load_embeddings():
    embeddings = {}
    if os.path.exists("actor_embeddings.pkl"):
        with open("actor_embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
    return embeddings

def process_video(video_url, embeddings, threshold=0.6, interval=30, min_interval=60):
    try:
        response = requests.get(video_url, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="비디오 파일 다운로드에 실패했습니다")

        temp_video_id = uuid.uuid4().hex
        file_location = f"temp_video_{temp_video_id}.mp4"
        with open(file_location, "wb") as file_object:
            for chunk in response.iter_content(chunk_size=8192):
                file_object.write(chunk)

        cap = cv2.VideoCapture(file_location)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        timestamps = {actor: [] for actor in embeddings.keys()}

        with tqdm(total=frame_count // interval) as pbar:
            while cap.isOpened():
                frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_pos % interval == 0:
                    try:
                        adjusted_frame = adjust_brightness_contrast(frame, brightness=30, contrast=30)
                        detected_faces = DeepFace.represent(adjusted_frame, model_name="Facenet", enforce_detection=False)
                        if detected_faces is None or len(detected_faces) == 0:
                            continue
                        for face in detected_faces:
                            face_embedding = face["embedding"]
                            for actor, actor_embedding in embeddings.items():
                                similarity = distance.cosine(actor_embedding, face_embedding)
                                if similarity < threshold:
                                    timestamp = frame_pos / fps
                                    if not timestamps[actor] or (timestamp - timestamps[actor][-1] > min_interval):
                                        timestamps[actor].append(timestamp)
                                    break
                    except Exception as e:
                        print(f"프레임 {frame_pos} 분석 중 오류가 발생했습니다: {e}")
                    pbar.update(1)
                    gc.collect()
        cap.release()

        try:
            os.remove(file_location)
        except OSError as e:
            print(f"Error deleting temp video file: {e}")

        final_results = []
        for actor, ts in timestamps.items():
            if len(ts) > 100:
                continue
            for t in ts:
                final_results.append({"time": convert_seconds_to_hms(t), "label": actor})

        return final_results
    except Exception as e:
        print(f"Exception in process_video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_filename_from_url(url: str) -> str:
    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path).split('.')[0]

@router.post("/")
async def actor_face_movie(video_url: VideoURL):
    print("배우 얼굴 인식 요청을 받았습니다.")
    try:
        update_actors(video_url.actors, embeddings)
        print("배우 정보를 성공적으로 업데이트했습니다.")

        final_results = process_video(video_url.url, embeddings)
        
        video_filename = get_filename_from_url(video_url.url)

        # 결과를 results 폴더에 저장
        os.makedirs('results', exist_ok=True)
        filename = f"{video_filename}_actor_face_results.json"
        with open(os.path.join('results', filename), 'w', encoding='utf-8') as f:
            json.dump({"actor_timestamps": final_results}, f, ensure_ascii=False, indent=4)

        return JSONResponse(content={"actor_timestamps": final_results})

    except Exception as e:
        print(f"예외 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

embeddings = load_embeddings()
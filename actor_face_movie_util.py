import os
import cv2
import gc
import pickle
from deepface import DeepFace
from scipy.spatial import distance
from fastapi import HTTPException
import requests
from tqdm import tqdm
import uuid
import numpy as np
from PIL import Image, ImageEnhance
from mtcnn import MTCNN
import tempfile

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
    return f"{h:02}:{m:02}:{s:06.3f}"

# 이미지 전처리
def preprocess_image_from_url(img_url):
    try:
        # url에서 이미지 데이터를 불러오고, 유효성 검사
        response = requests.get(img_url, stream=True)
        response.raise_for_status()
        
        # 이미지 데이터를 numpy배열데이터로 변환
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        
        # 이미지 numpy배열 데이터를 컬러이미지로 디코딩하기
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to load image from URL.")
        
        # 얼굴 인식 라이브러리 MTCNN에서 함수 선언
        detector = MTCNN()
        
        # 이미지에서 얼굴 이미지 찾기
        faces = detector.detect_faces(img)
        
        if len(faces) == 0:
            raise ValueError("No face detected in the image.")
        
        # 찾아낸 얼굴 이미지 좌표를 원본 이미지에서 추출
        x, y, w, h = faces[0]['box']
        face_img = img[y:y+h, x:x+w]
        
        # 이미지 크기를 160,160으로 일관화
        face_resized = cv2.resize(face_img, (160, 160))
        
        # PIL라이브러리로 이미지의 명암을 1.5비율로 극대화 전처리 작업
        face_pil = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
        enhancer = ImageEnhance.Contrast(face_pil)
        face_enhanced = enhancer.enhance(1.5)  # Adjust contrast factor as needed
        return face_enhanced
        
    except (requests.exceptions.RequestException, ValueError, IndexError) as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")


def update_actors(actors, embeddings):
    for actor, img_url in actors.items():
        try:
            # 이미지 전처리 함수로 전처리
            img_url_full = "https://image.tmdb.org/t/p/original" + img_url
            preprocessed_img = preprocess_image_from_url(img_url_full)
            
            # DeepFace.represent의 인자로 보낼 이미지 경로를 설정하기 위해 임의의 파일 생성
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_img:
                temp_img_path = temp_img.name
                preprocessed_img.save(temp_img_path)
                
                # Deepface 임베딩
                embedding = DeepFace.represent(img_path=temp_img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                embeddings[actor] = embedding
            
                # 임시 파일 삭제
                temp_img.close()
                os.remove(temp_img_path)
            
        except (requests.exceptions.RequestException, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Failed to process image for actor {actor}: {e}")

    # 임베딩 데이터를 파일에 저장(실제 분석 모델에서 사용하기 위해)
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
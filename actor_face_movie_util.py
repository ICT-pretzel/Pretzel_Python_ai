import os  # OS 모듈을 불러옵니다
import cv2  # OpenCV 모듈을 불러옵니다
import gc  # 가비지 컬렉션 모듈을 불러옵니다
from deepface import DeepFace  # DeepFace 모듈을 불러옵니다
from scipy.spatial import distance  # 거리 계산 모듈을 불러옵니다
import pickle  # 피클 모듈을 불러옵니다

# 배우들의 이름과 이미지 경로를 딕셔너리로 정의합니다
actors = {
    "Tom Cruise": "profiles\Tom_Cruise.png",
}

embeddings = {}  # 배우 임베딩을 저장할 딕셔너리를 생성합니다
if not os.path.exists("actor_embeddings.pkl"):  # 임베딩 파일이 존재하지 않는 경우
    for actor, img_path in actors.items():  # 각 배우와 이미지 경로를 순회합니다
        try:
            embedding = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=True)[0]["embedding"]  # 얼굴 임베딩을 계산합니다
            embeddings[actor] = embedding  # 임베딩을 딕셔너리에 저장합니다
        except ValueError as e:
            print(f"Error processing {img_path}: {e}")  # 오류가 발생하면 출력합니다

    with open("actor_embeddings.pkl", "wb") as f:  # 임베딩을 피클 파일로 저장합니다
        pickle.dump(embeddings, f)
else:
    with open("actor_embeddings.pkl", "rb") as f:  # 피클 파일이 존재하는 경우
        embeddings = pickle.load(f)  # 임베딩을 로드합니다

def adjust_brightness_contrast(image, brightness=0, contrast=0):  # 밝기와 대비를 조정하는 함수
    brightness = max(-127, min(127, brightness))  # 밝기 범위를 제한합니다
    contrast = max(-127, min(127, contrast))  # 대비 범위를 제한합니다
    if brightness != 0:  # 밝기가 0이 아닌 경우
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)  # 밝기를 조정합니다
    if contrast != 0:  # 대비가 0이 아닌 경우
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)  # 대비를 조정합니다
    return image

def convert_seconds_to_hms(seconds):  # 초를 HH:MM:SS 형식으로 변환하는 함수
    h = int(seconds // 3600)  # 시 계산
    m = int((seconds % 3600) // 60)  # 분 계산
    s = seconds % 60  # 초 계산
    return f"{h:02}:{m:02}:{s:05.2f}"  # 형식에 맞게 문자열로 반환

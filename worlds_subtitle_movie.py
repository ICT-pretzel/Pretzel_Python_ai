import os
import re
import tempfile
from urllib.parse import unquote
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google.cloud import storage, translate_v3 as translate
from google.api_core.exceptions import ServiceUnavailable
from google.api_core.retry import Retry

router = APIRouter()

class SubtitleRequest(BaseModel):
    url: str

def preprocess_text(text):
    text = re.sub(r'\[.*?\]', '', text)  # 대괄호 안의 내용을 제거
    text = re.sub(r'\(.*?\)', '', text)  # 소괄호 안의 내용을 제거
    text = re.sub(r'\{.*?\}', '', text)  # 중괄호 안의 내용을 제거
    text = re.sub(r'\<.*?\>', '', text)  # 꺾쇠 괄호 안의 내용을 제거
    text = text.strip()  # 텍스트 양쪽 끝의 공백 제거
    return text  # 전처리된 텍스트 반환

def postprocess_text(text):
    text = re.sub(r'\s+', ' ', text)  # 여러 개의 공백을 하나의 공백으로 대체
    text = text.strip()  # 텍스트 양쪽 끝의 공백 제거
    text = text[0].upper() + text[1:] if text else text  # 첫 글자를 대문자로 변환
    return text  # 후처리된 텍스트 반환

def translate_text(text, target_language):
    if target_language == "ko":  # 타겟 언어가 한국어인 경우
        return text  # 번역을 생략하고 원본 텍스트 반환
    client = translate.TranslationServiceClient()  # Google Translate 클라이언트 초기화
    parent = f"projects/translate-movie-427703/locations/global"  # 프로젝트와 위치 설정
    
    retry = Retry(initial=1.0, maximum=10.0, multiplier=2.0, deadline=60.0)  # 재시도 설정
    try:
        response = client.translate_text(
            request={
                "parent": parent,  # 부모 경로 설정
                "contents": [text],  # 번역할 내용 설정
                "mime_type": "text/plain",  # MIME 타입 설정
                "source_language_code": "ko",  # 소스 언어 코드 설정
                "target_language_code": target_language,  # 타겟 언어 코드 설정
            },
            retry=retry  # 재시도 로직 적용
        )
        return response.translations[0].translated_text  # 번역된 텍스트 반환
    except ServiceUnavailable as e:  # 서비스 사용 불가 예외 처리
        print(f"Service is unavailable: {e}")  # 예외 메시지 출력
        return text  # 번역 실패 시 원본 텍스트 반환

def srt_to_vtt_time(srt_time):
    return srt_time.replace(',', '.')  # SRT 시간 형식을 VTT 시간 형식으로 변환

def download_srt_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()  # Google Cloud Storage 클라이언트 초기화
    bucket = storage_client.bucket(bucket_name)  # 버킷 객체 가져오기
    blob = bucket.blob(source_blob_name)  # 블롭 객체 가져오기
    blob.download_to_filename(destination_file_name)  # 블롭을 로컬 파일로 다운로드
    print(f"Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}")

def translate_srt_to_vtt(input_srt_path, target_languages, base_filename):
    with open(input_srt_path, 'r', encoding='utf-8') as file:  # SRT 파일 열기
        subs = file.read().split('\n\n')  # 각 자막을 구분하여 리스트로 분리
    
    results = {}
    output_dir = "translated_vtt_files"  # 번역된 VTT 파일들을 저장할 디렉토리
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성

    for target_language in target_languages:  # 각 타겟 언어에 대해 반복
        print(f"Translating to {target_language}...")  # 진행 상황 출력
        vtt_content = "WEBVTT\n\n"  # VTT 파일 헤더 초기화
        for index, sub in enumerate(subs):  # 각 자막 블록에 대해 반복
            if sub.strip():  # 자막 블록이 비어있지 않은 경우
                parts = sub.split('\n')  # 자막 블록을 줄 단위로 분리
                idx = parts[0]  # 자막 인덱스
                time = parts[1]  # 자막 시간 정보
                time = srt_to_vtt_time(time)  # SRT 시간 형식을 VTT 시간 형식으로 변환
                original_text = ' '.join(parts[2:])  # 자막 텍스트 결합
                preprocessed_text = preprocess_text(original_text)  # 자막 텍스트 전처리
                translated_text = translate_text(preprocessed_text, target_language)  # 자막 텍스트 번역
                postprocessed_text = postprocess_text(translated_text)  # 번역된 텍스트 후처리
                
                vtt_content += f"{idx}\n{time}\n{postprocessed_text}\n\n"  # VTT 콘텐츠에 추가
            
            # 진행 상황 출력
            progress = (index + 1) / len(subs) * 100  # 진행률 계산
            print(f"Progress: {progress:.2f}% for language {target_language}")  # 진행률 출력
        
        output_file_path = os.path.join(output_dir, f"{base_filename}_{target_language}.vtt")  # 출력 파일 경로 설정
        with open(output_file_path, 'w', encoding='utf-8') as output_file:  # VTT 파일 쓰기
            output_file.write(vtt_content)  # VTT 콘텐츠 저장

        results[target_language] = output_file_path  # 결과 딕셔너리에 파일 경로 저장
    
    return results

def translate_subtitle(subtitle_request: SubtitleRequest):
    gcs_url = subtitle_request.url
    target_languages = ["ko", "ja", "zh", "en", "fr", "de", "es", "it", "pt", "ru", "hi"]
    
    if not gcs_url:
        raise HTTPException(status_code=400, detail="URL is required")

    try:
        bucket_name, source_blob_name = re.match(r"https://storage.googleapis.com/([^/]+)/(.+)", gcs_url).groups()
        source_blob_name = unquote(source_blob_name)  # URL 디코딩
        base_filename = os.path.splitext(os.path.basename(source_blob_name))[0]  # 기본 파일 이름 추출 (확장자 제거)
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            download_srt_from_gcs(bucket_name, source_blob_name, temp_file.name)
            translations = translate_srt_to_vtt(temp_file.name, target_languages, base_filename)
            return JSONResponse(content=translations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from safetensors.torch import load_file
import re

def extract_str(srt_file_path):
    with open(srt_file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2}.\d{3} --> \d{2}:\d{2}:\d{2}.\d{3})\n(.*?)\n\n', re.DOTALL)
    matches = pattern.findall(content)

    subtitles = []
    for match in matches:
        index, timestamp, text = match
        # 텍스트의 여러 줄을 하나로 합치기
        text = text.replace('\n', ' ').strip()
        subtitles.append((timestamp, text))
    
    return subtitles

# 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("monologg/kobert")

# safetensors 파일에서 가중치 로드
state_dict = load_file("kobert_safetensors\model.safetensors")

# 모델 로드
model = BertForSequenceClassification.from_pretrained(
    "monologg/kobert",
    num_labels=5,
    state_dict=state_dict
)

# 예측할 텍스트 추출
file_path = '스물.srt'
subtitles = extract_str(file_path)

# 모델을 평가 모드로 설정
model.eval()

# 타임스탬프별로 텍스트 예측 수행
for timestamp, text in subtitles:
    # 입력 텍스트를 토크나이징
    inputs = tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors="pt")
    
    # 입력 데이터를 모델에 전달하여 예측 수행
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 로짓(logits) 추출 및 소프트맥스 함수 적용하여 확률 값 계산
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # 예측된 레이블 출력
    predicted_label_index  = torch.argmax(probabilities, dim=1).item()
    labels = ['무감정', '슬픔', '분노', '기쁨', '불안']
    predicted_label = labels[predicted_label_index]
    
    # 결과 출력
    print(f"Timestamp: {timestamp}")
    print(f"Text: {text}")
    print(f"Predicted label: {predicted_label}\n")
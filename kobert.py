import torch
from transformers import BertTokenizer, BertForSequenceClassification
from safetensors.torch import load_file
import re

def extract_str(content):
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2}.\d{3} --> \d{2}:\d{2}:\d{2}.\d{3})\n(.*?)\n\n', re.DOTALL)
    matches = pattern.findall(content)
    print(matches)
    subtitles = []
    for match in matches:
        index, timestamp, text = match
        # 각 줄을 분리하고 [로 시작해서 ]로 끝나는 부분과 (로 시작해서 )로 끝나는 부분을 제거
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # [로 시작해서 ]로 끝나는 부분 제거
            line = re.sub(r'\[.*?\]', '', line)
            # (로 시작해서 )로 끝나는 부분 제거
            line = re.sub(r'\(.*?\)', '', line)
            cleaned_lines.append(line.strip())
        # 텍스트의 여러 줄을 하나로 합치기
        text = ' '.join(cleaned_lines).strip()
        if text:  # 빈 문자열이 아닌 경우에만 추가
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

# 모델을 평가 모드로 설정
model.eval()

# 타임스탬프별로 텍스트 예측 수행
def kobert_eval(text):
    content = extract_str(text)
    timelog = []
    bert_count={        
        "sad": 0,
        "hap": 0,
        "neu": 0,
        "ang": 0,
        "anx": 0
    }
    for timestamp, text in content:
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
        labels = ['neu', 'sad', 'ang', 'hap', 'anx']
        predicted_label = labels[predicted_label_index]
        bert_count[predicted_label] += 1
        result = {"time": timestamp,"label": predicted_label}
        timelog.append(result)
    return timelog, bert_count
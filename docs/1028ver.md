# AI-BE Project (1028 Version)

AI 서버와 백엔드 서버 연동을 위한 AI 서비스 모듈입니다.  
1028 버전 기준 파일 구조, 각 모듈 사용법, 파이프라인 예시를 정리했습니다.

## Installation
```python
pip install -r requirements.txt
```
> GitHub의 파일 크기 제한(100MB)으로 인해 [checkpoints/](https://drive.google.com/drive/folders/1NLY7QJuLbwZaZUeSBRGEyPdA_irSyelO?usp=sharing) 와 [data/](https://drive.google.com/drive/folders/1o12-FR_m8ddtWtmll3r0lQ3KAptRZEpz?usp=sharing) 디렉토리는 저장소에 포함되지 않았습니다.
아래 Google Drive 링크에서 다운로드 받아 프로젝트 루트에 배치해야 합니다.

배치 후 최종 구조 예시:
```bash
project_root/
├── libs/
├── service/
├── model_manager/
├── modelS/
├── checkpoints/         # from Google Drive
├── data/                # from Google Drive
├── requirements.txt
└── README.md
```


## Project Structure
```bash
project_root/
├── main.py                         # FastAPI 서버 진입점 (BE팀 관리)
│
├── api/                            # API 엔드포인트 정의 (BE팀)
│   ├── nia.py                      # 피부 분석 (1-1)
│   ├── feedback.py                 # 피부 피드백 (1-2)
│   ├── product.py                  # 제품 추천 (1-3)
│   ├── style.py                    # 스타일 추천 (2-1)
│   ├── makeup.py                   # 메이크업 시뮬레이션 (2-2)
│   └── customization.py            # 커스터마이징 (3-1)
│
├── service/                        # 추론 로직 (AI팀)
│   ├── nia_service.py
│   ├── feedback_service.py
│   ├── product_service.py
│   ├── style_service.py
│   ├── makeup_service.py           
│   └── customization_service.py
│
├── model_manager/                  # 모델 로딩 및 캐시 관리 (AI팀)
│   ├── nia_manager.py
│   ├── feedback_manager.py
│   ├── product_manager.py
│   ├── clip_manager.py
│   ├── makeup_manager.py          
│   └── customization_manager.py
│
├── checkpoints/                    # 학습된 모델 가중치 (AI팀)
│   ├── nia/
│   │   ├── class/
│   │   │   ├── dryness/state_dict.bin
│   │   │   ├── pigmentation/state_dict.bin
│   │   │   ├── pore/state_dict.bin
│   │   │   ├── sagging/state_dict.bin
│   │   │   └── wrinkle/state_dict.bin
│   │   └── regression/
│   │       ├── elasticity_R2/state_dict.bin
│   │       ├── moisture/state_dict.bin
│   │       ├── pigmentation/state_dict.bin
│   │       ├── pore/state_dict.bin
│   │       └── wrinkle_Ra/state_dict.bin
│   │
│   ├── customization/customization.pt
│   ├── style/clip-vit-base.pt
│   └── makeup/                     
│       ├── pytorch_model.bin        
│       ├── pytorch_model_1.bin  
│       └── pytorch_model_2.bin      
│
├── data/
│   ├── product.xlsx
│   ├── style-recommendation/...
│   ├── inference.jpg
│   ├── predictions.json             # nia 실행 시 생성
│   ├── output/                      # makeup 결과 이미지 저장 폴더 (실행 시 자동 생성)
│   └── test_imgs_makeup/            # 로컬 테스트용
│
├── libs/                            # 내부 공용 모듈
│   ├── __init__.py                 
│   ├── face_utils.py
│   ├── pipeline_sd15.py
│   ├── spiga_draw.py
│   └── detail_encoder/
│       ├── __init__.py           
│       └── encoder_plus.py
│ 
├── requirements_org/               # 모듈별 requirements
│   ├── requirements_customization.txt
│   ├── requirements_feedback.txt
│   ├── requirements_nia.txt
│   ├── requirements_product.txt
│   ├── requirements_makeup.txt
│   └── requirements_style.txt
│
├── requirements.txt                # 통합 requirements
├── Dockerfile                      # (선택) 배포용 컨테이너 환경
└── README.md                       # 실행법/구조 설명
```


## Module: NIA (Skin Analysis)
#### Usage Example
```python
import base64
from app.service.nia_service import run_inference

# 이미지 파일을 Base64로 인코딩
with open("얼굴이미지.jpg", "rb") as f:
  image_base64 = base64.b64encode(f.read()).decode()

request = {
  "image_base64": image_base64
}
result = run_inference(request)
print(result)
```

#### Workflow
- Base64 → PIL 변환
- 얼굴 크롭 (MediaPipe, 기본 활성화)
- 256x256 정규화
- Classification (5개) + Regression (5개) 모델 추론
- 0~100 점수로 정규화 (높을수록 좋은 상태)
- 결과 JSON 반환 및 data/predictions.json 저장

#### Notes
- 체크포인트 경로: checkpoints/nia
- 첫 호출 시 모델 로딩 시간 있음 (캐시 후 빠름)
- 모든 점수는 0~100으로 정규화됨 (높을수록 좋음)

#### Interpretation
- 점수가 낮을수록 관리 필요한 것

## Module: LLM Feedback
#### Environment Variable
```bash
export GEMINI_API_KEY="여기에_키"
export FEEDBACK_PREDICTIONS_PATH="data/predictions.json_경로입력"   # 선택
```
##### 코랩에서 진행시
```bash
import os
# 환경 변수 등록 (세션 전체에서 유효)
os.environ["GEMINI_API_KEY"] = "여기에_키"
os.environ["FEEDBACK_PREDICTIONS_PATH"] = "data/predictions.json_경로입력" #선택
```
#### Usage Example
```python
from app.service.feedback_service import run_inference

req = {}
print(run_inference(req))
```
#### Notes
- 고정 설명문 + 점수 JSON → LLM 입력 생성
- Gemini 모델 1회 로드 → 피드백 문장 생성

## Module: Product Recommendation
#### API Key
```python
import os
os.environ["GEMINI_API_KEY"] = "your-api-key-here"
```

#### Usage Example
```python
import os
from app.service.product_service import run_inference

# Gemini API 키 설정
os.environ["GEMINI_API_KEY"] = "your-api-key-here"

request = {
  "skin_analysis": {
      "dryness": 55,
      "pigmentation": 65,
      "pore": 50,
      "sagging": 70,
      "wrinkle": 45,
      "pigmentation_reg": 68,
      "moisture_reg": 60,
      "elasticity_reg": 72,
      "wrinkle_reg": 48,
      "pore_reg": 52
  },
  "recommended_categories": ["moisture", "wrinkle", "pore"],
  "filtered_products": [
      {
          "product_id": "P001",
          "product_name": "하이드레이팅 세럼",
          "brand": "라로슈포제",
          "category": "moisture",
          "price": 35000,
          "review_score": 4.5,
          "review_count": 1234,
          "ingredients": ["히알루론산", "글리세린", "세라마이드"]
      }
  ],
  "locale": "ko-KR"
}

result = run_inference(request)
print(result)
```

#### Notes
- Gemini API 키 필수 (환경변수로 설정)
- recommended_categories는 영문 전달 ("moisture", "elasticity",
"wrinkle", "pigmentation", "pore")
- 개별 제품 처리 실패 시에도 에러 메시지를 reason에 포함하여 반환
- 재시도 로직: API 호출 실패 시 최대 3회 재시도
- 타임아웃: 30초
    
#### 제품 추천 기준
백엔드는 아래 기준으로 recommended_categories를 결정:

- moisture_reg < 65 → "수분" 추천
- elasticity_reg < 60 → "탄력" 추천
- wrinkle_reg < 50 → "주름" 추천
- pigmentation_reg < 70 → "색소침착" 추천
- pore_reg < 55 → "모공" 추천


## Pipeline Flow (NIA → Feedback → Product)
#### Example
```python
from app.service.nia_service import run_inference as nia_inference
from app.service.feedback_service import run_inference as feedback_inference
from app.service.product_service import run_inference as product_inference
```

#### Flow
1. 사용자 이미지 업로드 → BE팀 → AI서버 NIA 분석
2. AI서버 predictions.json 생성
3. BE팀 → AI서버: Feedback 요청
4. BE팀 DB조회/필터링 후 → AI서버: Product Reason 요청
5. 최종 결과 사용자에게 전달


## Module: Style Recommendation
```python
import base64
from app.service.style_service import run_inference

with open("face.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

request = {
    "source_image_base64": image_base64,
    "keywords": ["pink blush", "warm tone", "red lip"]
}
result = run_inference(request, "data/style-recommendation")
print(result)
```

## Module: Makeup Simulation 
```python
pip install -r requirements_org/requirements_makeup.txt
python service/makeup_service.py
```


## Module: Customization
```python
import sys, base64
sys.path.insert(0, "/content/drive/MyDrive")

from app.service.customization_service import run_inference

with open("face.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

request = {
    "base_image_base64": image_base64,
    "edits": [
        {"region": "skin", "intensity": 20},
        {"region": "blush", "intensity": 30},
        {"region": "lip", "intensity": 70},
        {"region": "eye", "intensity": 50},
    ]
}
result = run_inference(request)
```

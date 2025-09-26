# AI-BE Project (0926 Version)

AI 서버와 백엔드 서버 연동을 위한 AI 서비스 모듈입니다.  
0926 버전 기준 파일 구조, 각 모듈 사용법, 파이프라인 예시를 정리했습니다.

## Installation
```bash
pip install -r requirements.txt

GitHub의 파일 크기 제한(100MB)으로 인해 checkpoints/ 와 data/ 디렉토리는 저장소에 포함되지 않았습니다.
아래 Google Drive 링크에서 다운로드 받아 프로젝트 루트에 배치해야 합니다.
[Download checkpoints](https://drive.google.com/drive/folders/1NLY7QJuLbwZaZUeSBRGEyPdA_irSyelO?usp=sharing)
[Download data](https://drive.google.com/drive/folders/1o12-FR_m8ddtWtmll3r0lQ3KAptRZEpz?usp=sharing)

배치 후 최종 구조 예시:

arduino
코드 복사
project_root/
├── app/
├── checkpoints/         # from Google Drive
├── data/                # from Google Drive
├── requirements.txt
└── README.md
Project Structure
bash
project_root/
├── app/                         
│   ├── main.py                     # FastAPI 서버 진입점 (BE팀 관리)
│   │
│   ├── api/                        # API 엔드포인트 정의 (BE팀)
│   │   ├── nia.py                  # 피부 분석 (1-1)
│   │   ├── feedback.py             # 피부 피드백 (1-2)
│   │   ├── product.py              # 제품 추천 (1-3)
│   │   ├── style.py                # 스타일 추천 (2-1)
│   │   ├── makeup.py               # 메이크업 시뮬레이션 (2-2, 미완성)
│   │   └── customization.py        # 커스터마이징 (3-1)
│   │
│   ├── service/                    # 추론 로직 (AI팀)
│   │   ├── nia_service.py
│   │   ├── feedback_service.py
│   │   ├── product_service.py
│   │   ├── style_service.py
│   │   ├── makeup_service.py       # 미완성 
│   │   └── customization_service.py
│   │
│   └── model_manager/              # 모델 로딩 및 캐시 관리 (AI팀)
│       ├── nia_manager.py
│       ├── feedback_manager.py
│       ├── product_manager.py
│       ├── clip_manager.py        
│       ├── makeup_manager.py       # 미완성 
│       └── customization_manager.py
│
├── checkpoints/                    # 학습된 모델 가중치 (AI팀, Drive 제공)
│   ├── nia/...
│   ├── customization/customization.pt
│   ├── style/clip-vit-base.pt
│   └── makeup/_.pt                 # 미완성
│
├── data/                           # 정적 자원 (AI팀, Drive 제공)
│   ├── product.xlsx
│   ├── style-recommendation/...
│   ├── inference.jpg
│   └── predictions.json
│
├── requirements_parts/             # 모듈별 requirements 
│   ├── requirements_customization.txt
│   ├── requirements_feedback.txt    
│   ├── requirements_nia.txt    
│   ├── requirements_product.txt    
│   └── requirements_style.txt       
│
├── requirements.txt                # 통합 requirements 
├── Dockerfile                      # (선택) 배포용 컨테이너 환경
└── README.md                       # 실행법/구조 설명
Module: NIA (Skin Analysis)
Usage Example
python
코드 복사
import sys, base64
sys.path.insert(0, "/content/drive/MyDrive")  # 프로젝트 루트로 수정

from app.service.nia_service import run_inference

with open("얼굴이미지.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

request = {"image_base64": image_base64, "crop_face": True}
result = run_inference(request)
print(result)
Workflow
Base64 → PIL 변환 → 얼굴 크롭 (MediaPipe) → 256x256 정규화

Classification (5개) + Regression (5개) 모델 추론

결과 JSON 반환

Notes
체크포인트 경로: checkpoints/nia

첫 호출 시 모델 로딩 시간 있음 (캐시 후 빠름)

Regression 결과 음수 가능성 → 후처리 필요

Interpretation
Classification: 숫자 ↑ = 상태 나쁨

Regression:

moisture < 70 → 부족

elasticity_R2 < 0.7 → 부족

wrinkle_Ra > 30 → 주의

pigmentation > 250 → 주의

pore > 1800 → 주의

Module: LLM Feedback
Environment Variable
bash
코드 복사
export GEMINI_API_KEY="여기에_키"
Usage Example
python
코드 복사
from app.service.feedback_service import run_inference

req = {"predictions_json_path": "data/predictions.json"}
print(run_inference(req))
Module: Product Recommendation
API Key
python
코드 복사
# product_service.py
GEMINI_API_KEY = ""  # 실제 키 또는 환경 변수 사용
Usage Example
python
코드 복사
import sys
sys.path.insert(0, "/content/drive/MyDrive")

from app.service.product_service import run_inference

request = {
    "predictions": {...},   # predictions.json 내용
    "needs": ["moisture", "pore"],
    "filtered_products": [...],
    "locale": "ko-KR"
}
result = run_inference(request)
Notes
Gemini API 키 필수

predictions.json 필요

needs/category는 영문 전달

제품 수만큼 LLM 호출 → 호출 비용 고려

Pipeline Flow (NIA → Feedback → Product)
Example
python
코드 복사
from app.service.nia_service import run_inference as nia_inference
from app.service.feedback_service import run_inference as feedback_inference
from app.service.product_service import run_inference as product_inference
Flow
사용자 이미지 업로드 → BE팀 → AI서버 NIA 분석

AI서버 predictions.json 생성

BE팀 → AI서버: Feedback 요청

BE팀 DB조회/필터링 후 → AI서버: Product Reason 요청

최종 결과 사용자에게 전달

Module: Style Recommendation
bash
코드 복사
pip install -r requirements_style.txt
python
코드 복사
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
Module: Customization
bash
코드 복사
pip install -r requirements_customization.txt
python
코드 복사
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
Version Info
현재 문서는 0926 버전 기준입니다.

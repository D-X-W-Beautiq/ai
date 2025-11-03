# Beautiq AI- Backend (FastAPI) _ 1103ver

<br>

## 주요 구성

| 구분 | 설명 |
|------|------|
| 엔트리 | `main.py` — FastAPI 앱 생성, `/v1` 통합 마운트 |
| 라우터 통합 | `api/router.py` — `/v1/nia`, `/v1/feedback`, `/v1/product`, `/v1/style`, `/v1/makeup`, `/v1/custom` |
| 서비스 코드 | `service/*_service.py` — 각 단계별 추론 로직 |
| 모델 매니저 | `model_manager/*_manager.py` — 모델 로딩/캐싱/싱글턴 |
| 공통 스키마 | `schemas.py` — 모든 요청·응답 모델 (팀 계약서 역할) |
| 헬스체크 | `api/health.py` — `/health`, `/ready`, `/version` |
| 환경 설정 | `config.py` — GEMINI API 키, 경로, 체크포인트 설정 |
| 유틸리티 | `utils/base64_utils.py`, `utils/errors.py` |

<br>

## 프로젝트 구조

```

project_root/
├── main.py                      # FastAPI 앱 생성 및 /v1 마운트
├── api/
│   ├── router.py                # 모든 하위 라우터 통합
│   ├── nia.py                   # /v1/nia/analyze (피부 분석)
│   ├── feedback.py              # /v1/feedback/generate
│   ├── product.py               # /v1/product/reason
│   ├── style.py                 # /v1/style/recommend
│   ├── makeup.py                # /v1/makeup/simulate
│   ├── customization.py         # /v1/custom/apply
│   └── health.py                # /health, /ready, /version
├── service/
│   ├── nia_service.py           # 피부 분석 (NIA 모델)
│   ├── feedback_service.py      # LLM 피드백 생성 (Gemini)
│   ├── product_service.py       # 제품 추천 이유 생성 (Gemini)
│   ├── style_service.py         # CLIP 기반 스타일 추천
│   ├── makeup_service.py        # Stable Makeup 전이
│   └── customization_service.py # SegFormer 기반 커스터마이즈
├── model_manager/               # 모델 로딩 및 캐싱
├── utils/                       # base64, error handler 등
├── data/                        # 예시 데이터 및 결과 저장
├── checkpoints/                 # AI 모델 가중치
├── test.py                 
├── precompute_embeddings.py          
└── requirements.txt

````

<br>

## 실행 방법

#### 1. 클론 및 환경 세팅

```bash
git clone https://github.com/D-X-W-Beautiq/ai.git
cd ai
pip install --no-cache-dir -r requirements.txt
````

> ⚠️ **GitHub 파일 크기 제한(100MB)** 으로 인해
> `checkpoints/` 와 `data/` 폴더는 저장소에 포함되어 있지 않습니다.
> 아래 링크에서 다운로드 후 루트에 배치해주세요.

* **Checkpoints:** [🔗 Google Drive](https://drive.google.com/drive/folders/1NLY7QJuLbwZaZUeSBRGEyPdA_irSyelO?usp=sharing)
* **Data:** [🔗 Google Drive](https://drive.google.com/drive/folders/1o12-FR_m8ddtWtmll3r0lQ3KAptRZEpz?usp=sharing)


#### 2. 환경 변수 설정

```bash
export GEMINI_API_KEY="your_api_key_here"
```


#### 3. FastAPI 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 1200
```


#### 4. 임베딩 사전 계산 및 테스트

```bash
python precompute_embeddings.py
python test.py
```

> 모든 체인 성공 시 출력:

```
============================================================
전체 Pipeline 테스트 완료 (모든 필수 체인 통과)!
============================================================
생성/확인 파일:
  - data/predictions.json (NIA)
  - data/output/makeup_result.png (Makeup)
  - data/output/final_result.png (Customization)
```

<br>

## End-to-End 파이프라인

```
NIA(피부 분석)
   ↓
Feedback(피드백 생성)
   ↓
Product(추천 이유 생성)
   ↓
Style(스타일 추천)
   ↓
Makeup(메이크업 전이)
   ↓
Customization(커스터마이징)
```

✅ **엔드투엔드 파이프라인 구현 완료** (NIA → Feedback → Product → Style → Makeup → Customization 순서로 연결)  
⚠️ Makeup 단계에서 **모델 로딩 시간 약 200초 소요** (최적화 필요)

<br>

## 확인 및 개선 필요사항

| 항목                      | 상태       | 비고                          |
| ----------------------- | -------- | --------------------------- |
| Style 추천                | ✅ 정상     | Top-3 결과 반환                 |
| Makeup 전이               | ⚠️ 지연 발생 | 모델 캐싱 확인 필요                 |
| Customization           | ✅ 정상     | eyelid, lip, blush, skin 지원 |
| NIA/Feedback/Product 체인 | ✅ 정상     | 요청·응답 스키마 일관성 유지            |
| 임베딩 캐싱                  | ⚙️ 코드 완료 | 실제 속도 개선 테스트 필요             |
| 결과 퀄리티 평가               | 🚧 예정    | 팀 내부 논의 필요                  |

<br>

## ⭐BE팀 전달사항⭐

1. **현재 AI팀 서버 환경 제한으로 인해 마지막 테스트를 진행하지 못했습니다.**  
   전체 파이프라인이 정상적으로 동작하는 것은 확인하였습니다.  
   다만 **메이크업 단계에서 처리 시간이 상당히 오래 걸리는 문제(약 200초 내외)** 가 있습니다.  
   모델 로딩 과정에서 병목 현상이 발생하는 것으로 추정되며, 이에 대한 추가적인 최적화가 필요합니다.  
   현재 실험 가능한 환경이 제한되어 있어 우선 코드를 전달드리며 추후 서버 자원이 확보되면 로딩 최적화(모델 프리로드 등) 테스트를 진행할 예정입니다.  

2. **피부 분석(`NIA`) 부분은 현재 회귀값 기반으로만 처리되어 있습니다.**  
   즉, 점수 산출이 임시로 단순 회귀 결과값을 기준으로 변환되는 형태로 되어 있습니다.  
   추후 BE단에서 점수 변환 및 스케일링 로직을 통합 관리하실 수 있도록 관련 구조를 조정하셔야 할 수 있습니다.  

3. **`Product`(제품 추천) 처리 관련 확인이 필요합니다.**  
   AI팀은 BE로부터 `filtered_products` 리스트를 전달받아, 각 제품별 추천 이유(`reason`)를 생성하도록 구성되어 있습니다.  
   따라서 BE에서 전달되는 `filtered_products`의 구조가 아래 스키마(`ProductIn`)와 동일한지 반드시 확인 부탁드립니다.  

   ```python
   class ProductIn(BaseModel):
       product_id: str
       product_name: str
       brand: str
       category: str
       price: int
       review_score: float
       review_count: int
       ingredients: List[str]
   ```

   또한 추천 이유 생성은 `Gemini` 모델을 통해 이루어지므로,
   요청 시 필드 누락이나 포맷 불일치가 없도록 유의해주시면 감사하겠습니다.

4. **전체 엔드투엔드 파이프라인은 아래 순서로 연결되어 있습니다.**
   각 단계의 요청 및 응답은 정상적으로 이어지며,
   최종 결과물은 `data/output/` 경로에 저장됩니다.

   ```
   /v1/nia/analyze
   → /v1/feedback/generate
   → /v1/product/reason
   → /v1/style/recommend
   → /v1/makeup/simulate
   → /v1/custom/apply
   ```

   현재까지 모든 단계의 응답 구조는 스키마(`schemas.py`) 기준으로 일관되게 동작함을 확인하였습니다.
   다만 메이크업 단계에서의 생성 시간 및 GPU 리소스 활용 최적화가 필요하므로,
   서버 환경 설정 시 해당 부분을 고려해주시면 감사하겠습니다.

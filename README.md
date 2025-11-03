```markdown
# 💄 Beautiq AI Backend (FastAPI)

---

## ⚙️ 주요 구성

| 구분 | 설명 |
|------|------|
| **엔트리** | `main.py` — FastAPI 앱 생성, `/v1` 통합 마운트 |
| **라우터 통합** | `api/router.py` — `/v1/nia`, `/v1/feedback`, `/v1/product`, `/v1/style`, `/v1/makeup`, `/v1/custom` |
| **서비스 코드** | `service/*_service.py` — 각 단계별 추론 로직 |
| **모델 매니저** | `model_manager/*_manager.py` — 모델 로딩/캐싱/싱글턴 |
| **공통 스키마** | `schemas.py` — 모든 요청·응답 모델 (팀 계약서 역할) |
| **헬스체크** | `api/health.py` — `/health`, `/ready`, `/version` |
| **환경 설정** | `config.py` — GEMINI API 키, 경로, 체크포인트 설정 |
| **유틸리티** | `utils/base64_utils.py`, `utils/errors.py` |

---

## 📂 프로젝트 구조

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

---

## 🚀 실행 방법

### 1️⃣ 클론 및 환경 세팅

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

---

### 2️⃣ 환경 변수 설정

```bash
export GEMINI_API_KEY="your_api_key_here"
```

---

### 3️⃣ FastAPI 서버 실행

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --timeout-keep-alive 1200
```

---

### 4️⃣ 임베딩 사전 계산 및 테스트

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

---

## 🧩 End-to-End 파이프라인

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

✅ **엔드투엔드 파이프라인 구현 완료**
✅ NIA → Feedback → Product → Style → Makeup → Customization 순서로 연결
⚠️ Makeup 단계에서 **모델 로딩 시간 약 200초 소요** (최적화 필요)

---

## ⏳ 확인 및 개선 필요사항

| 항목                      | 상태       | 비고                          |
| ----------------------- | -------- | --------------------------- |
| Style 추천                | ✅ 정상     | Top-3 결과 반환                 |
| Makeup 전이               | ⚠️ 지연 발생 | 모델 캐싱 확인 필요                 |
| Customization           | ✅ 정상     | eyelid, lip, blush, skin 지원 |
| NIA/Feedback/Product 체인 | ✅ 정상     | 요청·응답 스키마 일관성 유지            |
| 임베딩 캐싱                  | ⚙️ 코드 완료 | 실제 속도 개선 테스트 필요             |
| 결과 퀄리티 평가               | 🚧 예정    | 팀 내부 논의 필요                  |

---

## 🤝 BE팀 전달사항

1. **서버 환경 제한으로 마지막 속도 테스트 미실시**

   * Makeup 단계에서 모델 로딩 시 **약 200초** 소요
   * 원인: `makeup_manager.py`에서 캐시 미적용 또는 GPU 초기화 지연
   * **해결 제안:**
     서버 기동 시점(`startup` 이벤트)에서 `load_model()` 프리로드 처리

2. **피부분석(NIA) 결과는 회귀값 기반 임시 처리**

   * 점수 변환 로직은 현재 AI단에서 수행
   * 향후 BE단 점수 스케일링 일원화 검토 필요

3. **Product 추천 이유 생성(`product_service.py`)**

   * AI단은 BE에서 전달받는 `filtered_products` 리스트 기반으로
     각 제품별 추천 이유(`reason`)를 생성
   * BE의 필드 구조가 `schemas.ProductIn`과 동일한지 확인 필요

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

4. **엔드투엔드 요청 순서**

   ```
   /v1/nia/analyze
   → /v1/feedback/generate
   → /v1/product/reason
   → /v1/style/recommend
   → /v1/makeup/simulate
   → /v1/custom/apply
   ```

   각 단계의 결과는 `data/output/`에 저장되며 독립 호출도 가능.

---

## 🧠 모델 및 라이브러리

| 구분      | 사용 모델                                   |
| ------- | --------------------------------------- |
| 피부 분석   | NIA ResNet50 (분류5 + 회귀5)                |
| 스타일 추천  | CLIP (`openai/clip-vit-base-patch32`)   |
| 메이크업 전이 | Stable Diffusion v1.5 + ControlNet      |
| 커스터마이즈  | SegFormer (`jonathandinu/face-parsing`) |
| 텍스트 생성  | Google Gemini (2.0/2.5 flash)           |

(`README.md` 생성 + 커밋 메시지 예시 포함해서)
```

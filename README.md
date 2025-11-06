# Beautiq AI-Backend (FastAPI) _ 1107ver

<br>

## 프로젝트 구조

```

project_root/
├── main.py # FastAPI 앱 생성, CORS, /v1 마운트, 헬스체크, startup 프리로드
├── precompute_embeddings.py # 사전 임베딩 계산 스크립트
├── test.py # 전체 파이프라인 테스트 스크립트
├── test_timing.py # 전체 파이프라인 테스트 스크립트 (소요 시간 계산 과정 포함) 
│
├── api/ # 엔드포인트 라우터 모듈
│ ├── init.py
│ ├── router.py # /v1 하위 라우터 통합
│ ├── nia.py # /v1/nia/analyze (피부 분석)
│ ├── feedback.py # /v1/feedback/generate (피드백 생성)
│ ├── product.py # /v1/product/reason (추천 이유 생성)
│ ├── style.py # /v1/style/recommend (스타일 추천)
│ ├── makeup.py # /v1/makeup/simulate (메이크업 전이)
│ ├── customization.py # /v1/custom/apply (커스터마이즈 적용)
│ └── health.py # /health, /ready, /version
│
├── schemas.py # Pydantic 스키마 (요청/응답 구조 정의, 팀 계약서)
├── config.py # 환경 변수 및 경로 설정 (GEMINI_API_KEY, checkpoints 등)
│
├── utils/ # 공통 유틸리티
│ ├── init.py
│ ├── base64_utils.py # 이미지 Base64 인코딩/디코딩
│ └── errors.py # 공통 예외 및 에러 응답 포맷
│
├── service/ # AI 서비스 로직 (각 단계별 run_inference)
├── model_manager/ # 모델 로딩, 캐싱, 싱글턴 관리
│
├── checkpoints/ # 모델 가중치 (.bin / .pth)
├── data/ # 테스트 입력 및 결과 출력
├── libs/ # 공용 모듈 (예: detail_encoder, pipeline 등)
├── models/image_encodel_l/ 
│
├── requirements_org/ # 서브 요구사항 모음
├── requirements.txt # 통합 패키지 요구사항
└── README.md # 실행 방법 및 엔드포인트 문서

````

<br>

---

> ⚠️ **GitHub 파일 크기 제한** 으로 인해
> `checkpoints/`, `data/`, `models/image_encodel_l`폴더는 저장소에 포함되어 있지 않습니다.
> 아래 링크에서 다운로드 후 루트에 배치해주세요.

* **checkpoints:** [🔗 Google Drive](https://drive.google.com/drive/folders/1NLY7QJuLbwZaZUeSBRGEyPdA_irSyelO?usp=sharing)
* **data:** [🔗 Google Drive](https://drive.google.com/drive/folders/1o12-FR_m8ddtWtmll3r0lQ3KAptRZEpz?usp=sharing)
* **models/image_encodel_l:** [🔗 Google Drive](https://drive.google.com/drive/folders/18dSPE_PBMR4KryzCoiv_AFlvcCBKl__6?usp=sharing)
* 또는 위의 세 개를 [**zip 파일**](https://drive.google.com/drive/folders/10lZ3Yn5P042-dZfYE3OUm-kGnYPXRLHS?usp=sharing)로도 다운 받을 수 있습니다.

## 실행 방법

> **터미널 1 : FastAPI**

```bash
cd ~/jimin/ai_1107ver
pip install --no-cache-dir -r requirements.txt
```

```bash
export GEMINI_API_KEY="your_api_key_here"
```

```bash
# 혹시 살아있는 서버 있을까봐 선제 종료
pkill -f "uvicorn main:app" || true

# FastAPI 서버 실행 (ngrok 모드: root-path 미사용)
uvicorn main:app --host 0.0.0.0 --port 8000 \
  --workers 1 --timeout-keep-alive 1200
```

> **터미널 2 : ngrok**

```bash
# 서버 헬스 확인 (정상일 때 "status": "ok")
curl -s http://127.0.0.1:8000/health
```

```bash
# 실행 중인 ngrok 종료
pkill -f '^ngrok\b' 2>/dev/null || true

# 오래된 바이너리/압축 삭제
rm -f ~/.local/bin/ngrok ~/ngrok ~/ngrok.tgz

# ✅ v3 정식 링크로 새로 받기 (캐시 무시)
curl -fSL -H 'Cache-Control: no-cache' \
  -o ngrok-v3.tgz \
  https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz

tar -xzf ngrok-v3.tgz      # 여기서 ./ngrok 생성

# 설치 전에 바로 버전 확인 (반드시 3.x.x)
./ngrok version
```

```bash
# 설치 및 경로 갱신
install -m 755 ./ngrok ~/.local/bin/ngrok
hash -r
~/.local/bin/ngrok version  # 다시 3.x.x 확인
which -a ngrok
```

```bash
# ngrok 실행
ngrok config add-authtoken <token>

# 터널 열기 (아시아 리전)
ngrok http --region=ap 8000
# → 여기서 나온 HTTPS URL(BASE_URL)을 백엔드/프론트 팀에 전달하세요.
```

> **터미널 3 : End-to-End Test**

```bash
cd ~/jimin/ai_1107ver
```

```bash
python precompute_embeddings.py
python test_timing.py
```

> 모든 체인 성공 시 출력 예:

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

<br>

## 확인 및 개선 필요사항

| 항목                      | 상태       | 비고                          |
| ----------------------- | -------- | --------------------------- |
| Style 추천                | ✅ 정상     | Top-3 결과 반환                 |
| Makeup 전이               | ✅ 정상  | 생성 결과 반환                |
| Customization           | ✅ 정상     | eyelid, lip, blush, skin 지원 |
| NIA/Feedback/Product 체인 | ✅ 정상     | 요청·응답 스키마 일관성 유지            |
| 임베딩 캐싱                  | ⚙️ 코드 완료 | 실제 속도 개선 테스트 필요             |
| 결과 퀄리티 평가               | 🚧 예정    | 팀 내부 논의 필요                  |

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

# app/service/feedback_service.py
"""
피부 피드백 생성 서비스
- FastAPI에서 전달한 request: dict 를 받아 run_inference(request) 수행
- 입력 검증/예외 처리 포함
"""

from __future__ import annotations
import json
from typing import Any, Dict, Tuple

from app.model_manager.feedback_manager import load_feedback_model, get_loaded_model_name


# ---------------------------
# 유틸
# ---------------------------
def _extract_predictions(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    request에서 predictions 딕셔너리를 추출.
    허용 입력:
      1) request["predictions"] = dict
      2) request["predictions_json_path"] = "path/to/predictions.json"
         - 파일 구조가 {"status":..., "predictions": {...}} 면 내부 predictions만 추출
      3) request["predictions_json"] = JSON 문자열
    """
    if "predictions" in request and isinstance(request["predictions"], dict):
        return request["predictions"]

    if "predictions_json" in request:
        try:
            raw = json.loads(request["predictions_json"])
        except Exception as e:
            raise ValueError(f"predictions_json 파싱 실패: {e}")
        if isinstance(raw, dict) and "predictions" in raw and isinstance(raw["predictions"], dict):
            return raw["predictions"]
        if isinstance(raw, dict):
            return raw
        raise ValueError("predictions_json 내에 올바른 JSON 객체가 없습니다.")

    if "predictions_json_path" in request:
        path = request["predictions_json_path"]
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            raise ValueError(f"predictions_json_path 파일 로드 실패: {e}")
        if isinstance(raw, dict) and "predictions" in raw and isinstance(raw["predictions"], dict):
            return raw["predictions"]
        if isinstance(raw, dict):
            return raw
        raise ValueError("predictions_json_path의 JSON 구조가 올바르지 않습니다.")

    raise ValueError(
        "입력 누락: 'predictions' (dict) 또는 'predictions_json' (str) "
        "또는 'predictions_json_path' (str)가 필요합니다."
    )


def _build_prompt(pred: Dict[str, Any]) -> str:
    """Gemini에게 보낼 한국어 프롬프트 구성."""
    data_json = json.dumps(pred, ensure_ascii=False, indent=2)

    prompt = f"""
당신은 전문적인 피부 분석 전문가입니다.
다음 JSON은 한 사용자의 피부 분석 결과입니다. 아래 “데이터 정의”와 “해석 기준”을 엄격히 따르세요.
결과는 3줄 이내 한국어 문장으로 출력하며, (1) 부위/항목별 핵심 상태, (2) 필요한 관리 팁, (3) 전반 요약을 포함하세요.

[데이터 정의]
{{
  "status": "success",
  "predictions": {{
    // Classification (숫자↑ = 나쁨)
    "dryness": int,              // 0~4
    "pigmentation": int,         // 0~5
    "pore": int,                 // 0~5
    "sagging": int,              // 0~5
    "wrinkle": int,              // 0~6

    // Regression
    "forehead_moisture": int,        // 0~100
    "cheek_moisture": int,           // 0~100
    "chin_moisture": int,            // 0~100
    "forehead_elasticity_R2": float, // 0.0~1.0
    "cheek_elasticity_R2": float,    // 0.0~1.0
    "chin_elasticity_R2": float,     // 0.0~1.0
    "perocular_wrinkle_Ra": int,     // 0~50
    "forehead_pigmentation": int,    // 0~350
    "cheek_pigmentation": int,       // 0~350
    "cheek_pore": int                // 0~2600
  }}
}}

[해석 기준]
- Classification: 값이 높을수록 상태가 나쁨.
- Regression 경계값:
  - moisture < 70 → 수분 부족
  - elasticity_R2 < 0.7 → 탄력 부족
  - wrinkle_Ra ≥ 30 → 주의
  - pigmentation ≥ 250 → 주의
  - cheek_pore ≥ 1800 → 주의

[실제 데이터]
{data_json}

출력 형식 규칙:
- 반드시 3줄 이내 한국어 문장.
- 과장 표현, 광고성 문구 금지. 전문적이고 간결하게.
- 개인 민감 정보 추정 금지.
"""
    return prompt


def _call_llm(prompt: str) -> Tuple[str, str]:
    """
    Gemini LLM 호출.
    Returns:
        (text, model_name)
    """
    model = load_feedback_model()
    model_name = get_loaded_model_name() or "unknown"
    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()
    if not text:
        raise RuntimeError("LLM 응답이 비어 있습니다.")
    return text, model_name


# ---------------------------
# 공개 함수 (서비스 진입점)
# ---------------------------
def run_inference(request: dict) -> dict:
    """
    request: FastAPI에서 받은 입력(JSON)을 dict로 변환한 것
    return : {"status": "success", "feedback": str, "model": str} 또는 실패 메시지
    """
    try:
        # 1) 입력 검증/파싱
        predictions = _extract_predictions(request)

        # 2) 프롬프트 생성
        prompt = _build_prompt(predictions)

        # 3) LLM 호출
        text, model_name = _call_llm(prompt)

        return {
            "status": "success",
            "feedback": text,
            "model": model_name,
        }

    except Exception as e:
        return {"status": "failed", "message": str(e)}

# app/service/feedback_service.py
"""
피부 피드백 생성 서비스 (0~100, 모두 높을수록 좋음)
- 규칙 (1) run_inference(request: dict) -> dict
- 규칙 (2) 모델 로딩 단일화
- 규칙 (3) 순수 함수화
- 규칙 (4) 입력 검증 & 예외 처리
"""
from __future__ import annotations
import json
from typing import Any, Dict, Tuple
from app.model_manager.feedback_manager import load_model, get_loaded_model_name


# ---------------------------
# 고정 프롬프트 (모두 높을수록 좋음)
# ---------------------------
FIXED_PROMPT = (
    "당신은 전문적인 피부 분석 전문가입니다.\n"
    "아래 JSON은 한 사용자의 피부 분석 결과이며, 모든 점수는 0~100 범위입니다.\n"
    "모든 항목은 높을수록 좋은 상태를 의미합니다.\n"
    "결과를 3문장 이내로 작성하세요.\n"
    "- 첫 문장: 전반적인 피부 상태 요약\n"
    "- 두 번째 문장: 긍정적인 부분 강조\n"
    "- 세 번째 문장: 보완하면 좋을 점 제안\n"
    "전문적이고 간결하게, 광고성 표현 없이 한국어로 출력하세요."
)


# ---------------------------
# 입력 파싱 유틸
# ---------------------------
def _load_predictions_from_file(path: str) -> Dict[str, Any]:
    """파일 경로에서 JSON 로드 → 내부 predictions 추출."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        raise ValueError(f"predictions_json_path 파일 로드 실패: {e}")

    if isinstance(raw, dict) and "predictions" in raw:
        preds = raw["predictions"]
        if not isinstance(preds, dict):
            raise ValueError("'predictions' 값이 객체형(dict)이 아닙니다.")
        return preds
    if isinstance(raw, dict):
        return raw
    raise ValueError("predictions_json_path의 JSON 구조가 올바르지 않습니다.")


def _parse_predictions_from_json_str(s: str) -> Dict[str, Any]:
    """JSON 문자열에서 predictions 추출."""
    try:
        raw = json.loads(s)
    except Exception as e:
        raise ValueError(f"predictions_json 파싱 실패: {e}")

    if isinstance(raw, dict) and "predictions" in raw:
        preds = raw["predictions"]
        if not isinstance(preds, dict):
            raise ValueError("'predictions' 값이 객체형(dict)이 아닙니다.")
        return preds
    if isinstance(raw, dict):
        return raw
    raise ValueError("predictions_json 내 구조가 올바르지 않습니다.")


def _extract_predictions(request: Dict[str, Any]) -> Dict[str, Any]:
    """입력에서 predictions 딕셔너리 추출."""
    if "predictions_json_path" in request:
        path = request["predictions_json_path"]
        if not isinstance(path, str) or not path.strip():
            raise ValueError("'predictions_json_path'는 비어있지 않은 문자열이어야 합니다.")
        return _load_predictions_from_file(path.strip())

    if "predictions_json" in request:
        s = request["predictions_json"]
        if not isinstance(s, str) or not s.strip():
            raise ValueError("'predictions_json'은 비어있지 않은 문자열이어야 합니다.")
        return _parse_predictions_from_json_str(s.strip())

    if "predictions" in request and isinstance(request["predictions"], dict):
        return request["predictions"]

    raise ValueError(
        "피부분석 JSON 입력이 없습니다. 최소 하나 필요: "
        "'predictions_json_path' 또는 'predictions_json' 또는 'predictions(dict)'."
    )


# ---------------------------
# 프롬프트 구성
# ---------------------------
def _build_prompt(pred: Dict[str, Any]) -> str:
    """고정 프롬프트 + 실제 데이터."""
    data_json = json.dumps(pred, ensure_ascii=False, indent=2)
    prompt = f"{FIXED_PROMPT}\n\n[실제 데이터]\n{data_json}\n"
    return prompt


# ---------------------------
# LLM 호출
# ---------------------------
def _call_llm(prompt: str) -> Tuple[str, str]:
    model = load_model()
    model_name = get_loaded_model_name() or "unknown"
    resp = model.generate_content(prompt)
    text = (getattr(resp, "text", "") or "").strip()
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

    입력 (택1):
      - predictions_json_path: str
      - predictions_json: str
      - predictions: dict
    """
    try:
        predictions = _extract_predictions(request)
        prompt = _build_prompt(predictions)
        text, model_name = _call_llm(prompt)
        return {"status": "success", "feedback": text, "model": model_name}
    except Exception as e:
        return {"status": "failed", "message": str(e)}

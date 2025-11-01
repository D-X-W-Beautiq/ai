# service/feedback_service.py
from __future__ import annotations
import json
import os
from typing import Any, Dict
from model_manager.feedback_manager import load_model, get_loaded_model_name

DEFAULT_PREDICTIONS_PATH = os.getenv("FEEDBACK_PREDICTIONS_PATH", "data/predictions.json")

# ✅ 회귀값만 사용
REQUIRED_SCORE_KEYS = [
    "pigmentation_reg",
    "moisture_reg",
    "elasticity_reg",
    "wrinkle_reg",
    "pore_reg",
]

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

def _load_predictions_from_file(path: str) -> Dict[str, Any]:
    """JSON 파일에서 predictions 데이터 로드"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"predictions JSON 파일을 찾을 수 없습니다: {path}")
    except Exception as e:
        raise ValueError(f"predictions JSON 파일 로드 실패: {e}")
    
    # predictions 키가 있으면 그 안의 데이터 사용
    if isinstance(raw, dict) and "predictions" in raw:
        preds = raw["predictions"]
        if not isinstance(preds, dict):
            raise ValueError("'predictions' 값이 dict가 아닙니다.")
        return preds
    
    # predictions 키가 없으면 전체를 predictions로 간주
    if isinstance(raw, dict):
        return raw
    
    raise ValueError("predictions JSON 구조가 올바르지 않습니다.")

def _extract_predictions(request: Dict[str, Any]) -> Dict[str, Any]:
    """요청에서 predictions_json_path 추출 후 파일 로드"""
    if "predictions_json_path" in request:
        path = request["predictions_json_path"]
        if not isinstance(path, str) or not path.strip():
            raise ValueError("'predictions_json_path'는 비어있지 않은 문자열이어야 합니다.")
        return _load_predictions_from_file(path.strip())
    
    # 경로가 없으면 기본 경로 사용
    return _load_predictions_from_file(DEFAULT_PREDICTIONS_PATH)

def _to_int_score(val: Any, key: str) -> int:
    """값을 0~100 정수로 변환 및 검증"""
    if isinstance(val, str):
        val = val.strip()
        if val == "":
            raise ValueError(f"'{key}' 점수가 비어 있습니다.")
        try:
            as_int = int(round(float(val)))
        except Exception:
            raise ValueError(f"'{key}' 점수는 숫자여야 합니다. 현재 값: {val}")
        val = as_int
    
    if isinstance(val, bool):
        raise ValueError(f"'{key}' 점수는 불리언이 될 수 없습니다.")
    
    if isinstance(val, float):
        val = int(round(val))
    
    if not isinstance(val, int):
        raise ValueError(f"'{key}' 점수는 int여야 합니다. 현재 타입: {type(val).__name__}")
    
    if not (0 <= val <= 100):
        raise ValueError(f"'{key}' 점수는 0~100이어야 합니다. 현재 값: {val}")
    
    return val

def _validate_and_normalize(preds: Dict[str, Any]) -> Dict[str, int]:
    """필수 회귀 키 검증 및 정규화"""
    missing = [k for k in REQUIRED_SCORE_KEYS if k not in preds]
    if missing:
        raise ValueError(f"필수 회귀 점수 키 누락: {missing}")
    
    return {k: _to_int_score(preds[k], k) for k in REQUIRED_SCORE_KEYS}

def _build_prompt(pred: Dict[str, Any]) -> str:
    """프롬프트 생성"""
    data_json = json.dumps(pred, ensure_ascii=False, indent=2)
    return f"{FIXED_PROMPT}\n\n[실제 데이터]\n{data_json}\n"

def _call_llm(prompt: str) -> str:
    """LLM 호출"""
    model = load_model()
    _ = get_loaded_model_name()
    resp = model.generate_content(prompt)
    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        raise RuntimeError("LLM 응답이 비어 있습니다.")
    return text

def run_inference(request: dict) -> dict:
    """
    피드백 생성 메인 로직
    request: {"predictions_json_path": "data/predictions.json"}
    return: {"status": "success", "feedback": "..."} 또는 {"status": "failed", "message": "..."}
    """
    try:
        raw_preds = _extract_predictions(request or {})
        norm_preds = _validate_and_normalize(raw_preds)
        prompt = _build_prompt(norm_preds)
        feedback_text = _call_llm(prompt)
        
        return {"status": "success", "feedback": feedback_text}
    
    except Exception as e:
        return {"status": "failed", "message": str(e)}
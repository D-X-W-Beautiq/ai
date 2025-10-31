# service/feedback_service.py
from __future__ import annotations
import json
import os
from typing import Any, Dict, Optional

from model_manager.feedback_manager import load_model, get_loaded_model_name

DEFAULT_PREDICTIONS_PATH = os.getenv("FEEDBACK_PREDICTIONS_PATH", "data/predictions.json")

REQUIRED_SCORE_KEYS = [
    "dryness","pigmentation","pore","sagging","wrinkle",
    "pigmentation_reg","moisture_reg","elasticity_reg","wrinkle_reg","pore_reg",
]

FIXED_PROMPT = (
    "당신은 전문적인 피부 분석 전문가입니다.\n"
    "아래 점수(0~100)를 바탕으로 사용자의 피부 상태를 이해하기 쉬운 한국어로, "
    "친절하지만 간결하게 4~6문장으로 설명해 주세요.\n"
    "- 강점과 개선이 필요한 항목을 구분\n- 생활 습관/기초 케어 팁 2가지 제안\n"
    "- 제품 카테고리를 1~2개 권장(보습/탄력/주름/색소/모공 등)\n"
)

def _call_llm(prompt: str) -> str:
    model = load_model()
    # 실제 모델 호출부로 교체하세요.
    return f"[모델:{get_loaded_model_name()}]\n{prompt}\n\n요약: 보습과 탄력 케어를 함께 권장합니다."

# ---------- 경로 보정 유틸: 상대경로를 '프로젝트 루트' 기준으로 자동 보정 ----------
def _resolve_path(path: str) -> str:
    """
    상대경로를 '프로젝트 루트' → '현재 작업 디렉터리(CWD)' 순서로만 보정.
    둘 다 없으면 FileNotFoundError.
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path

    # 1) 프로젝트 루트 (…/project_root/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cand1 = os.path.normpath(os.path.join(project_root, path))
    if os.path.exists(cand1):
        return cand1

    # 2) 현재 작업 디렉터리(CWD)
    cand2 = os.path.normpath(os.path.join(os.getcwd(), path))
    if os.path.exists(cand2):
        return cand2

    raise FileNotFoundError(
        "Predictions file not found (relative path). Tried:\n"
        f"  - {cand1}\n"
        f"  - {cand2}"
    )

# ---------------------------------------------------------------------------

def _parse_predictions_file(path: str) -> Dict[str, Any]:
    resolved = _resolve_path(path)
    if not os.path.exists(resolved):
        raise ValueError(f"predictions.json 파일을 찾을 수 없습니다: {resolved}")
    with open(resolved, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "predictions" in raw and isinstance(raw["predictions"], dict):
        return raw["predictions"]
    if isinstance(raw, dict):
        return raw
    raise ValueError("predictions JSON 구조가 올바르지 않습니다.")

def _parse_predictions_from_json_str(s: str) -> Dict[str, Any]:
    raw = json.loads(s)
    if isinstance(raw, dict) and "predictions" in raw and isinstance(raw["predictions"], dict):
        return raw["predictions"]
    if isinstance(raw, dict):
        return raw
    raise ValueError("predictions_json 구조가 올바르지 않습니다.")

def _extract_predictions(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    우선순위:
      1) predictions (dict)
      2) predictions_json (비어있지 않은 문자열일 때만 처리; 빈 문자열은 무시)
      3) predictions_json_path (상대경로 허용; 프로젝트 루트 기준 보정)
      4) (없으면) DEFAULT_PREDICTIONS_PATH가 존재하면 사용
    """
    # 1) 점수 dict 직접 전달
    preds = request.get("predictions")
    if isinstance(preds, dict):
        return preds

    # 2) JSON 문자열 (빈 문자열은 무시)
    if "predictions_json" in request:
        s = request.get("predictions_json")
        if isinstance(s, str) and s.strip():
            return _parse_predictions_from_json_str(s.strip())
        # 빈 문자열이면 무시하고 다음 경로로 진행

    # 3) 파일 경로
    path = request.get("predictions_json_path")
    if isinstance(path, str) and path.strip():
        return _parse_predictions_file(path.strip())

    # 4) 기본 경로(옵션)
    if DEFAULT_PREDICTIONS_PATH and os.path.exists(_resolve_path(DEFAULT_PREDICTIONS_PATH)):
        return _parse_predictions_file(DEFAULT_PREDICTIONS_PATH)

    # 없으면 예외
    raise ValueError(
        "피드백 생성에는 predictions가 필요합니다. "
        "(predictions | predictions_json | predictions_json_path 중 하나)"
    )

def _to_int_score(val: Any, key: str) -> int:
    if isinstance(val, str):
        val = int(round(float(val.strip())))
    if isinstance(val, float):
        val = int(round(val))
    if not isinstance(val, int):
        raise ValueError(f"'{key}' 점수는 int여야 합니다.")
    if not (0 <= val <= 100):
        raise ValueError(f"'{key}' 점수는 0~100이어야 합니다.")
    return val

def _validate_and_normalize(preds: Dict[str, Any]) -> Dict[str, int]:
    missing = [k for k in REQUIRED_SCORE_KEYS if k not in preds]
    if missing:
        raise ValueError(f"필수 점수 키 누락: {missing}")
    return {k: _to_int_score(preds[k], k) for k in REQUIRED_SCORE_KEYS}

def _build_prompt(pred: Dict[str, Any]) -> str:
    data_json = json.dumps(pred, ensure_ascii=False, indent=2)
    return f"{FIXED_PROMPT}\n\n[실제 데이터]\n{data_json}\n"

def run_inference(request: dict) -> dict:
    """
    입력:
      - (A) prompt: str  → 이 경우 그대로 LLM에 전달
      - (B) predictions | predictions_json | predictions_json_path → 정규화 후 프롬프트 생성
    출력:
      {"status":"success","feedback":"..."} 또는 {"status":"error","message":"..."}
    """
    try:
        # (A) prompt 단독 지원
        if request.get("prompt") and not any(k in request for k in ("predictions","predictions_json","predictions_json_path")):
            feedback_text = _call_llm(str(request["prompt"]))
            return {"status": "success", "feedback": feedback_text}

        # (B) 파일/문자열/딕셔너리에서 점수 추출 → 검증 → 프롬프트 생성
        raw_preds = _extract_predictions(request or {})
        norm_preds = _validate_and_normalize(raw_preds)
        prompt = _build_prompt(norm_preds)
        feedback_text = _call_llm(prompt)
        return {"status": "success", "feedback": feedback_text}

    except Exception as e:
        # 에러 포맷 통일
        return {"status": "error", "message": str(e)}

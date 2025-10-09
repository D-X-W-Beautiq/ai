# model_manager/feedback_manager.py
from __future__ import annotations
import os
import threading
from typing import Optional

try:
    import google.generativeai as genai
except ImportError as e:
    raise ImportError(
        "google-generativeai 가 설치되어 있지 않습니다. "
        "requirements_org/requirements_feedback.txt 를 참고하여 설치하세요."
    ) from e

_model_lock = threading.Lock()
_model_obj: Optional["genai.GenerativeModel"] = None
_model_name: Optional[str] = None

def _configure() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다.")
    genai.configure(api_key=api_key)

def load_model(model_name: Optional[str] = None) -> "genai.GenerativeModel":
    """
    모델은 최초 1회만 생성/캐시.
    환경변수:
      - FEEDBACK_MODEL_NAME (기본: gemini-2.5-flash)
      - 선택: FEEDBACK_TEMPERATURE, FEEDBACK_TOP_P 등 파라미터가 필요하면 여기서 확장 가능
    """
    global _model_obj, _model_name
    with _model_lock:
        if _model_obj is not None:
            return _model_obj

        _configure()
        _model_name = model_name or os.getenv("FEEDBACK_MODEL_NAME", "gemini-2.5-flash")

        # 필요 시 generation_config 확장 가능
        gen_cfg = {
            # "temperature": float(os.getenv("FEEDBACK_TEMPERATURE", "0.6")),
            # "top_p": float(os.getenv("FEEDBACK_TOP_P", "0.9")),
        }
        # 빈 dict이면 None으로
        gen_cfg = gen_cfg if any(gen_cfg.values()) else None

        _model_obj = genai.GenerativeModel(model_name=_model_name, generation_config=gen_cfg)
        return _model_obj

def get_loaded_model_name() -> Optional[str]:
    return _model_name

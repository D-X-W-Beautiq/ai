# app/model_manager/feedback_manager.py
from __future__ import annotations
import os, threading
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
    global _model_obj, _model_name
    with _model_lock:
        if _model_obj is not None:
            return _model_obj
        _configure()
        _model_name = model_name or os.getenv("FEEDBACK_MODEL_NAME", "gemini-2.5-flash")
        _model_obj = genai.GenerativeModel(_model_name)
        return _model_obj


def get_loaded_model_name() -> Optional[str]:
    return _model_name

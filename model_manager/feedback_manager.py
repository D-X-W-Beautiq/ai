# app/model_manager/feedback_manager.py
"""
Gemini 기반 피부 피드백 LLM 매니저
- 서버 부팅 시 1회 로딩 후 캐시 재사용
- 서비스 레이어에서는 load_feedback_model()만 호출
"""

from __future__ import annotations
import os
import threading
from typing import Optional, Callable

try:
    import google.generativeai as genai
except ImportError as e:
    raise ImportError(
        "google-generativeai 가 설치되어 있지 않습니다. "
        "requirements_feedback.txt를 참고해 설치하세요."
    ) from e


# ---- 내부 캐시 및 락 ----
_model_lock = threading.Lock()
_model_obj: Optional["genai.GenerativeModel"] = None
_model_name: Optional[str] = None


def _configure_gemini() -> None:
    """환경변수에서 API 키를 읽어 Gemini SDK 설정."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다. "
            "배포/개발 환경 변수에 GEMINI_API_KEY를 추가하세요."
        )
    genai.configure(api_key=api_key)


def load_feedback_model(model_name: Optional[str] = None) -> "genai.GenerativeModel":
    """
    Gemini 모델을 로드(1회)하고 캐시 객체를 반환.
    Args:
        model_name: 사용할 모델명 (기본: 'gemini-2.5-flash')
    Returns:
        genai.GenerativeModel 인스턴스
    """
    global _model_obj, _model_name

    with _model_lock:
        if _model_obj is not None:
            return _model_obj

        _configure_gemini()
        _model_name = model_name or os.getenv("FEEDBACK_MODEL_NAME", "gemini-2.5-flash")
        _model_obj = genai.GenerativeModel(_model_name)
        return _model_obj


def get_loaded_model_name() -> Optional[str]:
    """로드된 모델명을 반환(디버그/로그용)."""
    return _model_name

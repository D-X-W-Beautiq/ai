# -*- coding: utf-8 -*-
import os
from pathlib import Path
from functools import lru_cache

class Settings:
    # ====== 환경 변수 ======
    APP_ENV: str = os.getenv("APP_ENV", "dev")
    SERVICE_VERSION: str = os.getenv("SERVICE_VERSION", "0.1.0")
    COMMIT_SHA: str = os.getenv("COMMIT_SHA", "unknown")

    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

    # ====== 경로 ======
    PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path.cwd()))
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = DATA_DIR / "output"
    CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
    MAKEUP_CKPT_DIR = CHECKPOINTS_DIR / "makeup"

    # ====== 기본 설정 ======
    DEFAULT_RESOLUTION: int = 512
    DEFAULT_STEPS: int = 30
    DEFAULT_GUIDANCE: float = 2.0

    # 유효성 점검(필수 아님)
    def readiness_checks(self) -> dict:
        def exists(p: Path) -> bool:
            try: return p.exists()
            except Exception: return False

        return {
            "env:GEMINI_API_KEY": bool(self.GEMINI_API_KEY),
            "paths:checkpoints/makeup": exists(self.MAKEUP_CKPT_DIR),
            "paths:data/output": exists(self.OUTPUT_DIR),
        }

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    # 필요 시 디렉토리 생성
    s.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return s

from fastapi import APIRouter
from config import get_settings
import time
router = APIRouter(tags=["Health"])
_started = time.time()

@router.get("/health")
def health():
    return {"status": "ok", "uptime_sec": round(time.time() - _started, 1)}

@router.get("/ready")
def ready():
    s = get_settings()
    checks = s.readiness_checks()
    return {"ready": all(checks.values()), "checks": checks}

@router.get("/version")
def version():
    s = get_settings()
    return {"service": "Beautiq AI-BE", "version": s.SERVICE_VERSION, "commit": s.COMMIT_SHA, "env": s.APP_ENV}

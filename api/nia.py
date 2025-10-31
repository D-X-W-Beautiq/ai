# api/nia.py
"""
NIA 피부 분석 API
POST /v1/nia/analyze - 얼굴 이미지 기반 피부 상태 분석
"""
from fastapi import APIRouter
from schemas import NIARequest, NIAResponse
import sys
from pathlib import Path

# service 모듈 임포트를 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

router = APIRouter(prefix="/nia", tags=["NIA - 피부 분석"])

@router.post("/analyze", response_model=NIAResponse, summary="피부 분석(10 지표)")
async def analyze_skin(request: NIARequest) -> NIAResponse:
    try:
        from service.nia_service import run_inference
        payload = request.model_dump()
        result = run_inference(payload)

        if result.get("status") == "success":
            return NIAResponse(status="success", predictions=result.get("predictions"))
        # 실패 바디를 스키마로 직접 반환
        return NIAResponse(status="error", message=result.get("message", "NIA 분석 실패"))

    except ValueError as e:
        return NIAResponse(status="error", message=str(e))
    except Exception as e:
        return NIAResponse(status="error", message=f"NIA 내부 오류: {str(e)}")

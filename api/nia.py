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

@router.post("/analyze", response_model=NIAResponse, response_model_exclude_none=True, summary="피부 분석(5 지표)")
async def analyze_skin(request: NIARequest) -> NIAResponse:
    try:
        from service.nia_service import run_inference
        from service.feedback_service import run_inference as feedback_inference

        # 1. 피부 분석 실행
        payload = request.model_dump()
        result = run_inference(payload)

        if result.get("status") != "success":
            return NIAResponse(
                status="error", message=result.get("message", "NIA 분석 실패")
            )

        predictions = result.get("predictions")

        # 2. 피드백 생성을 위한 요청 준비
        feedback_payload = {
            "predictions": {
                "moisture_reg": predictions.get("moisture_reg"),
                "elasticity_reg": predictions.get("elasticity_reg"),
                "wrinkle_reg": predictions.get("wrinkle_reg"),
                "pigmentation_reg": predictions.get("pigmentation_reg"),
                "pore_reg": predictions.get("pore_reg"),
            }
        }

        # 3. 피드백 생성 실행
        feedback_result = feedback_inference(feedback_payload)
        feedback_text = None

        if feedback_result.get("status") == "success":
            feedback_text = feedback_result.get("feedback")

        # 4. 통합 응답 반환
        return NIAResponse(
            status="success", predictions=predictions, feedback=feedback_text
        )

    except ValueError as e:
        return NIAResponse(status="error", message=str(e))
    except Exception as e:
        return NIAResponse(status="error", message=f"NIA 내부 오류: {str(e)}")

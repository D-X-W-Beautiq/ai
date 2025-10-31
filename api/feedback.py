# api/feedback.py
from fastapi import APIRouter
from schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["Feedback"])

@router.post("/generate", response_model=FeedbackResponse, summary="피드백/설명 생성")
async def generate_feedback(request: FeedbackRequest) -> FeedbackResponse:
    try:
        from service.feedback_service import run_inference
        result = run_inference(request.model_dump())

        if result.get("status") == "success":
            return FeedbackResponse(status="success", feedback=result.get("feedback"))

        msg = result.get("message", "피드백 생성 실패")
        return FeedbackResponse(status="error", message=msg)

    except Exception as e:
        return FeedbackResponse(status="error", message=f"피드백 처리 중 오류: {str(e)}")

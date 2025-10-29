# api/feedback.py
from fastapi import APIRouter, HTTPException
from schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["Feedback"])

@router.post("/generate", response_model=FeedbackResponse, summary="피드백/설명 생성")
async def generate_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """
    JSON 파일 경로를 받아 피드백 생성
    - predictions_json_path를 service로 전달
    - service가 실패 시 적절한 HTTP 상태 코드로 변환
    """
    try:
        # Lazy import
        from service.feedback_service import run_inference

        # 요청 데이터를 dict로 변환하여 service에 전달
        payload = request.model_dump()

        # 서비스 호출
        result = run_inference(payload)

        # 성공 처리
        if result.get("status") == "success":
            return FeedbackResponse(
                status="success",
                feedback=result.get("feedback")
            )

        # 실패 처리
        msg = result.get("message", "피드백 생성 실패")
        
        # 입력/파일 관련 오류는 400
        if any(keyword in msg for keyword in ["파일", "누락", "점수", "JSON", "경로"]):
            raise HTTPException(
                status_code=400,
                detail={"status": "failed", "message": msg}
            )
        
        # LLM 관련 오류는 500
        raise HTTPException(
            status_code=500,
            detail={"status": "failed", "message": msg}
        )

    except HTTPException:
        raise
    except Exception as e:
        # 예상치 못한 오류
        raise HTTPException(
            status_code=500,
            detail={"status": "failed", "message": f"피드백 처리 중 오류: {str(e)}"}
        )
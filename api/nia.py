"""
NIA 피부 분석 API
POST /v1/nia/analyze - 얼굴 이미지 기반 피부 상태 분석
"""
from fastapi import APIRouter, HTTPException
from schemas import NIARequest, NIAResponse
import sys
from pathlib import Path

# service 모듈 임포트를 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

router = APIRouter(
    prefix="/nia",
    tags=["NIA - 피부 분석"]
)


@router.post(
    "/analyze",
    response_model=NIAResponse,
    summary="피부 분석 실행",
    description="""
    얼굴 이미지를 분석하여 피부 상태 점수를 반환합니다.
    
    **처리 과정:**
    1. Base64 이미지 디코딩
    2. MediaPipe로 얼굴 크롭
    3. 10개 모델 추론 (Classification 5개 + Regression 5개)
    4. 0-100 점수로 정규화 (높을수록 좋은 상태)
    
    **반환 점수:**
    - dryness: 건조
    - pigmentation: 색소침착
    - pore: 모공
    - sagging: 처짐
    - wrinkle: 주름
    - *_reg: 각 지표의 회귀 분석 점수
    
    **참고:** 결과는 data/predictions.json에도 저장됩니다.
    """
)
async def analyze_skin(request: NIARequest) -> NIAResponse:
    """
    피부 분석 API
    
    Args:
        request: NIARequest (image_base64)
    
    Returns:
        NIAResponse (status, predictions or message)
    
    Raises:
        HTTPException: 500 - 내부 처리 오류
    """
    try:
        # service 모듈을 여기서 import (lazy loading)
        from service.nia_service import run_inference
        
        # Pydantic 모델을 dict로 변환하여 service에 전달
        payload = request.model_dump()
        
        # service 계층 호출
        result = run_inference(payload)
        
        # 결과 검증 및 응답
        if result.get("status") == "success":
            return NIAResponse(
                status="success",
                predictions=result.get("predictions")
            )
        else:
            # service에서 에러 반환 시
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "NIA 분석 중 알 수 없는 오류 발생")
            )
    
    except HTTPException:
        # 이미 HTTPException인 경우 그대로 전파
        raise
    
    except ValueError as e:
        # 입력 검증 오류 (service 계층에서 발생)
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
    except Exception as e:
        # 예상치 못한 오류
        raise HTTPException(
            status_code=500,
            detail=f"NIA 분석 처리 중 오류: {str(e)}"
        )


@router.get(
    "/health",
    summary="NIA 모듈 상태 확인",
    description="NIA 모델 로딩 상태 및 체크포인트 존재 여부 확인"
)
async def health_check():
    """NIA 모듈 헬스체크"""
    try:
        from model_manager.nia_manager import get_checkpoint_path
        import os
        
        checkpoint_path = get_checkpoint_path()
        class_path = os.path.join(checkpoint_path, "class")
        reg_path = os.path.join(checkpoint_path, "regression")
        
        return {
            "status": "healthy",
            "module": "NIA",
            "checkpoints": {
                "classification": os.path.exists(class_path),
                "regression": os.path.exists(reg_path)
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "module": "NIA",
            "error": str(e)
        }
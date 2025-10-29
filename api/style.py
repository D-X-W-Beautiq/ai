# api/style.py
from fastapi import APIRouter, HTTPException
from schemas import StyleRequest, StyleResponse
import os

router = APIRouter(prefix="/style", tags=["Style Recommendation"])

@router.post("/recommend", response_model=StyleResponse, summary="스타일 추천")
def recommend_style(request: StyleRequest):
    try:
        # ⬇️ 여기서 지연 임포트 (서버 기동 시 torch 로딩 안 함)
        from service.style_service import run_inference

        json_dir = os.path.join("data", "celeb")
        if not os.path.exists(json_dir):
            raise HTTPException(status_code=404, detail=f"데이터 경로를 찾을 수 없습니다: {json_dir}")

        result = run_inference(request.dict(), json_dir=json_dir)
        if result.get("status") != "success":
            raise HTTPException(status_code=500, detail=result.get("message", "Unknown error"))
        return StyleResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


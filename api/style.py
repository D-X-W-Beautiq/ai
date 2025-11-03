# api/style.py
from fastapi import APIRouter
from schemas import StyleRequest, StyleResponse, StyleResult
import os

router = APIRouter(prefix="/style", tags=["Style Recommendation"])

@router.post("/recommend", response_model=StyleResponse, response_model_exclude_none=True)
def recommend_style(request: StyleRequest):
    try:
        from service.style_service import run_inference
        json_dir = os.path.join("data", "style-recommendation")
        if not os.path.exists(json_dir):
            return StyleResponse(status="error", message="데이터 경로를 찾을 수 없습니다: data/style-recommendation")

        svc = run_inference(request.model_dump(), json_dir=json_dir)

        if svc.get("status") != "success":
            return StyleResponse(status="error", message=svc.get("message", "스타일 추천 실패"))

        results_out = [StyleResult(style_id=r.get("style_id",""), style_image_base64=r.get("style_image_base64","")) 
                       for r in svc.get("results", [])[:3]]
        return StyleResponse(status="success", results=results_out)

    except Exception as e:
        return StyleResponse(status="error", message=f"스타일 추천 처리 중 오류: {str(e)}")
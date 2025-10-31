# api/product.py
"""
Product 제품 추천 이유 생성 API
POST /v1/product/reason - LLM 기반 개인화 제품 추천 이유 생성
"""
from fastapi import APIRouter
from schemas import ProductRequest, ProductResponse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

router = APIRouter(prefix="/product", tags=["Product - 제품 추천"])

@router.post("/reason", response_model=ProductResponse, summary="제품 추천 이유 생성")
async def generate_recommendation_reason(request: ProductRequest) -> ProductResponse:
    try:
        from service.product_service import run_inference
        result = run_inference(request.model_dump())

        if result.get("status") == "success":
            return ProductResponse(status="success", recommendations=result.get("recommendations", []))

        return ProductResponse(
            status="error",
            message=result.get("message", "제품 추천 이유 생성 실패"),
            error_code=result.get("error_code", "UNKNOWN_ERROR"),
        )

    except ValueError as e:
        return ProductResponse(status="error", message=str(e), error_code="INVALID_REQUEST")
    except Exception as e:
        return ProductResponse(status="error", message=f"내부 오류: {str(e)}", error_code="INTERNAL_ERROR")

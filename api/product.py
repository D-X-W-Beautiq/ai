"""
Product 제품 추천 이유 생성 API
POST /v1/product/reason - LLM 기반 개인화 제품 추천 이유 생성
"""
from fastapi import APIRouter, HTTPException
from schemas import ProductRequest, ProductResponse
import sys
from pathlib import Path

# service 모듈 임포트를 위한 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

router = APIRouter(
    prefix="/product",
    tags=["Product - 제품 추천"]
)


@router.post(
    "/reason",
    response_model=ProductResponse,
    summary="제품 추천 이유 생성",
    description="""
    피부 분석 결과와 제품 정보를 기반으로 LLM이 개인화된 추천 이유를 생성합니다.
    
    **처리 과정:**
    1. 피부 분석 결과에서 부족한 영역 파악
    2. 각 제품의 카테고리, 성분, 리뷰 정보 분석
    3. Gemini LLM으로 개인화된 추천 이유 생성 (2-3문장)
    
    **필수 환경변수:**
    - GEMINI_API_KEY: Gemini API 키
    
    **카테고리:**
    - moisture: 수분
    - elasticity: 탄력
    - wrinkle: 주름
    - pigmentation: 색소침착
    - pore: 모공
    
    **참고:**
    - 개별 제품 처리 실패 시에도 에러 메시지를 reason에 포함하여 반환
    - API 호출 실패 시 최대 3회 재시도
    - 타임아웃: 30초
    """
)
async def generate_recommendation_reason(request: ProductRequest) -> ProductResponse:
    """
    제품 추천 이유 생성 API
    
    Args:
        request: ProductRequest (skin_analysis, recommended_categories, filtered_products, locale)
    
    Returns:
        ProductResponse (status, recommendations or message/error_code)
    
    Raises:
        HTTPException: 400 - 잘못된 요청, 500 - 내부 처리 오류
    """
    try:
        # service 모듈을 여기서 import (lazy loading)
        from service.product_service import run_inference
        
        # Pydantic 모델을 dict로 변환하여 service에 전달
        payload = request.model_dump()
        
        # service 계층 호출
        result = run_inference(payload)
        
        # 결과 검증 및 응답
        if result.get("status") == "success":
            return ProductResponse(
                status="success",
                recommendations=result.get("recommendations")
            )
        else:
            # service에서 에러 반환 시
            raise HTTPException(
                status_code=500,
                detail={
                    "message": result.get("message", "제품 추천 이유 생성 중 알 수 없는 오류 발생"),
                    "error_code": result.get("error_code", "UNKNOWN_ERROR")
                }
            )
    
    except HTTPException:
        # 이미 HTTPException인 경우 그대로 전파
        raise
    
    except ValueError as e:
        # 입력 검증 오류 (service 계층에서 발생)
        raise HTTPException(
            status_code=400,
            detail={
                "message": str(e),
                "error_code": "INVALID_REQUEST"
            }
        )
    
    except Exception as e:
        # 예상치 못한 오류
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"제품 추천 처리 중 오류: {str(e)}",
                "error_code": "INTERNAL_ERROR"
            }
        )


@router.get(
    "/health",
    summary="Product 모듈 상태 확인",
    description="Gemini API 키 설정 여부 확인"
)
async def health_check():
    """Product 모듈 헬스체크"""
    try:
        import os
        
        api_key_set = bool(os.getenv("GEMINI_API_KEY"))
        
        return {
            "status": "healthy" if api_key_set else "warning",
            "module": "Product",
            "gemini_api_key": "configured" if api_key_set else "not_configured",
            "message": "OK" if api_key_set else "GEMINI_API_KEY 환경변수를 설정해주세요"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "module": "Product",
            "error": str(e)
        }


@router.get(
    "/categories",
    summary="지원 카테고리 목록",
    description="제품 추천에 사용 가능한 카테고리 목록 반환"
)
async def get_categories():
    """지원하는 카테고리 목록"""
    return {
        "categories": [
            {
                "key": "moisture",
                "name_ko": "수분",
                "name_en": "Moisture",
                "threshold": 65,
                "description": "수분 부족 시 추천"
            },
            {
                "key": "elasticity",
                "name_ko": "탄력",
                "name_en": "Elasticity",
                "threshold": 60,
                "description": "탄력 부족 시 추천"
            },
            {
                "key": "wrinkle",
                "name_ko": "주름",
                "name_en": "Wrinkle",
                "threshold": 50,
                "description": "주름 개선 필요 시 추천"
            },
            {
                "key": "pigmentation",
                "name_ko": "색소침착",
                "name_en": "Pigmentation",
                "threshold": 70,
                "description": "색소침착 개선 필요 시 추천"
            },
            {
                "key": "pore",
                "name_ko": "모공",
                "name_en": "Pore",
                "threshold": 55,
                "description": "모공 관리 필요 시 추천"
            }
        ]
    }
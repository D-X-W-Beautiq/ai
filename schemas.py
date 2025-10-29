# -*- coding: utf-8 -*-
"""
API Request/Response 스키마 정의 (통합: NIA → Feedback → Product → Style → Makeup → Customization)
모든 엔드포인트의 데이터 계약(Contract)을 Pydantic 모델로 정의
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# NIA (피부 분석)
# ============================================================================
class NIARequest(BaseModel):
    """NIA 피부 분석 요청"""
    image_base64: str = Field(..., description="Base64로 인코딩된 얼굴 이미지")

class NIAPredictions(BaseModel):
    """
    NIA 피부 분석 결과 (0-100 점수, 높을수록 '양호')
    회귀 결과만 포함
    """
    pigmentation_reg: int = Field(..., ge=0, le=100)
    moisture_reg: int = Field(..., ge=0, le=100)
    elasticity_reg: int = Field(..., ge=0, le=100)
    wrinkle_reg: int = Field(..., ge=0, le=100)
    pore_reg: int = Field(..., ge=0, le=100)

class NIAResponse(BaseModel):
    """NIA 피부 분석 응답"""
    status: str = Field(..., description="success | error")
    predictions: Optional[NIAPredictions] = Field(None, description="분석 결과 (성공 시)")
    message: Optional[str] = Field(None, description="에러 메시지 (실패 시)")


# ============================================================================
# Feedback (설명/피드백 생성)
# ============================================================================
class FeedbackRequest(BaseModel):
    """
    피부 분석 결과 JSON 파일 경로를 받아 피드백 생성
    - predictions_json_path: 예측 결과 JSON 파일 경로
    - 제공되지 않으면 서비스 기본값('data/predictions.json') 사용
    """
    predictions_json_path: Optional[str] = Field(
        None,
        description="예측 결과 JSON 파일 경로 (예: data/predictions.json)"
    )

class FeedbackResponse(BaseModel):
    """피드백 생성 응답"""
    status: str = Field(..., description="success | failed")
    feedback: Optional[str] = Field(None, description="피드백 텍스트 (성공 시)")
    message: Optional[str] = Field(None, description="에러 메시지 (실패 시)")


# ============================================================================
# Product (제품 추천 이유)
# ============================================================================
class SkinAnalysis(BaseModel):
    """피부 분석 데이터 (NIA 결과)"""
    dryness: int = Field(..., ge=0, le=100)
    pigmentation: int = Field(..., ge=0, le=100)
    pore: int = Field(..., ge=0, le=100)
    sagging: int = Field(..., ge=0, le=100)
    wrinkle: int = Field(..., ge=0, le=100)
    pigmentation_reg: int = Field(..., ge=0, le=100)
    moisture_reg: int = Field(..., ge=0, le=100)
    elasticity_reg: int = Field(..., ge=0, le=100)
    wrinkle_reg: int = Field(..., ge=0, le=100)
    pore_reg: int = Field(..., ge=0, le=100)

class ProductInfo(BaseModel):
    """제품 정보"""
    product_id: str
    product_name: str
    brand: str
    category: str
    price: int = Field(..., ge=0)
    review_score: float = Field(..., ge=0, le=5)
    review_count: int = Field(..., ge=0)
    ingredients: List[str] = Field(default=[])

class ProductRequest(BaseModel):
    """제품 추천 이유 생성 요청"""
    skin_analysis: SkinAnalysis
    recommended_categories: List[str] = Field(..., min_length=1)
    filtered_products: List[ProductInfo] = Field(..., min_length=1)
    locale: Optional[str] = Field("ko-KR", description="언어 설정 (ko-KR | en-US)")

class ProductRecommendation(BaseModel):
    """개별 제품 추천 결과"""
    product_id: str
    reason: str

class ProductResponse(BaseModel):
    """제품 추천 이유 생성 응답"""
    status: str
    recommendations: Optional[List[ProductRecommendation]] = None
    message: Optional[str] = None
    error_code: Optional[str] = None


# ============================================================================
# Style (스타일 추천)
# ============================================================================
class StyleRequest(BaseModel):
    source_image_base64: str
    keywords: Optional[List[str]] = []

class StyleResponse(BaseModel):
    items: List[dict]  # 추천 스타일 리스트


# ============================================================================
# Makeup (메이크업 시뮬레이션)
# ============================================================================
class MakeupRequest(BaseModel):
    # 필수
    source_image_base64: str = Field(..., description="원본 얼굴 이미지(base64)")
    style_image_base64: str  = Field(..., description="스타일 참조 이미지(base64)")
    # 선택
    pose_image_base64: Optional[str] = Field(None, description="포즈 이미지(base64)")
    resolution: Optional[int] = 512
    steps: Optional[int] = 30
    guidance: Optional[float] = 2.0
    precision: Optional[str] = "fp16"  # "fp16" | "bf16" | "fp32"
    seed: Optional[int] = None
    save_to_disk: Optional[bool] = False
    output_dir: Optional[str] = "data/output"
    id_name: Optional[str] = None
    ref_name: Optional[str] = None
    id_path: Optional[str] = None
    ref_path: Optional[str] = None

class MakeupResponse(BaseModel):
    # 팀 규약: API 레이어는 result로 감싸 반환
    # 내부에는 service.run_inference의 결과(status/result_image_base64 등)
    result: Dict[str, Any]


# ============================================================================
# Customization (커스터마이징)
# ============================================================================
class CustomizationRequest(BaseModel):
    base_image_base64: str  # 필수
    edits: List[dict]       # 필수, 예: [{"region": "lip", "intensity": 50}]

class CustomizationResponse(BaseModel):
    status: str
    result_image_base64: Optional[str]
    message: Optional[str]

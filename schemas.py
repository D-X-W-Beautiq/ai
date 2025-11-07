# schemas.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator

# ------------------------- NIA -------------------------
class NIAPredictions(BaseModel):
    # 회귀(0~100로 스케일링해 출력)
    moisture_reg: int = Field(..., ge=0, le=100)
    elasticity_reg: int = Field(..., ge=0, le=100)
    wrinkle_reg: int = Field(..., ge=0, le=100)
    pigmentation_reg: int = Field(..., ge=0, le=100)
    pore_reg: int = Field(..., ge=0, le=100)

class NIARequest(BaseModel):
    image_base64: str

class NIAResponse(BaseModel):
    status: str
    predictions: Optional[NIAPredictions] = None
    feedback: Optional[str] = None
    message: Optional[str] = None

# ---------------------- Feedback -----------------------
class FeedbackRequest(BaseModel):
    """
    최소 요구: prompt 또는 (predictions | predictions_json | predictions_json_path) 중 하나
    """
    prompt: Optional[str] = Field(default=None, description="사용자 프리-포맷 프롬프트. 단독 입력 가능.")
    predictions: Optional[Dict[str, int]] = Field(default=None, description="NIA 점수 맵(0~100). 예: {'moisture_reg': 40, ...}")
    predictions_json: Optional[str] = Field(default=None, description="predictions를 담은 JSON 문자열(또는 {'predictions': {...}} 형태의 문자열)")
    predictions_json_path: Optional[str] = Field(default=None, description="predictions.json 파일 경로(서버 로컬, 상대경로 허용)")

    @model_validator(mode="after")
    def _normalize_and_require_one(self) -> "FeedbackRequest":
        # 빈 문자열 → None 정규화 (Swagger가 ""를 기본값으로 넣는 문제 회피)
        def _nz(s: Optional[str]) -> Optional[str]:
            if isinstance(s, str) and s.strip() == "":
                return None
            return s

        self.prompt = _nz(self.prompt)
        self.predictions_json = _nz(self.predictions_json)
        self.predictions_json_path = _nz(self.predictions_json_path)

        if not any([self.prompt, self.predictions, self.predictions_json, self.predictions_json_path]):
            raise ValueError(
                "FeedbackRequest는 최소 하나의 입력이 필요합니다: "
                "prompt 또는 (predictions | predictions_json | predictions_json_path)"
            )
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                # 예1: prompt만
                {"prompt": "피부 점수 요약해 주고 제품 카테고리 2개 추천해줘."},

                # 예2: predictions 직접 전달
                {"predictions": {
                    "moisture_reg": 55, "elasticity_reg": 38, "wrinkle_reg": 31, 
                    "pigmentation_reg": 44, "pore_reg": 58
                }},

                # 예3: predictions_json(문자열) 전달
                {"predictions_json": "{\"predictions\": {\"moisture_reg\": 55, \"elasticity_reg\": 38, "
                                      "\"wrinkle_reg\": 31, \"pigmentation_reg\": 44, \"pore_reg\": 58}}"},

                # 예4: 서버 파일 경로 전달 (상대경로 허용)
                {"predictions_json_path": "data/predictions.json"}
            ]
        }
    }

class FeedbackResponse(BaseModel):
    status: str
    feedback: Optional[str] = None
    message: Optional[str] = None

# ---------------------- Product ------------------------
class ProductIn(BaseModel):
    product_id: str
    product_name: str
    brand: str
    category: str
    price: int
    review_score: float
    review_count: int
    ingredients: List[str]

class ProductRequest(BaseModel):
    skin_analysis: NIAPredictions
    recommended_categories: List[str]
    filtered_products: List[ProductIn]
    locale: str

class ProductReco(BaseModel):
    product_id: str
    reason: str

class ProductResponse(BaseModel):
    status: str
    recommendations: Optional[List[ProductReco]] = None
    message: Optional[str] = None
    error_code: Optional[str] = None

# ----------------------- Style -------------------------
class StyleRequest(BaseModel):
    source_image_base64: str = Field(..., description="사용자 얼굴 이미지 base64")
    keywords: List[str] = Field(default_factory=list, description="스타일 힌트 키워드 리스트")

class StyleResult(BaseModel):
    style_id: str = Field(default="", description="스타일 식별자")
    style_image_base64: str = Field(default="", description="추천 스타일 이미지(base64)")

class StyleResponse(BaseModel):
    status: str = Field(..., description='"success" | "error"')
    results: Optional[List[StyleResult]] = Field(default=None, description="성공 시 Top-N 스타일 결과")
    message: Optional[str] = Field(default=None, description="실패 시 메시지")

    @model_validator(mode="after")
    def _validate_union(self) -> "StyleResponse":
        if self.status == "success":
            if not self.results:
                raise ValueError("status=success 인 경우 results가 필요합니다.")
        else:
            if not self.message:
                self.message = "스타일 추천 실패"
        return self

# ---------------------- Makeup -------------------------
class MakeupRequest(BaseModel):
    source_image_base64: str
    style_image_base64: Optional[str] = None

class MakeupResponse(BaseModel):
    status: str
    result_image_base64: Optional[str] = None
    message: Optional[str] = None

# ------------------- Customization ---------------------
class EditItem(BaseModel):
    region: str  # "skin" | "eye" | "lip" | "blush"
    intensity: int

class CustomizationRequest(BaseModel):
    base_image_base64: str
    edits: List[EditItem]

class CustomizationResponse(BaseModel):
    status: str
    result_image_base64: Optional[str] = None
    message: Optional[str] = None
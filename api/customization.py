# api/customization.py
from fastapi import APIRouter
from schemas import CustomizationRequest, CustomizationResponse
from typing import Dict
import os, base64, io
from datetime import datetime
from PIL import Image

router = APIRouter(prefix="/custom", tags=["Customization"])

# 저장 경로: 환경변수로 오버라이드 가능 (기본 NIA/Makeup과 동일)
OUTPUT_DIR = os.getenv("CUSTOM_OUTPUT_DIR", os.path.join("data", "output"))

# region 매핑(서비스가 eyelid만 받는 경우 등)
_REGION_MAP: Dict[str, str] = {
    "eye": "eyelid",
    "eyelid": "eyelid",
    "lip": "lip",
    "blush": "blush",
    "skin": "skin",
}

def _b64_to_image(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")

def _image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@router.post("/apply", response_model=CustomizationResponse, summary="Apply Customization")
async def apply_customization(request: CustomizationRequest):
    """
    - 입력: base_image_base64 + edits[{region, intensity(0~100)}]
    - 동작: 서비스 호출 → 결과 이미지를 서버에 저장(NIA/메이크업과 동일) → base64 + 저장 경로 반환
    """
    try:
        from service.customization_service import run_inference

        # 1) region 정규화 & intensity 클램프
        for e in request.edits:
            e.region = _REGION_MAP.get(e.region, e.region)
            if e.intensity < 0: e.intensity = 0
            if e.intensity > 100: e.intensity = 100

        # 2) 서비스 실행 (서비스는 base64 기반 처리라고 가정)
        result = run_inference(request.model_dump())  # {status, result_image_base64?, message?}

        if result.get("status") != "success":
            return CustomizationResponse(status="error", message=result.get("message", "커스터마이즈 실패"))

        # 3) 서버 저장 (NIA/메이크업과 동일 정책)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        img_b64 = result.get("result_image_base64")
        if not img_b64:
            return CustomizationResponse(status="error", message="결과 이미지가 비어있습니다.")

        img = _b64_to_image(img_b64)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUTPUT_DIR, f"custom_{ts}.png")
        img.save(out_path)

        # 4) 응답
        return CustomizationResponse(
            status="success",
            result_image_base64=_image_to_b64(img),
            message=f"saved: {out_path}"
        )

    except Exception as e:
        return CustomizationResponse(status="error", message=f"Internal Server Error: {e}")

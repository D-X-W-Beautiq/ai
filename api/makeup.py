# api/makeup.py
from fastapi import APIRouter
from schemas import MakeupRequest, MakeupResponse
from service.makeup_service import run_inference
from io import BytesIO
from PIL import Image
import base64
import torch  # ✅ 추가

router = APIRouter(prefix="/makeup", tags=["Makeup"])

def _b64_to_pil(b64: str) -> Image.Image:
    from io import BytesIO
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")

@router.post("/simulate", response_model=MakeupResponse, response_model_exclude_none=True)
async def simulate(req: MakeupRequest):
    try:
        # 입력 검증
        if not req.source_image_base64:
            return MakeupResponse(
                status="error", message="source_image_base64가 필요합니다."
            )
        if not req.style_image_base64:
            return MakeupResponse(
                status="error", message="style_image_base64가 필요합니다."
            )

        id_img = _b64_to_pil(req.source_image_base64)
        ref_img = _b64_to_pil(req.style_image_base64)

        result_img = run_inference(
            id_image=id_img,
            makeup_image=ref_img,
            guidance_scale=getattr(req, "guidance", 1.6),
            size=getattr(req, "resolution", 512),
            num_inference_steps=getattr(req, "steps", 30),
            seed=getattr(req, "seed", None),
            device="cuda" if torch.cuda.is_available() else "cpu",  # ✅ torch 사용 가능
        )

        buf = BytesIO()
        result_img.save(buf, format="PNG")
        return MakeupResponse(status="success", result_image_base64=base64.b64encode(buf.getvalue()).decode())

    except Exception as e:
        return MakeupResponse(status="error", message=f"Internal Server Error: {e}")

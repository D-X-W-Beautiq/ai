# api/makeup.py
from fastapi import APIRouter
from schemas import MakeupRequest, MakeupResponse
import base64, io, os
from datetime import datetime
from PIL import Image

router = APIRouter(prefix="/makeup", tags=["Makeup"])

# 저장 경로 환경변수로도 오버라이드 가능 (없으면 NIA처럼 data/output 사용)
OUTPUT_DIR = os.getenv("MAKEUP_OUTPUT_DIR", os.path.join("data", "output"))

def _b64_to_image(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")

def _image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@router.post("/simulate", response_model=MakeupResponse, summary="Makeup Simulate")
def makeup_simulate(req: MakeupRequest):
    try:
        from service.makeup_service import run_inference

        # --- 입력 디코드 ---
        id_img = _b64_to_image(req.source_image_base64)

        # 스타일 이미지가 없으면 동일 이미지를 참조(Fallback)
        # 전이 효과를 보려면 클라이언트에서 style_image_base64 제공 권장
        ref_img = _b64_to_image(req.style_image_base64) if req.style_image_base64 else id_img

        # --- 추론 ---
        out_img = run_inference(
            id_image=id_img,
            makeup_image=ref_img,
            guidance_scale=1.6,
            size=512,
            num_inference_steps=30,
            seed=None
        )

        # --- NIA처럼: 서버에서 결과 저장 ---
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUTPUT_DIR, f"makeup_{ts}.png")
        out_img.save(out_path)

        # --- 응답 ---
        return MakeupResponse(
            status="success",
            result_image_base64=_image_to_b64(out_img),
            message=f"saved: {out_path}"
        )

    except ValueError as e:
        return MakeupResponse(status="error", message=str(e))
    except Exception as e:
        return MakeupResponse(status="error", message=f"메이크업 처리 중 오류: {e}")

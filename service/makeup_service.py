# service/makeup_service.py
import io
import base64
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from PIL import Image

from model_manager.makeup_manager import load_model


# ---------------- Utils ----------------
def _b64_to_image(b64: str) -> Image.Image:
    """base64 -> PIL.Image"""
    try:
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid base64 image: {e}")


def _image_to_b64(img: Image.Image) -> str:
    """PIL.Image -> base64 (PNG)"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _resize_square(img: Image.Image, res: Optional[int]) -> Image.Image:
    """정사각 리사이즈(Stable Diffusion 입력 정규화용)"""
    if not res:
        return img
    return img.resize((int(res), int(res)), Image.BICUBIC)


def _stem(s: str) -> str:
    """경로/확장자 제거한 파일명 베이스 추출(한글/공백 유지)"""
    if not s:
        return ""
    s = s.replace("\\", "/").split("/")[-1]
    if "." in s:
        s = ".".join(s.split(".")[:-1])
    return s.strip()


# ---------------- Core ----------------
def run_inference(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    스펙(간단):
    Request
    {
      "source_image_base64": "string",   # 필수 (기존 id_image_base64과 동등)
      "style_image_base64":  "string"    # 필수 (기존 ref_image_base64과 동등)
      // (선택) 아래 파라미터는 있으면 사용, 없으면 기본값
      "pose_image_base64": "string",
      "resolution": 512,
      "steps": 30,
      "guidance": 2.0,
      "precision": "fp16",
      "seed": 42,

      // (선택) 디스크 저장 옵션
      "save_to_disk": true,
      "output_dir": "data/output",
      "id_name": "고윤정", "ref_name": "스모키",
      "id_path": "data/.../고윤정.jpg", "ref_path": "data/.../스모키.jpg"
    }

    Response (성공)
    { "status": "success", "result_image_base64": "string" }

    Response (실패)
    { "status": "failed", "message": "string" }
    """
    try:
        # ---- 0) 입력 키 호환 처리 ----
        # 새 스펙 키 (우선)
        src_b64 = request.get("source_image_base64")
        sty_b64 = request.get("style_image_base64")

        # 구 스펙 키 (백워드 호환)
        if src_b64 is None and "id_image_base64" in request:
            src_b64 = request.get("id_image_base64")
        if sty_b64 is None and "ref_image_base64" in request:
            sty_b64 = request.get("ref_image_base64")

        if not src_b64:
            raise ValueError("Missing 'source_image_base64'")
        if not sty_b64:
            raise ValueError("Missing 'style_image_base64'")

        # ---- 1) 파라미터 ----
        pretrained = request.get("pretrained", "runwayml/stable-diffusion-v1-5")
        checkpoints_dir = request.get("checkpoints_dir", "checkpoints/makeup")
        precision = request.get("precision", "fp16")
        image_encoder_path = request.get("image_encoder_path", "./models/image_encoder_l")

        resolution = int(request.get("resolution", 512))
        steps = int(request.get("steps", 30))
        guidance = float(request.get("guidance", 2.0))
        seed = request.get("seed")
        if seed is not None:
            seed = int(seed)

        # 디스크 저장(선택)
        save_to_disk: bool = bool(request.get("save_to_disk", False))
        output_dir: str = request.get("output_dir", "data/output")
        id_name_hint: str = request.get("id_name", "")
        ref_name_hint: str = request.get("ref_name", "")
        id_path_hint: str = request.get("id_path", "")
        ref_path_hint: str = request.get("ref_path", "")

        # ---- 2) 모델 로드(캐시 재사용) ----
        pipe, mk_encoder, device, autocast_device, use_amp = load_model(
            pretrained=pretrained,
            checkpoints_dir=checkpoints_dir,
            precision=precision,
            image_encoder_path=image_encoder_path,
        )

        # ---- 3) 이미지 디코딩 ----
        id_img = _b64_to_image(src_b64)
        ref_img = _b64_to_image(sty_b64)
        pose_b64 = request.get("pose_image_base64")
        pose_img = _b64_to_image(pose_b64) if pose_b64 else id_img

        # 정규화
        id_img_s = _resize_square(id_img, resolution)
        ref_img_s = _resize_square(ref_img, resolution)
        pose_img_s = _resize_square(pose_img, resolution)

        # ---- 4) 생성 ----
        if seed is not None:
            torch.manual_seed(seed)

        with torch.autocast(autocast_device, enabled=use_amp):
            out = mk_encoder.generate(
                id_image=[id_img_s, pose_img_s],
                makeup_image=ref_img_s,
                guidance_scale=guidance,
                num_inference_steps=steps,
                seed=seed,
                pipe=pipe,
            )

        # ---- 5) 디스크 저장 (옵션) ----
        if save_to_disk:
            id_stem = _stem(id_name_hint or id_path_hint) or "source"
            ref_stem = _stem(ref_name_hint or ref_path_hint) or "style"
            fname = f"{id_stem}_{ref_stem}.png"
            outdir = Path(output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            outpath = outdir / fname
            try:
                out.save(outpath)
            except Exception as se:
                # 저장 실패해도 응답은 성공으로 주고, 메시지에 힌트만 남김
                # (진짜 실패를 실패 응답으로 바꾸고 싶으면 아래 return을 raise로 바꿔도 됨)
                return {"status": "failed", "message": f"Failed to save result to disk: {se}"}

        # ---- 6) 응답(간단 스펙) ----
        return {
            "status": "success",
            "result_image_base64": _image_to_b64(out)
        }

    except Exception as e:
        return {"status": "failed", "message": str(e)}

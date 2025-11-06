# service/makeup_service.py
"""
메이크업 추론 서비스 (추론 전용)
- API 레벨에서 파일 저장/응답 포맷을 처리하고,
  여기서는 이미지 전이(inference)만 책임집니다.
"""

import os
import sys
import torch
from typing import Optional, Union
from PIL import Image
import numpy as np

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 내부 모듈
from model_manager.makeup_manager import load_model
from libs.spiga_draw import get_draw  # 포즈/랜드마크 기반 draw 이미지
from facelib import FaceDetector  # 얼굴 검출기 (모델 웜업/보조용)


# ------------------------------------------------------------
# 패딩 유틸
# ------------------------------------------------------------
def resize_with_padding(pil_img: Image.Image, target: int = 512, pad_mode: str = "edge") -> Image.Image:
    """
    종횡비를 유지해 긴 변 기준으로 리사이즈한 뒤, 패딩을 넣어 정사각(512x512)으로 맞춘다.
    pad_mode: "edge" | "reflect" | "constant"
    """
    w, h = pil_img.size
    if w == 0 or h == 0:
        raise ValueError("Invalid image size")

    # 종횡비 유지 리사이즈
    if w >= h:
        new_w = target
        new_h = int(round(h * (target / w)))
    else:
        new_h = target
        new_w = int(round(w * (target / h)))

    img_resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

    # Numpy로 패딩
    arr = np.array(img_resized)
    pad_top = (target - new_h) // 2
    pad_bottom = target - new_h - pad_top
    pad_left = (target - new_w) // 2
    pad_right = target - new_w - pad_left

    if pad_mode == "constant":
        # 흰색 패딩(255)
        arr_padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=255)
    elif pad_mode == "reflect":
        # 반사 패딩
        arr_padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="reflect")
    else:
        # 가장자리 반복(edge)
        arr_padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="edge")

    return Image.fromarray(arr_padded)


# ------------------------------------------------------------
# Face Detector (옵셔널, 웜업/보조)
# ------------------------------------------------------------
_FACE_DETECTOR = None

def get_face_detector():
    """Face Detector 싱글톤 (가중치가 있으면 로컬 사용, 없으면 기본 생성)"""
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        weight_path = "./models/mobilenet0.25_Final.pth"
        if os.path.exists(weight_path):
            _FACE_DETECTOR = FaceDetector(weight_path=weight_path)
        else:
            _FACE_DETECTOR = FaceDetector()  # 내부에서 자동 다운로드 시도
    return _FACE_DETECTOR


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------
def run_inference(
    id_image: Union[Image.Image, str],
    makeup_image: Union[Image.Image, str],
    guidance_scale: float = 1.6,
    size: int = 512,
    num_inference_steps: int = 30,
    seed: Optional[int] = None,
    device: str = "cuda",
) -> Image.Image:
    """
    메이크업 전이 추론.
    Args:
        id_image: 대상 얼굴 이미지(PIL.Image or 경로)
        makeup_image: 참조 메이크업 이미지(PIL.Image or 경로)
        guidance_scale: CFG scale
        size: 정사각 리사이즈 크기
        num_inference_steps: 디퓨전 스텝 수
        seed: 고정 시드(재현성)
        device: "cuda" | "cpu"

    Returns:
        PIL.Image: 전이된 결과 이미지
    """

    # 1) 이미지 로드/전처리
    if isinstance(id_image, str):
        id_image = Image.open(id_image).convert("RGB")
    if isinstance(makeup_image, str):
        makeup_image = Image.open(makeup_image).convert("RGB")

    # 2) 512 정규화 (종횡비 유지 + 패딩)
    id_image = resize_with_padding(id_image, target=size, pad_mode="edge")
    makeup_image = resize_with_padding(makeup_image, target=size, pad_mode="edge")

    # 3) 얼굴 검출기 웜업
    _ = get_face_detector()

    # 4) 포즈/랜드마크 기반 보조 이미지 생성
    pose_image = get_draw(id_image, size=size)

    # 5) 모델 로드(캐시 사용)
    pipeline, makeup_encoder = load_model(device=device)

    # 6) 시드 고정(선택)
    if seed is not None:
        torch.manual_seed(seed)

    # 7) 전이 실행
    result_img = makeup_encoder.generate(
        id_image=[id_image, pose_image],
        makeup_image=makeup_image,
        pipe=pipeline,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )

    return result_img


# ------------------------------------------------------------
# CLI 테스트용 (API 경유가 아니라 직접 실행할 때만)
# ------------------------------------------------------------
def main():
    """
    CLI 테스트:
        python -m service.makeup_service
    """
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║              Stable-Makeup Inference Service                 ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )

    id_input = "./data/test_imgs_makeup/id/제니.jpg"
    makeup_ref = "./data/test_imgs_makeup/makeup/스모키.jpg"
    output_dir = "./data/output"

    if not os.path.exists(id_input):
        print(f"❌ Source image not found: {id_input}")
        sys.exit(1)
    if not os.path.exists(makeup_ref):
        print(f"❌ Makeup reference not found: {makeup_ref}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    id_name = os.path.basename(id_input).split(".")[0]
    makeup_name = os.path.basename(makeup_ref).split(".")[0]
    output_path = os.path.join(output_dir, f"{id_name}_{makeup_name}.png")

    try:
        print("\n" + "=" * 70)
        print("🎨 Makeup Transfer")
        print("=" * 70)
        print(f"📂 Source: {id_input}")
        print(f"📂 Makeup: {makeup_ref}")
        print("⚙️  Processing...")

        result = run_inference(
            id_image=id_input,
            makeup_image=makeup_ref,
            guidance_scale=1.6,
            size=512,
            num_inference_steps=30,
            seed=None,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        result.save(output_path)

        print(f"✅ Saved: {output_path}")
        print("=" * 70)
        print("\n🎉 Inference completed successfully!\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

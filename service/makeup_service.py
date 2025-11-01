# service/makeup_service.py
"""
메이크업 추론 서비스
"""

import os
import sys
import torch
from PIL import Image
from typing import Optional, Union, List

# 프로젝트 루트를 path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_manager.makeup_manager import load_model
from libs.spiga_draw import get_draw  # 🔧 libs에서 import
from facelib import FaceDetector 


# Face Detector 초기화 (글로벌)
_FACE_DETECTOR = None


def get_face_detector():
    """Face Detector 싱글톤"""
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        # 루트의 models 폴더 확인
        weight_path = "./models/mobilenet0.25_Final.pth"
        if os.path.exists(weight_path):
            _FACE_DETECTOR = FaceDetector(weight_path=weight_path)
        else:
            # 없으면 자동 다운로드
            _FACE_DETECTOR = FaceDetector()
    return _FACE_DETECTOR


def inference(
    id_image: Union[Image.Image, str],
    makeup_image: Union[Image.Image, str],
    guidance_scale: float = 1.6,
    size: int = 512,
    num_inference_steps: int = 30,
    seed: Optional[int] = None,
    device: str = "cuda"
) -> Image.Image:
    """
    메이크업 전이 추론
    
    Args:
        id_image: 원본 얼굴 이미지 (PIL.Image 또는 경로)
        makeup_image: 메이크업 참조 이미지 (PIL.Image 또는 경로)
        guidance_scale: 가이던스 스케일
        size: 출력 이미지 크기
        num_inference_steps: 디노이징 스텝 수
        seed: 랜덤 시드
        device: 실행 디바이스
    
    Returns:
        PIL.Image: 메이크업 전이된 결과 이미지
    """
    # 이미지 로드
    if isinstance(id_image, str):
        id_image = Image.open(id_image).convert("RGB")
    if isinstance(makeup_image, str):
        makeup_image = Image.open(makeup_image).convert("RGB")
    
    # 리사이즈
    id_image = id_image.resize((size, size))
    makeup_image = makeup_image.resize((size, size))
    
    # 포즈 이미지 생성
    detector = get_face_detector()
    pose_image = get_draw(id_image, size=size)
    
    # 모델 로드 (캐시 사용)
    pipeline, makeup_encoder = load_model(device=device)
    
    # 시드 설정
    if seed is not None:
        torch.manual_seed(seed)
    
    # 추론 실행
    result_img = makeup_encoder.generate(
        id_image=[id_image, pose_image],
        makeup_image=makeup_image,
        pipe=pipeline,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed
    )
    
    return result_img


def batch_inference(
    id_images: List[Union[Image.Image, str]],
    makeup_images: List[Union[Image.Image, str]],
    **kwargs
) -> List[Image.Image]:
    """
    배치 추론
    """
    if len(id_images) != len(makeup_images):
        raise ValueError("id_images와 makeup_images의 길이가 같아야 합니다.")
    
    results = []
    for id_img, makeup_img in zip(id_images, makeup_images):
        result = inference(id_img, makeup_img, **kwargs)
        results.append(result)
    
    return results


def main():
    """직접 실행"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              Stable-Makeup Inference Service                 ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
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
    
    id_name = os.path.basename(id_input).split('.')[0]
    makeup_name = os.path.basename(makeup_ref).split('.')[0]
    output_path = os.path.join(output_dir, f"{id_name}_{makeup_name}.png")
    
    try:
        print(f"\n{'='*70}")
        print(f"🎨 Makeup Transfer")
        print(f"{'='*70}")
        print(f"📂 Source: {id_input}")
        print(f"📂 Makeup: {makeup_ref}")
        print(f"⚙️  Processing...")
        
        result = inference(id_input, makeup_ref, guidance_scale=1.6)
        result.save(output_path)
        
        print(f"✅ Saved: {output_path}")
        print(f"{'='*70}")
        print("\n🎉 Inference completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
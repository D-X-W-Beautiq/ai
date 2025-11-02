# model_manager/makeup_manager.py
"""
메이크업 모델 로딩 및 캐시 관리
"""

import os
import sys
import torch
from typing import Optional, Tuple

# 프로젝트 루트를 path에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# libs에서 import
from libs.pipeline_sd15 import StableDiffusionControlNetPipeline
from diffusers import DDIMScheduler, ControlNetModel
from diffusers import UNet2DConditionModel as OriginalUNet2DConditionModel
from libs.detail_encoder.encoder_plus import detail_encoder

# 글로벌 캐시
_CACHED_PIPELINE = None
_CACHED_MAKEUP_ENCODER = None


def load_model(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    checkpoint_path: str = "./checkpoints/makeup",  # 🔧 루트 경로
    image_encoder_path: str = "./models/image_encoder_l",  # 🔧 루트 경로 (또는 HF)
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    force_reload: bool = False
) -> Tuple[object, object]:
    """
    메이크업 모델 로드 (캐시 사용)
    
    Args:
        model_id: Stable Diffusion 모델 ID
        checkpoint_path: 체크포인트 디렉토리 경로 (기본: ./checkpoints/makeup)
        image_encoder_path: CLIP 이미지 인코더 경로 (기본: ./models/image_encoder_l)
        device: 실행 디바이스
        dtype: 모델 데이터 타입
        force_reload: 강제 재로드 여부
    
    Returns:
        (pipeline, makeup_encoder) 튜플
    """
    global _CACHED_PIPELINE, _CACHED_MAKEUP_ENCODER
    
    # 캐시 확인
    if not force_reload and _CACHED_PIPELINE is not None and _CACHED_MAKEUP_ENCODER is not None:
        return _CACHED_PIPELINE, _CACHED_MAKEUP_ENCODER
    
    # 체크포인트 경로 설정
    makeup_encoder_path_file = os.path.join(checkpoint_path, "pytorch_model.bin")
    id_encoder_path_file = os.path.join(checkpoint_path, "pytorch_model_1.bin")
    pose_encoder_path_file = os.path.join(checkpoint_path, "pytorch_model_2.bin")
    
    # 체크포인트 존재 확인
    required_files = [makeup_encoder_path_file, id_encoder_path_file, pose_encoder_path_file]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        raise FileNotFoundError(
            f"Required checkpoint files not found in {checkpoint_path}:\n" +
            "\n".join(f"  - {os.path.basename(f)}" for f in missing_files)
        )
    
    # image_encoder_path가 로컬 경로인지 HF 모델명인지 확인
    if not os.path.exists(image_encoder_path):
        # HuggingFace에서 다운로드 (예: "openai/clip-vit-large-patch14")
        print(f"  Image encoder not found locally, will download from HuggingFace: {image_encoder_path}")
        image_encoder_path = "openai/clip-vit-large-patch14"
    
    # UNet 로드
    unet = OriginalUNet2DConditionModel.from_pretrained(
        model_id, 
        subfolder="unet",
        torch_dtype=dtype
    ).to(device)
    
    # ControlNet 초기화
    id_encoder = ControlNetModel.from_unet(unet)
    pose_encoder = ControlNetModel.from_unet(unet)
    
    # Makeup Encoder 초기화
    makeup_encoder = detail_encoder(
        unet, 
        image_encoder_path, 
        device, 
        dtype=dtype
    )
    
    # 체크포인트 로드
    id_state_dict = torch.load(id_encoder_path_file, map_location="cpu")
    pose_state_dict = torch.load(pose_encoder_path_file, map_location="cpu")
    makeup_state_dict = torch.load(makeup_encoder_path_file, map_location="cpu")
    
    id_encoder.load_state_dict(id_state_dict, strict=False)
    pose_encoder.load_state_dict(pose_state_dict, strict=False)
    makeup_encoder.load_state_dict(makeup_state_dict, strict=False)
    
    # GPU로 이동
    id_encoder.to(device, dtype=dtype)
    pose_encoder.to(device, dtype=dtype)
    makeup_encoder.to(device, dtype=dtype)
    
    # 파이프라인 생성
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        model_id,
        safety_checker=None,
        unet=unet,
        controlnet=[id_encoder, pose_encoder],
        torch_dtype=dtype
    ).to(device)
    
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    # 캐시 저장
    _CACHED_PIPELINE = pipeline
    _CACHED_MAKEUP_ENCODER = makeup_encoder
    
    return pipeline, makeup_encoder


def clear_cache():
    """캐시된 모델 해제"""
    global _CACHED_PIPELINE, _CACHED_MAKEUP_ENCODER
    
    _CACHED_PIPELINE = None
    _CACHED_MAKEUP_ENCODER = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

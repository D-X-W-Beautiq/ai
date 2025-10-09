# model_manager/makeup_manager.py
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from transformers import AutoTokenizer, PretrainedConfig

# 내부 모듈 (libs 폴더)
from libs.pipeline_sd15 import StableDiffusionControlNetPipeline
from libs.detail_encoder.encoder_plus import detail_encoder

# ===== 글로벌 캐시 =====
_ctx: Optional[Tuple[object, object, str, str, bool]] = None
# (pipe, mk_encoder, device, autocast_device, use_amp)


def _import_text_encoder_cls(model_name_or_path: str, revision: str = None):
    cfg = PretrainedConfig.from_pretrained(
        model_name_or_path, subfolder="text_encoder", revision=revision
    )
    if cfg.architectures[0] == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    raise ValueError("Unsupported text encoder.")


def _load_base_models(pretrained: str, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained, subfolder="tokenizer", use_fast=False
    )
    text_encoder_cls = _import_text_encoder_cls(pretrained, None)
    text_encoder = text_encoder_cls.from_pretrained(pretrained, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained, subfolder="unet")

    vae.to(device, dtype=dtype)
    unet.to(device, dtype=dtype)
    text_encoder.to(device, dtype=dtype)
    return tokenizer, text_encoder, vae, unet


def _build_pipeline(pretrained, vae, text_encoder, tokenizer, unet,
                    controlnet_id, controlnet_pose, device, dtype):
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        pretrained,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=[controlnet_id, controlnet_pose],
        safety_checker=None,
        torch_dtype=dtype,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    return pipe.to(device)


def _pick_dtype(precision: str):
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return torch.float32


def load_model(
    pretrained: str = "runwayml/stable-diffusion-v1-5",
    checkpoints_dir: str = "checkpoints/makeup",
    precision: str = "fp16",
    image_encoder_path: Optional[str] = "./models/image_encoder_l",
) -> Tuple[object, object, str, str, bool]:
    """
    파이프라인과 디테일 인코더를 전역 캐시로 로드.
    Returns: (pipe, mk_encoder, device, autocast_device, use_amp)
    """
    global _ctx
    if _ctx is not None:
        return _ctx

    # 환경 안전장치
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _pick_dtype(precision)

    # Torch2 SDPA가 CLIPVision에서 이슈를 일으킬 수 있어 비활성화
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # 1) 베이스 모델
    tokenizer, text_encoder, vae, unet = _load_base_models(pretrained, device, dtype)

    # 2) 컨트롤넷 두 개 준비 + 선택적 가중치 로드
    controlnet_id = ControlNetModel.from_unet(unet)
    controlnet_pose = ControlNetModel.from_unet(unet)

    ckpt_dir = Path(checkpoints_dir)
    mk_path = ckpt_dir / "pytorch_model.bin"
    id_path = ckpt_dir / "pytorch_model_1.bin"
    pose_path = ckpt_dir / "pytorch_model_2.bin"

    if id_path.exists():
        controlnet_id.load_state_dict(torch.load(id_path, map_location="cpu"), strict=False)
    if pose_path.exists():
        controlnet_pose.load_state_dict(torch.load(pose_path, map_location="cpu"), strict=False)

    controlnet_id.to(device, dtype=dtype)
    controlnet_pose.to(device, dtype=dtype)

    # 3) 메이크업 디테일 인코더
    if not image_encoder_path or not os.path.exists(image_encoder_path):
        image_encoder_path = "openai/clip-vit-large-patch14"

    mk_encoder = detail_encoder(
        unet=unet,
        image_encoder_path=image_encoder_path,
        device=device,
        dtype=dtype
    )
    if mk_path.exists():
        try:
            sd = torch.load(mk_path, map_location="cpu")
            mk_encoder.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"[makeup_manager] warn: failed to load mk_encoder weights: {e}")

    # 4) 파이프라인 구성
    pipe = _build_pipeline(
        pretrained, vae, text_encoder, tokenizer, unet,
        controlnet_id, controlnet_pose, device, dtype
    )
    pipe.set_progress_bar_config(disable=True)

    autocast_device = "cuda" if device == "cuda" else "cpu"
    use_amp = (dtype != torch.float32)

    _ctx = (pipe, mk_encoder, device, autocast_device, use_amp)
    return _ctx

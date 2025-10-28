# detail_encoder/encoder_plus.py

from typing import List
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from torchvision import transforms
from transformers import CLIPImageProcessor
from transformers import CLIPVisionModel as OriginalCLIPVisionModel

from ._clip import CLIPVisionModel
from .resampler import Resampler

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

if is_torch2_available():
    from .attention_processor import (
        SSRAttnProcessor2_0 as SSRAttnProcessor,
        AttnProcessor2_0 as AttnProcessor,
    )
else:
    from .attention_processor import SSRAttnProcessor, AttnProcessor


class detail_encoder(nn.Module):
    """from SSR-encoder"""
    def __init__(self, unet, image_encoder_path, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # ---- Load CLIP-ViT-L/14 (force eager attention to avoid SDPA issue) ----
        # NOTE: We load HF's official vision model, copy weights into our local CLIPVisionModel,
        #       then free the original to save memory.
        clip_encoder = OriginalCLIPVisionModel.from_pretrained(
            image_encoder_path,
            attn_implementation="eager",
        )
        # Be explicit in case transformers tries to flip it later
        clip_encoder.config.attn_implementation = "eager"

        # Our local implementation that mirrors the config
        self.image_encoder = CLIPVisionModel(clip_encoder.config)

        # Copy weights from the HF model
        state_dict = clip_encoder.state_dict()
        missing, unexpected = self.image_encoder.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            # Not fatal; just loggable (leave as print to avoid importing logger here)
            print(f"[detail_encoder] Loaded CLIP weights with missing={len(missing)}, unexpected={len(unexpected)}")

        # Move to device/dtype and freeze (no grad)
        self.image_encoder.to(self.device, dtype=self.dtype)
        for p in self.image_encoder.parameters():
            p.requires_grad = False

        del clip_encoder

        self.clip_image_processor = CLIPImageProcessor()

        # ---- Install SSR Attention processors on UNet ----
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim

            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                # Fallback; should not really happen
                hidden_size = unet.config.block_out_channels[0]

            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = SSRAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=1024,
                    scale=1,
                ).to(self.device, dtype=self.dtype)

        unet.set_attn_processor(attn_procs)
        adapter_modules = nn.ModuleList(unet.attn_processors.values())
        self.SSR_layers = adapter_modules.to(self.device, dtype=self.dtype)

        # Resampler head
        self.resampler = self.init_proj()

    def init_proj(self):
        resampler = Resampler().to(self.device, dtype=self.dtype)
        return resampler

    def forward(self, img):
        """
        img: (B, 3, H, W) already normalized for CLIP if upstream handled it.
        Returns: (B, <proj_dim>, <seq_len>) depending on your Resampler
        """
        outputs = self.image_encoder(img, output_hidden_states=True)
        # take every other hidden state starting from layer 2 (as in SSR)
        image_embeds_list = outputs["hidden_states"][2::2]
        image_embeds = torch.cat(image_embeds_list, dim=1)
        image_embeds = self.resampler(image_embeds)
        return image_embeds

    @torch.inference_mode()
    def get_image_embeds(self, pil_image):
        """
        Accepts a PIL.Image or list of PIL.Image, returns (cond_embeds, uncond_embeds)
        """
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]

        batch = []
        for pil in pil_image:
            tensor_image = self.clip_image_processor(images=pil, return_tensors="pt").pixel_values
            tensor_image = tensor_image.to(self.device, dtype=self.dtype)
            batch.append(tensor_image)

        clip_image = torch.cat(batch, dim=0)

        # conditional
        cond_out = self.image_encoder(clip_image, output_hidden_states=True)
        cond_list = cond_out["hidden_states"][2::2]
        cond = torch.cat(cond_list, dim=1)

        # unconditional (zeros-like)
        zeros = torch.zeros_like(clip_image)
        uncond_out = self.image_encoder(zeros, output_hidden_states=True)
        uncond_list = uncond_out["hidden_states"][2::2]
        uncond = torch.cat(uncond_list, dim=1)

        cond = self.resampler(cond)
        uncond = self.resampler(uncond)
        return cond, uncond

    def generate(
        self,
        id_image,
        makeup_image,
        seed=None,
        guidance_scale=2,
        num_inference_steps=30,
        pipe=None,
        **kwargs,
    ):
        """
        id_image: list or tuple like [id_rgb, pose_rgb] as used by your pipeline
        makeup_image: PIL.Image (reference)
        """
        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds(makeup_image)

        prompt_embeds = image_prompt_embeds
        negative_prompt_embeds = uncond_image_prompt_embeds

        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        image = pipe(
            image=id_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images[0]

        return image

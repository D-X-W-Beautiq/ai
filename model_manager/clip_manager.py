import torch
from transformers import CLIPProcessor, CLIPModel

_model = None
_processor = None

def load_clip(model_name="openai/clip-vit-base-patch32"):
    """
    CLIP 모델과 Processor를 한 번만 로드하고 반환
    """
    global _model, _processor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _model is None or _processor is None:
        _model = CLIPModel.from_pretrained(model_name).to(device)
        _processor = CLIPProcessor.from_pretrained(model_name)
    return _model, _processor, device
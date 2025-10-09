# model_manager/customization_manager.py
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

_model = None
_processor = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_customization_model():
    """
    SegFormer 모델을 전역 1회만 로드하여 재사용
    """
    global _model, _processor
    if _model is None or _processor is None:
        _processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        _model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        _model.to(_device).eval()
    return _model, _processor, _device
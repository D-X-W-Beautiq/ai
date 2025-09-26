# app/model_manager/customization_manager.py
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

processor = None
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    global processor, model
    if processor is None or model is None:
        processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        model.to(device).eval()
    return processor, model
import torch
import torch.nn as nn
from torchvision import models
import os
from pathlib import Path

_class_models = None
_regression_models = None
_device = None

def get_checkpoint_path():
    """체크포인트 기본 경로 반환 (상대경로)"""
    # 현재 파일 기준 상대 경로
    current_dir = Path(__file__).parent.parent.parent  # app/model_manager -> project_root
    checkpoint_dir = current_dir / "checkpoints" / "nia"
    return str(checkpoint_dir)

def load_classification_models(checkpoint_path=None):
    """Classification 모델 로딩 (dryness, pigmentation, pore, sagging, wrinkle)"""
    global _class_models, _device
    
    if _class_models is not None:
        return _class_models, _device
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(get_checkpoint_path(), "class")
    
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_num_class = {
        "dryness": 5,
        "pigmentation": 6,
        "pore": 6,
        "sagging": 6,
        "wrinkle": 7
    }
    
    _class_models = {}
    
    for key, num_classes in model_num_class.items():
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
        
        model_path = os.path.join(checkpoint_path, key, "state_dict.bin")
        
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=_device)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(_device)
        model.eval()
        _class_models[key] = model
        print(f"✓ Loaded classification model: {key}")
    
    return _class_models, _device

def load_regression_models(checkpoint_path=None):
    """Regression 모델 로딩 (moisture, elasticity_R2, wrinkle_Ra, pigmentation, pore)"""
    global _regression_models, _device
    
    if _regression_models is not None:
        return _regression_models, _device
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(get_checkpoint_path(), "regression")
    
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_num_class = {
        "pigmentation": 1,
        "moisture": 1,
        "elasticity_R2": 1,
        "wrinkle_Ra": 1,
        "pore": 1,
    }
    
    _regression_models = {}
    
    for key, num_outputs in model_num_class.items():
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_outputs, bias=True)
        
        model_path = os.path.join(checkpoint_path, key, "state_dict.bin")
        
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=_device)
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(_device)
        model.eval()
        _regression_models[key] = model
        print(f"✓ Loaded regression model: {key}")
    
    return _regression_models, _device

def get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import base64
from io import BytesIO
import mediapipe as mp
import sys
import json
import os
from pathlib import Path

sys.path.insert(0, '/content/drive/MyDrive')

from app.model_manager.nia_manager import (
    load_classification_models,
    load_regression_models,
    get_device
)

def base64_to_image(base64_string):
    """Base64 문자열을 PIL Image로 변환"""
    if 'base64,' in base64_string:
        base64_string = base64_string.split('base64,')[1]
    
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

def detect_and_crop_face(image, margin=0.2):
    """MediaPipe를 사용한 얼굴 검출 및 크롭"""
    image_np = np.array(image)
    
    try:
        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.5
        ) as face_detection:
            results = face_detection.process(image_np)
            
            if results.detections:
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image_np.shape
                
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                margin_x = int(width * margin)
                margin_y = int(height * margin)
                
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(w, x + width + margin_x)
                y2 = min(h, y + height + margin_y)
                
                cropped_np = image_np[y1:y2, x1:x2]
                return Image.fromarray(cropped_np)
    
    except Exception as e:
        print(f"Face detection failed: {e}, using original image")
    
    return image

def preprocess_image(image, resolution=256, crop_face=True):
    """이미지 전처리"""
    if crop_face:
        image = detect_and_crop_face(image)
    
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    return transform(image).unsqueeze(0)

def denormalize_regression(value, indicator):
    """회귀 예측값 역정규화"""
    if indicator == "elasticity_R2":
        return float(value)
    elif indicator == "moisture":
        return int(value * 100)
    elif indicator == "wrinkle_Ra":
        return int(value * 50)
    elif indicator in ["pigmentation", "count"]:
        return int(value * 350)
    elif indicator == "pore":
        return int(value * 2600)
    else:
        return float(value)

def run_inference(request: dict) -> dict:
    """NIA 피부 분석 추론"""
    try:
        if "image_base64" not in request:
            raise ValueError("Missing required field: image_base64")
        
        crop_face = request.get("crop_face", True)
        
        image = base64_to_image(request["image_base64"])
        image_tensor = preprocess_image(image, resolution=256, crop_face=crop_face)
        
        class_models, device = load_classification_models()
        regression_models, _ = load_regression_models()
        
        image_tensor = image_tensor.to(device)
        
        predictions = {}
        
        # Classification
        with torch.no_grad():
            for model_name, model in class_models.items():
                output = model(image_tensor)
                predicted_class = output.argmax(dim=1).item()
                predictions[model_name] = int(predicted_class)
        
        # Regression
        with torch.no_grad():
            regression_raw = {}
            
            for model_name, model in regression_models.items():
                output = model(image_tensor)
                raw_value = output.squeeze().item()
                actual_value = denormalize_regression(raw_value, model_name)
                regression_raw[model_name] = actual_value
        
        # 부위별 분리
        predictions["forehead_pigmentation"] = regression_raw.get("pigmentation", 0)
        predictions["cheek_pigmentation"] = regression_raw.get("pigmentation", 0)
        
        moisture_value = regression_raw.get("moisture", 0)
        predictions["forehead_moisture"] = moisture_value
        predictions["cheek_moisture"] = moisture_value
        predictions["chin_moisture"] = moisture_value
        
        elasticity_value = regression_raw.get("elasticity_R2", 0.0)
        predictions["forehead_elasticity_R2"] = elasticity_value
        predictions["cheek_elasticity_R2"] = elasticity_value
        predictions["chin_elasticity_R2"] = elasticity_value
        
        predictions["perocular_wrinkle_Ra"] = regression_raw.get("wrinkle_Ra", 0)
        predictions["cheek_pore"] = regression_raw.get("pore", 0)
        
        result = {
            "status": "success",
            "predictions": predictions
        }
        
        # JSON 파일로 저장 (상대 경로: app/service → data/)
        current_dir = Path(__file__).parent.parent.parent  # app/service → project_root
        data_dir = current_dir / "data"
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / "predictions.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"NIA 결과 저장: {output_path}")
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
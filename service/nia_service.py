import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import base64
from io import BytesIO
import mediapipe as mp
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_manager.nia_manager import (
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

def normalize(value, min_val, max_val):
    """값을 0~1로 정규화 (클리핑 포함)"""
    value = max(min_val, min(value, max_val))
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

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

def convert_to_score(regression_raw):
    """회귀 예측값을 0~100 점수로 변환"""

    # 좋은 지표 (높을수록 좋음): 정변환
    moisture_value = regression_raw.get("moisture", 0)
    moisture_score = int(normalize(moisture_value, 0, 100) * 100)

    elasticity_value = regression_raw.get("elasticity_R2", 0.0)
    elasticity_score = int(normalize(elasticity_value, 0.0, 1.0) * 100)

    # 나쁜 지표 (높을수록 나쁨): 역변환
    pigmentation_value = regression_raw.get("pigmentation", 0)
    pigmentation_score = int(100 - normalize(pigmentation_value, 0, 350) * 100)

    wrinkle_value = regression_raw.get("wrinkle_Ra", 0)
    wrinkle_score = int(100 - normalize(wrinkle_value, 0, 50) * 100)

    pore_value = regression_raw.get("pore", 0)
    pore_score = int(100 - normalize(pore_value, 0, 2600) * 100)

    return {
        "moisture_reg": moisture_score,
        "elasticity_reg": elasticity_score,
        "wrinkle_reg": wrinkle_score,
        "pigmentation_reg": pigmentation_score,
        "pore_reg": pore_score
    }

def run_inference(request: dict) -> dict:
    """NIA 피부 분석 추론"""
    try:
        if "image_base64" not in request:
            raise ValueError("Missing required field: image_base64")

        # crop_face는 내부적으로만 사용 (기본값 True)
        crop_face = request.get("crop_face", True)

        image = base64_to_image(request["image_base64"])
        image_tensor = preprocess_image(image, resolution=256, crop_face=crop_face)

        regression_models, device = load_regression_models()

        image_tensor = image_tensor.to(device)

        # Regression 추론
        with torch.no_grad():
            regression_raw = {}

            for model_name, model in regression_models.items():
                output = model(image_tensor)
                raw_value = output.squeeze().item()
                actual_value = denormalize_regression(raw_value, model_name)
                regression_raw[model_name] = actual_value

        # 0~100 점수로 변환
        predictions = convert_to_score(regression_raw)

        result = {
            "status": "success",
            "predictions": predictions
        }

        # JSON 파일로 저장 (상대 경로: ai/service → ai/)
        current_dir = Path(__file__).parent.parent  # ai/service → ai/
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
# service/customization_service.py
import base64
import io
import torch
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import mediapipe as mp
import torch.nn.functional as F
from model_manager.customization_manager import load_customization_model


# ==========================
# 유틸 함수
# ==========================
def smooth_mask(mask, blur_size=21):
    mask_float = mask.astype(np.float32)
    mask_blur = cv2.GaussianBlur(mask_float, (blur_size, blur_size), 0)
    return mask_blur / mask_blur.max() if mask_blur.max() > 0 else mask_blur


def extract_region_color(image, mask):
    region_pixels = image[mask.astype(bool)]
    if len(region_pixels) == 0:
        return (0, 0, 0)
    return tuple(region_pixels.mean(axis=0).astype(int))


def apply_intensity_mask(image, mask, intensity=50, blur_size=21):
    scale_factor = 0.01
    intensity = (intensity - 50) * scale_factor
    smooth = smooth_mask(mask, blur_size)
    base_color = extract_region_color(image, mask)
    overlay = np.zeros_like(image, dtype=np.float32)
    overlay[:] = base_color

    if intensity >= 0:
        blended = image.astype(np.float32) * (1 - intensity * smooth[..., None]) + overlay * (intensity * smooth[..., None])
    else:
        intensity = intensity * 0.25
        target_overlay = np.full_like(image, 255, dtype=np.float32)
        blended = image.astype(np.float32) * (1 + intensity * smooth[..., None]) + target_overlay * (-intensity * smooth[..., None])

    return np.clip(blended, 0, 255).astype(np.uint8)


def create_eyelid_mask(mask_resized, eye_label, brow_label, top_offset=2, bottom_offset=2):
    eye_coords = np.column_stack(np.where(mask_resized == eye_label))
    brow_coords = np.column_stack(np.where(mask_resized == brow_label))
    if len(eye_coords) == 0 or len(brow_coords) == 0:
        return np.zeros_like(mask_resized, dtype=bool)

    all_coords = np.vstack((eye_coords, brow_coords))
    min_y = int(min(eye_coords[:, 0].min(), brow_coords[:, 0].min()) + top_offset)
    max_y = int(max(eye_coords[:, 0].max(), brow_coords[:, 0].max()) - bottom_offset)
    region_coords = all_coords[(all_coords[:, 0] >= min_y) & (all_coords[:, 0] <= max_y)]

    if len(region_coords) == 0:
        return np.zeros_like(mask_resized, dtype=bool)

    hull = cv2.convexHull(region_coords[:, [1, 0]].astype(np.int32))
    mask_out = np.zeros_like(mask_resized, dtype=np.uint8)
    cv2.fillConvexPoly(mask_out, hull, 1)
    mask_out[(mask_resized == eye_label) | (mask_resized == brow_label)] = 0
    return mask_out.astype(bool)


def create_landmarks_df(image_rgb):
    h, w, _ = image_rgb.shape
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        coords = [{'x': int(lm.x * w), 'y': int(lm.y * h)} for lm in landmarks]
        return pd.DataFrame(coords)


def apply_blusher_intensity(image_array, landmarks_df, intensity=50, blur_size=121):
    h, w, _ = image_array.shape
    blush_mask = np.zeros((h, w), dtype=np.uint8)
    LEFT_CHEEK_INDEXES = [123, 117, 118, 119, 56, 205, 207, 192, 213]
    RIGHT_CHEEK_INDEXES = [352, 346, 347, 348, 266, 425, 427, 416, 352]
    left_cheek_points = landmarks_df.loc[LEFT_CHEEK_INDEXES, ['x', 'y']].values.astype(np.int32)
    right_cheek_points = landmarks_df.loc[RIGHT_CHEEK_INDEXES, ['x', 'y']].values.astype(np.int32)
    cv2.fillPoly(blush_mask, [left_cheek_points], 1)
    cv2.fillPoly(blush_mask, [right_cheek_points], 1)
    return apply_intensity_mask(image_array, blush_mask, intensity=intensity, blur_size=blur_size)


# ==========================
# 메인 추론 함수
# ==========================
def run_inference(request: dict) -> dict:
    try:
        if "base_image_base64" not in request:
            raise ValueError("Missing key: base_image_base64")
        if "edits" not in request or not isinstance(request["edits"], list):
            raise ValueError("Missing key or invalid type: edits (must be list)")

        # 모델 로드
        model, processor, device = load_customization_model()

        # base64 → PIL → np.array
        image_data = base64.b64decode(request["base_image_base64"])
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_np = np.array(image)

        # SegFormer 추론: mask 생성
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            mask_resized = F.interpolate(mask_tensor, size=image_np.shape[:2], mode="nearest").squeeze().numpy().astype(np.int64)

        skin_mask = np.logical_or(mask_resized == 1, mask_resized == 2)
        lip_mask = np.logical_or(mask_resized == 11, mask_resized == 12)
        left_eyelid = create_eyelid_mask(mask_resized, 4, 6)
        right_eyelid = create_eyelid_mask(mask_resized, 5, 7)
        eyelid_mask = np.logical_or(left_eyelid, right_eyelid)

        # Mediapipe landmarks (볼터치)
        landmarks_df = create_landmarks_df(image_np)

        # edits 적용
        for edit in request["edits"]:
            region = edit.get("region")
            intensity = edit.get("intensity", 50)
            if region == "skin":
                image_np = apply_intensity_mask(image_np, skin_mask, intensity=intensity)
            elif region == "lip":
                image_np = apply_intensity_mask(image_np, lip_mask, intensity=intensity)
            elif region == "eyelid":
                image_np = apply_intensity_mask(image_np, eyelid_mask, intensity=intensity)
            elif region == "blush" and landmarks_df is not None:
                image_np = apply_blusher_intensity(image_np, landmarks_df, intensity=intensity)

        # 결과 이미지 → base64
        result_img = Image.fromarray(image_np)
        buffered = io.BytesIO()
        result_img.save(buffered, format="JPEG")
        result_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {"status": "success", "result_image_base64": result_b64}

    except Exception as e:
        return {"status": "failed", "message": str(e)}

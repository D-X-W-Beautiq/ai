# app/service/customization_service.py
import base64
import io
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from app.model_manager.customization_manager import load_model
import mediapipe as mp
import torch
import torch.nn.functional as F

# ======================
# 유틸 함수 (기존 코드 그대로)
# ======================
def smooth_mask(mask, blur_size=21):
    mask_float = mask.astype(np.float32)
    mask_blur = cv2.GaussianBlur(mask_float, (blur_size, blur_size), 0)
    mask_norm = mask_blur / mask_blur.max() if mask_blur.max() > 0 else mask_blur
    return mask_norm

def extract_region_color(image, mask):
    region_pixels = image[mask.astype(bool)]
    if len(region_pixels) == 0:
        return (0, 0, 0)
    mean_color = region_pixels.mean(axis=0)
    return tuple(mean_color.astype(int))

def apply_intensity_mask(image, mask, intensity=0.3, blur_size=21, color=None):
    smooth = smooth_mask(mask, blur_size)
    if color is None:
        base_color = extract_region_color(image, mask)
    else:
        base_color = color
    overlay = np.zeros_like(image, dtype=np.float32)
    overlay[:] = base_color
    blended = image.astype(np.float32) * (1 - intensity * smooth[..., None]) + overlay * (intensity * smooth[..., None])
    return blended.astype(np.uint8)

def create_eyelid_mask(mask_resized, eye_label, brow_label, top_offset=2, bottom_offset=2):
    eye_coords = np.column_stack(np.where(mask_resized == eye_label))
    brow_coords = np.column_stack(np.where(mask_resized == brow_label))
    if len(eye_coords) == 0 or len(brow_coords) == 0:
        return np.zeros_like(mask_resized, dtype=bool)
    all_coords = np.vstack((eye_coords, brow_coords))
    min_y = int(min(eye_coords[:,0].min(), brow_coords[:,0].min()) + top_offset)
    max_y = int(max(eye_coords[:,0].max(), brow_coords[:,0].max()) - bottom_offset)
    region_coords = all_coords[(all_coords[:,0] >= min_y) & (all_coords[:,0] <= max_y)]
    if len(region_coords) == 0:
        return np.zeros_like(mask_resized, dtype=bool)
    hull = cv2.convexHull(region_coords[:, [1,0]].astype(np.int32))
    mask_out = np.zeros_like(mask_resized, dtype=np.uint8)
    cv2.fillConvexPoly(mask_out, hull, 1)
    mask_out[(mask_resized == eye_label) | (mask_resized == brow_label)] = 0
    return mask_out.astype(bool)

def apply_blusher_intensity(image_array, landmarks_df, intensity=0.4, blur_size=121):
    h, w, _ = image_array.shape
    blush_mask = np.zeros((h, w), dtype=np.uint8)
    LEFT_CHEEK_INDEXES = [123, 117, 118, 119, 56, 205, 207, 192, 213]
    RIGHT_CHEEK_INDEXES = [352, 346, 347, 348, 266, 425, 427, 416, 352]
    left_cheek_points = landmarks_df.loc[LEFT_CHEEK_INDEXES, ['x', 'y']].values.astype(np.int32)
    right_cheek_points = landmarks_df.loc[RIGHT_CHEEK_INDEXES, ['x', 'y']].values.astype(np.int32)
    cv2.fillPoly(blush_mask, [left_cheek_points], 1)
    cv2.fillPoly(blush_mask, [right_cheek_points], 1)
    return apply_intensity_mask(image_array, blush_mask, intensity=intensity, blur_size=blur_size)

def create_landmarks_df(image_rgb):
    h, w, _ = image_rgb.shape
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        coords = [{'x': int(lm.x * w), 'y': int(lm.y * h)} for lm in landmarks]
        df = pd.DataFrame(coords)
        return df

def pil_from_base64(base64_str):
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return img

# ======================
# run_inference
# ======================
def run_inference(request: dict) -> dict:
    try:
        if "base_image_base64" not in request:
            raise ValueError("No source image provided")
        img = pil_from_base64(request["base_image_base64"])
        img_np = np.array(img)

        # edits 적용 (기본값)
        skin_alpha = lip_alpha = eyelid_alpha = blush_intensity = 0.3
        edits = request.get("edits", [])
        for edit in edits:
            region = edit.get("region")
            intensity = edit.get("intensity", 30) / 100.0
            color_dict = edit.get("color")
            color = None
            if color_dict:
                color = (color_dict.get("r",0), color_dict.get("g",0), color_dict.get("b",0))
            if region == "skin":
                skin_alpha = intensity
            elif region == "lip":
                lip_alpha = intensity
            elif region == "eye":
                eyelid_alpha = intensity
            elif region == "blush":
                blush_intensity = intensity

        # SegFormer 모델 로드
        processor, model = load_model()
        inputs = processor(images=img, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
        mask_resized = F.interpolate(mask_tensor, size=img_np.shape[:2], mode="nearest").squeeze().numpy().astype(np.int64)

        skin_mask = np.logical_or(mask_resized == 1, mask_resized == 2)
        lip_mask = np.logical_or(mask_resized == 11, mask_resized == 12)
        left_eyelid = create_eyelid_mask(mask_resized, 4, 6)
        right_eyelid = create_eyelid_mask(mask_resized, 5, 7)
        eyelid_mask = np.logical_or(left_eyelid, right_eyelid)

        result_img = img_np.copy()
        result_img = apply_intensity_mask(result_img, skin_mask, intensity=skin_alpha)
        result_img = apply_intensity_mask(result_img, lip_mask, intensity=lip_alpha)
        result_img = apply_intensity_mask(result_img, eyelid_mask, intensity=eyelid_alpha)

        landmarks_df = create_landmarks_df(img_np)
        if landmarks_df is not None:
            result_img = apply_blusher_intensity(result_img, landmarks_df, intensity=blush_intensity)

        # numpy -> base64
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
        result_b64 = base64.b64encode(buffer).decode()

        return {"status": "success", "result_image_base64": result_b64}

    except Exception as e:
        return {"status": "failed", "message": str(e)}
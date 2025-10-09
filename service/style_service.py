import io
import base64
import torch
import json
import os
from PIL import Image
from model_manager.clip_manager import load_clip

# ==============================================
# 유틸: Base64 변환 함수
# ==============================================
def _decode_image(base64_str: str) -> Image.Image:
    """Base64 → PIL.Image 변환"""
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def _encode_image(image: Image.Image) -> str:
    """PIL.Image → Base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ==============================================
# 데이터셋 로드
# ==============================================
def _load_dataset(json_dir: str):
    """4개의 JSON 파일을 로드"""
    json_files = [
        "makeup_captions_mood_detailed.json",
        "makeup_captions_mood_final.json",
        "makeup_captions_tone_detailed.json",
        "makeup_captions_tone_final.json"
    ]

    dataset = []
    for jf in json_files:
        json_path = os.path.join(json_dir, jf)
        if not os.path.exists(json_path):
            print(f"[경고] JSON 없음: {json_path}")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            if "image_path" in item:  # detailed JSON
                img_path = os.path.join(json_dir, item["image_path"])
                if os.path.exists(img_path):
                    text = item.get("caption", {}).get("sentence_english", "")
                    dataset.append({
                        "style_id": item.get("style_id", ""),
                        "style_image_base64": None,
                        "embedding": item.get("embedding", []),
                        "caption": text,
                        "image_path": img_path
                    })
            elif "request" in item:  # final JSON
                img_path = os.path.join(json_dir, item["request"]["이미지경로"])
                if os.path.exists(img_path):
                    text = item["response"].get("caption", "") or item["response"].get("prompt_en", "")
                    dataset.append({
                        "style_id": item.get("style_id", ""),
                        "style_image_base64": None,
                        "embedding": item.get("embedding", []),
                        "caption": text,
                        "image_path": img_path
                    })

    return dataset

# ==============================================
# 스타일 추천 (메인 함수)
# ==============================================
def run_inference(request: dict, json_dir: str) -> dict:
    """
    request = {
        "source_image_base64": "string",
        "keywords": ["pink blush", "warm tone", ...]
    }
    """
    try:
        if "source_image_base64" not in request:
            raise ValueError("Missing key: source_image_base64")
        if "keywords" not in request or not isinstance(request["keywords"], list):
            raise ValueError("Missing or invalid key: keywords")

        # 사용자 이미지 디코딩
        user_image = _decode_image(request["source_image_base64"])

        # 모델 로드
        model, processor, device = load_clip()

        # 사용자 이미지 임베딩
        user_inputs = processor(images=user_image, return_tensors="pt").to(device)
        with torch.no_grad():
            user_emb = model.get_image_features(**user_inputs)
            user_emb = user_emb / user_emb.norm(p=2, dim=-1, keepdim=True)

        # 키워드 → 텍스트 임베딩
        caption = "A style with " + ", ".join(request["keywords"]) + "."
        text_inputs = processor(text=[caption], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_emb = model.get_text_features(**text_inputs)
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)

        # 데이터셋 로드
        dataset = _load_dataset(json_dir)
        results = []

        for item in dataset:
            emb = torch.tensor(item["embedding"]).to(device)
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)

            score = torch.matmul(user_emb, emb.T).item()
            results.append({
                "style_id": item["style_id"],
                "style_image_base64": item["style_image_base64"],
                "score": round(score, 4)
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)[:3]

        return {"status": "success", "results": results}

    except Exception as e:
        return {"status": "failed", "message": str(e)}

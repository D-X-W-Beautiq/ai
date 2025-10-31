# service/style_service.py
import io
import base64
import torch
import json
import os
from PIL import Image
from model_manager.clip_manager import load_clip

def _decode_image(base64_str: str) -> Image.Image:
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def _encode_image(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _load_dataset(json_dir: str):
    dataset = []
    json_files = [
        os.path.join(json_dir, "style_data.json"),
        os.path.join(json_dir, "style_data_2.json"),
        os.path.join(json_dir, "final_style_data.json"),
    ]
    for json_path in json_files:
        if not os.path.exists(json_path):
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
                        "embedding": item.get("embedding", []),
                        "caption": text,
                        "image_path": img_path
                    })
    return dataset

def run_inference(request: dict, json_dir: str = "data/style-recommendation") -> dict:
    try:
        if "source_image_base64" not in request:
            raise ValueError("Missing key: source_image_base64")
        if "keywords" not in request or not isinstance(request["keywords"], list):
            raise ValueError("Missing or invalid key: keywords")

        user_image = _decode_image(request["source_image_base64"])
        model, processor, device = load_clip()

        # 사용자 임베딩
        user_inputs = processor(images=user_image, return_tensors="pt").to(device)
        with torch.no_grad():
            user_emb = model.get_image_features(**user_inputs)
            user_emb = user_emb / user_emb.norm(p=2, dim=-1, keepdim=True)

        # 텍스트 임베딩 (힌트)
        caption = "A style with " + ", ".join(request["keywords"]) + "."
        text_inputs = processor(text=[caption], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_emb = model.get_text_features(**text_inputs)
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)

        dataset = _load_dataset(json_dir)
        results = []
        for item in dataset:
            img_path = item.get("image_path")
            if not img_path or not os.path.exists(img_path):
                continue

            # 1) 임베딩이 있으면 활용
            emb = item.get("embedding")
            if emb:
                emb = torch.tensor(emb, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                # 2) 임베딩이 없으면 이미지에서 즉석 계산
                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception:
                    continue
                img_inputs = processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    emb = model.get_image_features(**img_inputs)

            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
            score_img = torch.matmul(user_emb, emb.T).item()
            score_txt = torch.matmul(text_emb, text_emb.T).item()  # 가벼운 가중치용
            score = round(0.85 * score_img + 0.15 * score_txt, 4)

            # base64 채우기
            try:
                if 'img' not in locals():  # 위에서 이미지 안 열었을 수도 있으니
                    img = Image.open(img_path).convert("RGB")
                img_b64 = _encode_image(img)
            except Exception:
                img_b64 = ""

            results.append({"style_id": item.get("style_id", ""), "style_image_base64": img_b64, "score": score})

        if not results:
            return {"status": "failed", "message": "추천 가능한 스타일 후보가 없습니다. 데이터 경로 또는 이미지/임베딩을 확인하세요."}

        results = sorted(results, key=lambda x: x["score"], reverse=True)[:3]
        return {"status": "success", "results": results}

    except Exception as e:
        return {"status": "failed", "message": str(e)}

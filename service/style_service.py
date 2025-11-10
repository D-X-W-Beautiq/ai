import io
import base64
import torch
import json
import os
import unicodedata
from PIL import Image
from model_manager.clip_manager import load_clip

_cached_dataset = None
_cached_json_dir = None


KOR_TO_ENG_KEYWORDS = {
    "사랑스러운": "lovable beauty",
    "청순": "innocent beauty",
    "핑크블러셔": "pink blush",
    "피치블러셔": "peach blush",
    "오렌지블러셔": "orange blush",
    "매트립(광택없는)": "matte lips",
    "핑크립": "pink lips",
    "오렌지립": "orange lips",
    "웜톤": "warm tone",
    "쿨톤": "cool tone",
    "투명피부": "clear skin",
    "매트피부": "matte skin",
    "물광피부": "dewy glass skin",
    "진한눈썹": "bold brows",
    "세미스모키": "semi-smoky eyes",
    "자연스러운눈썹": "natural brows",
}


def _kor_to_eng_keywords(keywords):
    eng_keywords = []
    for kw in keywords:
        kw = kw.strip()
        eng_kw = KOR_TO_ENG_KEYWORDS.get(kw, kw)
        eng_keywords.append(eng_kw)
    return eng_keywords


def _decode_image(base64_str: str) -> Image.Image:
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _encode_image(image: Image.Image) -> str:
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _load_dataset(json_dir: str):
    json_files = [
        "makeup_captions_mood_detailed.json",
        "makeup_captions_mood_final.json",
        "makeup_captions_tone_detailed.json",
        "makeup_captions_tone_final.json"
    ]

    dataset = []
    seen_style_ids = set()
    seen_image_paths = set()

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
                    image_name = item.get("image_name", "")
                    style_id = os.path.splitext(image_name)[0] if image_name else ""
                    if style_id in seen_style_ids or img_path in seen_image_paths:
                        continue
                    seen_style_ids.add(style_id)
                    seen_image_paths.add(img_path)
                    dataset.append({
                        "style_id": style_id,
                        "caption": text,
                        "image_path": img_path
                    })

            elif "request" in item:  # final JSON
                rel_path = unicodedata.normalize('NFC', item["request"]["이미지경로"])
                img_path = os.path.join(json_dir, rel_path)
                if os.path.exists(img_path):
                    text = item["response"].get("caption", "") or item["response"].get("prompt_en", "")
                    image_name = item.get("image_name", "") or os.path.basename(rel_path)
                    style_id = os.path.splitext(image_name)[0]
                    if style_id in seen_style_ids or img_path in seen_image_paths:
                        continue
                    seen_style_ids.add(style_id)
                    seen_image_paths.add(img_path)
                    dataset.append({
                        "style_id": style_id,
                        "caption": text,
                        "image_path": img_path
                    })
    return dataset


def get_dataset(json_dir: str):
    global _cached_dataset, _cached_json_dir
    if _cached_dataset is None or _cached_json_dir != json_dir:
        _cached_dataset = _load_dataset(json_dir)
        _cached_json_dir = json_dir
    return _cached_dataset


def run_inference(request: dict, json_dir: str) -> dict:
    """
    CLIP 원리 기반 스타일 추천 (이미지 ↔ 텍스트 유사도)
    request = {
        "source_image_base64": "string",
        "keywords": ["핑크립", "청순", ...]
    }
    """
    try:
        if "source_image_base64" not in request:
            raise ValueError("Missing key: source_image_base64")
        if "keywords" not in request or not isinstance(request["keywords"], list):
            raise ValueError("Missing or invalid key: keywords")

        # 1️⃣ 한국어 키워드를 영어로 변환
        eng_keywords = _kor_to_eng_keywords(request["keywords"])
        caption = "A style with " + ", ".join(eng_keywords) + "."

        # 2️⃣ 모델 로드
        model, processor, device = load_clip()

        # 3️⃣ 사용자 이미지 임베딩
        user_image = _decode_image(request["source_image_base64"])
        image_inputs = processor(images=user_image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # 4️⃣ 사용자 텍스트(키워드) 임베딩
        text_inputs = processor(text=[caption], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # 5️⃣ 데이터셋 로드
        dataset = get_dataset(json_dir)
        results = []

        # 6️⃣ 각 후보와 CLIP 유사도 계산
        for idx, item in enumerate(dataset):
            img_path = item["image_path"]
            if not os.path.exists(img_path):
                continue

            # 후보 caption → 텍스트 임베딩
            caption_text = item.get("caption", "").strip()
            if caption_text == "":
                continue

            text_inputs = processor(text=[caption_text], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                item_text_features = model.get_text_features(**text_inputs)
                item_text_features = item_text_features / item_text_features.norm(p=2, dim=-1, keepdim=True)

            # (A) 이미지 유사도: 사용자 이미지 ↔ 후보 이미지
            img_inputs = processor(images=Image.open(img_path).convert("RGB"), return_tensors="pt").to(device)
            with torch.no_grad():
                item_img_features = model.get_image_features(**img_inputs)
                item_img_features = item_img_features / item_img_features.norm(p=2, dim=-1, keepdim=True)

            score_img = torch.matmul(image_features, item_img_features.T).item()
            score_txt = torch.matmul(text_features, item_text_features.T).item()

            # CLIP 스타일: 두 score를 가중합
            score = round(0.5 * score_img + 0.5 * score_txt, 4)

            results.append({
                "style_id": item["style_id"],
                "image_path": img_path,
                "score": score
            })

        if not results:
            return {"status": "failed", "message": "추천 가능한 스타일 후보가 없습니다."}

        # 7️⃣ 정렬 및 상위 3개 반환
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:3]

        final_results = []
        for r in results:
            try:
                img = Image.open(r["image_path"]).convert("RGB")
                img_b64 = _encode_image(img)
            except Exception:
                img_b64 = ""
            final_results.append({
                "style_id": r["style_id"],
                "style_image_base64": img_b64
            })

        return {"status": "success", "results": final_results}

    except Exception as e:
        return {"status": "failed", "message": str(e)}
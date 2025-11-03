import io
import base64
import torch
import json
import os
import unicodedata
from PIL import Image
from model_manager.clip_manager import load_clip


# 글로벌 데이터셋 캐시
_cached_dataset = None
_cached_json_dir = None


def _decode_image(base64_str: str) -> Image.Image:
    """Base64 → PIL.Image 변환"""
    image_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _encode_image(image: Image.Image) -> str:
    """PIL.Image → Base64"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _load_dataset(json_dir: str):
    """스타일 후보 데이터셋 로드 (기존 style_dataset.json 대신 4개 JSON 파일 사용)"""
    json_files = [
        "makeup_captions_mood_detailed.json",
        "makeup_captions_mood_final.json",
        "makeup_captions_tone_detailed.json",
        "makeup_captions_tone_final.json"
    ]

    dataset = []
    seen_style_ids = set()  # 중복 체크용 (style_id)
    seen_image_paths = set()  # 중복 체크용 (image_path)

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
                    # image_name에서 확장자 제거
                    image_name = item.get("image_name", "")
                    style_id = os.path.splitext(image_name)[0] if image_name else ""

                    # 중복 체크: 이미 본 style_id 또는 image_path면 건너뛰기
                    if style_id and style_id in seen_style_ids:
                        continue
                    if img_path in seen_image_paths:
                        continue

                    if style_id:
                        seen_style_ids.add(style_id)
                    seen_image_paths.add(img_path)

                    dataset.append({
                        "style_id": style_id,
                        "style_image_base64": None,
                        "embedding": item.get("embedding", []),
                        "caption": text,
                        "image_path": img_path
                    })
            elif "request" in item:  # final JSON
                rel_path = unicodedata.normalize('NFC', item["request"]["이미지경로"])
                img_path = os.path.join(json_dir, rel_path)
                if os.path.exists(img_path):
                    text = item["response"].get("caption", "") or item["response"].get("prompt_en", "")
                    # image_name에서 확장자 제거 (final JSON에도 있다고 가정)
                    image_name = item.get("image_name", "")
                    if not image_name:  # image_name이 없으면 경로에서 추출
                        image_name = os.path.basename(rel_path)
                    style_id = os.path.splitext(image_name)[0]

                    # 중복 체크: 이미 본 style_id 또는 image_path면 건너뛰기
                    if style_id and style_id in seen_style_ids:
                        continue
                    if img_path in seen_image_paths:
                        continue

                    if style_id:
                        seen_style_ids.add(style_id)
                    seen_image_paths.add(img_path)

                    dataset.append({
                        "style_id": style_id,
                        "style_image_base64": None,
                        "embedding": item.get("embedding", []),
                        "caption": text,
                        "image_path": img_path
                    })

    return dataset


def get_dataset(json_dir: str):
    """
    데이터셋을 글로벌 캐시에서 가져오거나, 없으면 로드 후 캐시
    """
    global _cached_dataset, _cached_json_dir

    # 캐시된 데이터가 없거나 json_dir이 변경된 경우에만 로드
    if _cached_dataset is None or _cached_json_dir != json_dir:
        _cached_dataset = _load_dataset(json_dir)
        _cached_json_dir = json_dir

    return _cached_dataset


def run_inference(request: dict, json_dir: str) -> dict:
    """
    스타일 추천 추론
    request = {
        "source_image_base64": "string",
        "keywords": ["pink blush", "warm tone", ...]
    }
    """
    try:
        # 입력 검증
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

        # 텍스트 임베딩
        caption = "A style with " + ", ".join(request["keywords"]) + "."
        text_inputs = processor(text=[caption], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_emb = model.get_text_features(**text_inputs)
            text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)

        # 스타일 후보 로드 (캐시 활용)
        dataset = get_dataset(json_dir)
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

            # L2 정규화
            emb = emb / emb.norm(p=2, dim=-1, keepdim=True)

            # 유사도 계산
            score_img = torch.matmul(user_emb, emb.T).item()
            score_txt = torch.matmul(text_emb, text_emb.T).item()
            score = round(0.85 * score_img + 0.15 * score_txt, 4)

            # 유사도 계산 단계에서는 이미지 인코딩 안 함 (경로만 저장)
            results.append({
                "style_id": item.get("style_id", ""),
                "image_path": img_path,
                "score": score
            })

        if not results:
            return {"status": "failed", "message": "추천 가능한 스타일 후보가 없습니다."}

        # Top-3 추천 (중복 제거)
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        # style_id 기준으로 중복 제거하면서 Top-3 선택
        unique_results = []
        seen_ids = set()

        for r in results:
            sid = r.get("style_id", "")

            # 빈 문자열 체크
            if not sid or sid.strip() == "":
                continue

            if sid not in seen_ids:
                unique_results.append(r)
                seen_ids.add(sid)
                if len(unique_results) >= 3:
                    break

        # Top-3에 대해서만 이미지 base64 인코딩 (Lazy Loading)
        final_results = []
        for r in unique_results:
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
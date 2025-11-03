"""
임베딩 사전 계산 스크립트
"""
import json
import os
import torch
from PIL import Image
from model_manager.clip_manager import load_clip


def precompute_embeddings(json_dir: str):
    json_files = [
        "makeup_captions_mood_detailed.json",
        "makeup_captions_mood_final.json",
        "makeup_captions_tone_detailed.json",
        "makeup_captions_tone_final.json"
    ]

    print("모델 로딩 중...")
    model, processor, device = load_clip()
    print(f"완료 (device: {device})\n")

    for jf in json_files:
        json_path = os.path.join(json_dir, jf)
        if not os.path.exists(json_path):
            continue

        print(f"처리 중: {jf}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        count = 0
        for item in data:
            if "image_path" in item:
                img_path = os.path.join(json_dir, item["image_path"])
            elif "request" in item:
                rel_path = item["request"]["이미지경로"]
                img_path = os.path.join(json_dir, rel_path)
            else:
                continue

            if item.get("embedding") and len(item.get("embedding", [])) > 0:
                continue

            if not os.path.exists(img_path):
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                img_inputs = processor(images=img, return_tensors="pt").to(device)

                with torch.no_grad():
                    emb = model.get_image_features(**img_inputs)
                    emb = emb / emb.norm(p=2, dim=-1, keepdim=True)
                    emb_list = emb.cpu().squeeze(0).tolist()

                item["embedding"] = emb_list
                count += 1

            except Exception:
                continue

        if count > 0:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  → {count}개 임베딩 계산 완료\n")
        else:
            print(f"  → 이미 모두 계산됨\n")


if __name__ == "__main__":
    json_dir = "data/style-recommendation"
    precompute_embeddings(json_dir)

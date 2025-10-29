# -*- coding: utf-8 -*-
import base64, io
from PIL import Image

class Base64Error(ValueError):
    pass

def is_base64(s: str) -> bool:
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False

def b64_to_bytes(b64: str) -> bytes:
    try:
        return base64.b64decode(b64, validate=True)
    except Exception as e:
        raise Base64Error(f"Invalid base64: {e}")

def b64_to_image(b64: str) -> Image.Image:
    data = b64_to_bytes(b64)
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise Base64Error(f"Invalid image bytes: {e}")

def image_to_b64(img: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# test_v2_chain_strict.py
"""
AI Pipeline 연결 테스트 (스펙 검증 강화 + 체인 강제 v2, 정리본)
NIA → Feedback → Product → Style → Makeup → Customization

- 5/6단계는 반드시 선행 단계 성공 결과가 있어야만 진행
- 실패 시 즉시 중단(fail-fast)
- Style 결과가 없으면 Makeup 중단
- Makeup 실패 시 Customization 중단
- message가 None인 경우 안전 처리
- main() 가드로 중복 실행 방지
- ⏱ 스텝별 실행 시간 및 총 소요시간 기록
"""

import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

# ============================================================================
# 설정
# ============================================================================
# 프록시 없이 직접 붙으면 예: "http://127.0.0.1:8000"
# root-path("/proxy/8000")로 띄운 서버면 예: "http://127.0.0.1:8000/proxy/8000"
BASE_URL = "http://127.0.0.1:8000"

TIMEOUT_NIA = 60
TIMEOUT_FEEDBACK = 60
TIMEOUT_PRODUCT = 120
TIMEOUT_STYLE = 90
TIMEOUT_MAKEUP = 600
TIMEOUT_CUSTOM = 600

# TEST_IMAGE = Path("../test_data_512_padding/test3.png")
# TEST_IMAGE = Path("../test_data/test1.jpg")
TEST_IMAGE = Path("data/inference.jpg")

# 타임 측정 저장소
_TIMES: Dict[str, float] = {}

def _fmt(sec: float) -> str:
    # 보기 좋게 mm:ss.s 포맷
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:05.2f}s" if m else f"{s:0.2f}s"

def _tick(label: str) -> None:
    _TIMES[f"__start_{label}"] = time.perf_counter()

def _tock(label: str) -> float:
    start = _TIMES.get(f"__start_{label}")
    end = time.perf_counter()
    dt = (end - start) if start else 0.0
    _TIMES[label] = dt
    return dt

def _print_time(label: str) -> None:
    dt = _TIMES.get(label, 0.0)
    print(f"⏱ {label:>12}  { _fmt(dt) }")

def _print_time_summary(total_key: str = "TOTAL") -> None:
    print("\n" + "="*60)
    print("실행 시간 요약")
    print("="*60)
    ordered = ["NIA", "Feedback", "Product", "Style", "Makeup", "Customization"]
    for k in ordered:
        if k in _TIMES:
            _print_time(k)
    if total_key in _TIMES:
        print("-" * 60)
        _print_time(total_key)
    print()

# ============================================================================
# 검증 함수
# ============================================================================
def validate_nia_response(resp: Dict[str, Any]) -> List[str]:
    e = []
    if "status" not in resp:
        e.append("❌ 필수 필드 누락: status"); return e
    if resp["status"] == "success":
        if "predictions" not in resp:
            e.append("❌ 필수 필드 누락: predictions")
        else:
            pred = resp["predictions"]
            for k in ["moisture_reg","elasticity_reg","wrinkle_reg","pigmentation_reg","pore_reg"]:
                if k not in pred: e.append(f"❌ predictions.{k} 누락")
                elif not isinstance(pred[k], int): e.append(f"❌ predictions.{k} 타입오류:{type(pred[k]).__name__}")
                elif not (0 <= pred[k] <= 100): e.append(f"❌ predictions.{k} 범위오류:{pred[k]}")
    elif resp["status"] == "error" and "message" not in resp:
        e.append("❌ 필수 필드 누락: message")
    return e

def validate_feedback_response(resp: Dict[str, Any]) -> List[str]:
    e = []
    if "status" not in resp: e.append("❌ 필수 필드 누락: status"); return e
    if resp["status"] == "success":
        if "feedback" not in resp or not isinstance(resp["feedback"], str):
            e.append("❌ feedback 누락/타입 오류")
    elif resp["status"] == "error" and "message" not in resp:
        e.append("❌ 필수 필드 누락: message")
    return e

def validate_product_response(resp: Dict[str, Any]) -> List[str]:
    e = []
    if "status" not in resp: e.append("❌ 필수 필드 누락: status"); return e
    if resp["status"] == "success":
        recs = resp.get("recommendations")
        if not isinstance(recs, list): e.append("❌ recommendations 누락/타입 오류")
        else:
            for i, r in enumerate(recs):
                if not isinstance(r.get("product_id"), str): e.append(f"❌ rec[{i}].product_id 누락/타입 오류")
                if not isinstance(r.get("reason"), str): e.append(f"❌ rec[{i}].reason 누락/타입 오류")
    elif resp["status"] == "error":
        if "message" not in resp: e.append("❌ 필수 필드 누락: message")
        if "error_code" not in resp: e.append("❌ 필수 필드 누락: error_code")
    return e

def validate_style_response(resp: Dict[str, Any]) -> List[str]:
    e = []
    if "status" not in resp: e.append("❌ 필수 필드 누락: status"); return e
    if resp["status"] == "success":
        results = resp.get("results")
        if not isinstance(results, list): e.append("❌ results 누락/타입 오류")
        else:
            for i, r in enumerate(results):
                if not isinstance(r.get("style_id"), str): e.append(f"❌ results[{i}].style_id 누락/타입 오류")
                if not isinstance(r.get("style_image_base64"), str): e.append(f"❌ results[{i}].style_image_base64 누락/타입 오류")
    elif resp["status"] == "error" and "message" not in resp:
        e.append("❌ 필수 필드 누락: message")
    return e

def validate_makeup_response(resp: Dict[str, Any]) -> List[str]:
    e = []
    if "status" not in resp: e.append("❌ 필수 필드 누락: status"); return e
    if resp["status"] == "success":
        if not isinstance(resp.get("result_image_base64"), str): e.append("❌ result_image_base64 누락/타입 오류")
    elif resp["status"] == "error" and "message" not in resp:
        e.append("❌ 필수 필드 누락: message")
    return e

def validate_custom_response(resp: Dict[str, Any]) -> List[str]:
    e = []
    if "status" not in resp: e.append("❌ 필수 필드 누락: status"); return e
    if resp["status"] == "success":
        if not isinstance(resp.get("result_image_base64"), str): e.append("❌ result_image_base64 누락/타입 오류")
    elif resp["status"] == "error" and "message" not in resp:
        e.append("❌ 필수 필드 누락: message")
    return e

def print_validation_result(name: str, errors: List[str]):
    if not errors: print(f"  ✅ {name} 스펙 준수 완료")
    else:
        print(f"  🔍 {name} 검증 결과:")
        for err in errors: print(f"     {err}")

# ============================================================================
# 유틸
# ============================================================================
def load_image_base64(p: Path) -> str:
    if not p.exists(): raise FileNotFoundError(f"입력 이미지 없음: {p}")
    return base64.b64encode(p.read_bytes()).decode()

def print_response(step: str, resp: requests.Response):
    print("\n" + "="*60)
    print(f"[{step}] Status Code: {resp.status_code}")
    print("="*60)
    try:
        print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
    except Exception:
        print(resp.text)
    print()

def require_success(name: str, resp: Dict[str, Any]):
    if resp.get("status") != "success":
        # 실패 시점까지의 시간 요약 출력 후 종료
        _tock(name)  # 혹시 시작만 해둔 상태일 수 있으므로 마무리
        _print_time_summary()
        raise SystemExit(f"❌ {name} 실패 — 중단합니다. 상세: {resp.get('message','알 수 없는 에러')}")

# ============================================================================
# 단계 실행
# ============================================================================
def step1_nia(image_b64: str) -> Dict[str, Any]:
    print("\n" + "="*60); print("STEP 1: NIA - 피부 분석"); print("="*60)
    r = requests.post(f"{BASE_URL}/nia/analyze", json={"image_base64": image_b64}, timeout=TIMEOUT_NIA)
    print_response("NIA", r)
    data = r.json()
    print_validation_result("NIA", validate_nia_response(data))
    require_success("NIA", data)
    preds = data["predictions"]
    print("피부 분석 완료!")
    print("  - 수분:{moisture_reg}  탄력:{elasticity_reg}  주름:{wrinkle_reg}  색소:{pigmentation_reg}  모공:{pore_reg}".format(**preds))
    pf = Path("data/predictions.json")
    print(f"  {'✅' if pf.exists() else '⚠️'} 결과 파일: {pf}")
    return preds

def step2_feedback(predictions_path: Path):
    print("\n" + "="*60); print("STEP 2: Feedback - 피부 피드백 생성"); print("="*60)
    r = requests.post(f"{BASE_URL}/feedback/generate",
                      json={"predictions_json_path": str(predictions_path)},
                      timeout=TIMEOUT_FEEDBACK)
    print_response("Feedback", r)
    data = r.json()
    print_validation_result("Feedback", validate_feedback_response(data))
    require_success("Feedback", data)
    fb = data["feedback"]
    print("피드백 생성 완료!")
    print("  " + (fb[:200] + "..." if len(fb) > 200 else fb))

def step3_product(preds: Dict[str, Any]):
    print("\n" + "="*60); print("STEP 3: Product - 제품 추천 이유 생성"); print("="*60)
    payload = {
        "skin_analysis": preds,
        "recommended_categories": ["moisture", "elasticity"],
        "filtered_products": [
            {"product_id":"SKU123","product_name":"Hydra Boost Serum","brand":"BrandA",
             "category":"moisture","price":32000,"review_score":4.5,"review_count":320,
             "ingredients":["히알루론산","글리세린","판테놀"]},
            {"product_id":"SKU456","product_name":"Firming Peptide Cream","brand":"BrandB",
             "category":"elasticity","price":42000,"review_score":4.3,"review_count":210,
             "ingredients":["펩타이드","세라마이드","나이아신아마이드"]}
        ],
        "locale":"ko-KR"
    }
    r = requests.post(f"{BASE_URL}/product/reason", json=payload, timeout=TIMEOUT_PRODUCT)
    print_response("Product", r)
    data = r.json()
    print_validation_result("Product", validate_product_response(data))
    require_success("Product", data)
    print(f"제품 추천 완료! ({len(data['recommendations'])}개)")
    for i, rec in enumerate(data["recommendations"], 1):
        reason = rec["reason"]
        print(f"  [{i}] {rec['product_id']}: {reason[:100] + '...' if len(reason)>100 else reason}")

def step4_style(image_b64: str) -> str:
    print("\n" + "="*60); print("STEP 4: Style - 스타일 추천"); print("="*60)
    payload = {"source_image_base64": image_b64, "keywords": ["natural","pink blush","soft"]}
    r = requests.post(f"{BASE_URL}/style/recommend", json=payload, timeout=TIMEOUT_STYLE)
    print_response("Style", r)
    data = r.json()
    print_validation_result("Style", validate_style_response(data))
    require_success("Style", data)
    results = data.get("results", [])
    if not results:
        _print_time_summary()  # 진행된 시점까지 요약
        raise SystemExit("❌ Style 결과가 비어 있어 Makeup을 진행할 수 없습니다.")
    sid = results[0].get("style_id","")
    print(f"스타일 추천 완료! (Top-1 사용, style_id: {sid})")
    return results[0]["style_image_base64"]

def step5_makeup(src_b64: str, style_b64: str) -> str:
    print("\n" + "="*60); print("STEP 5: Makeup - 메이크업 시뮬레이션"); print("="*60)
    payload = {"source_image_base64": src_b64, "style_image_base64": style_b64}
    try:
        print(f"⏳ Makeup API 호출 중 (≤ {TIMEOUT_MAKEUP}s)...")
        r = requests.post(f"{BASE_URL}/makeup/simulate", json=payload, timeout=TIMEOUT_MAKEUP)
        print_response("Makeup", r)
        data = r.json()
        print_validation_result("Makeup", validate_makeup_response(data))
        require_success("Makeup", data)

        b64 = data["result_image_base64"]

        # 서버 저장 경로(선택)
        message = data.get("message")
        if isinstance(message, str) and "saved:" in message:
            saved_path = message.split("saved:", 1)[1].strip()
            print(f"  ↳ 서버 저장 경로 보고: {saved_path}")

        out = Path("data/output/makeup_result.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(b64))
        print(f"  ✅ 로컬 저장: {out}")
        return b64
    except requests.exceptions.Timeout:
        _print_time_summary()
        raise SystemExit(f"❌ Makeup API 타임아웃({TIMEOUT_MAKEUP}s). steps 축소/서버 GPU 확인/타임아웃 상향 필요.")
    except requests.exceptions.ConnectionError as e:
        _print_time_summary()
        raise SystemExit(f"❌ Makeup API 연결 오류: {e}")

def step6_custom(makeup_b64: str):
    print("\n" + "="*60); print("STEP 6: Customization - 메이크업 커스터마이징"); print("="*60)
    payload = {"base_image_base64": makeup_b64, "edits":[{"region":"lip","intensity":70},{"region":"blush","intensity":60}]}
    try:
        print(f"⏳ Customization API 호출 중 (≤ {TIMEOUT_CUSTOM}s)...")
        r = requests.post(f"{BASE_URL}/custom/apply", json=payload, timeout=TIMEOUT_CUSTOM)
        print_response("Customization", r)
        data = r.json()
        print_validation_result("Customization", validate_custom_response(data))
        require_success("Customization", data)

        b64 = data["result_image_base64"]
        message = data.get("message")
        if isinstance(message, str) and "saved:" in message:
            saved_path = message.split("saved:", 1)[1].strip()
            print(f"  ↳ 서버 저장 경로 보고: {saved_path}")

        out = Path("data/output/final_result.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base64.b64decode(b64))
        print(f"  ✅ 최종 결과 저장: {out}")
    except requests.exceptions.Timeout:
        _print_time_summary()
        raise SystemExit(f"❌ Customization API 타임아웃({TIMEOUT_CUSTOM}s). 서버 분할추론/세그 가속 확인 필요.")
    except requests.exceptions.ConnectionError as e:
        _print_time_summary()
        raise SystemExit(f"❌ Customization API 연결 오류: {e}")

# ============================================================================
# 메인
# ============================================================================
def main():
    # 총 시간 시작
    _tick("TOTAL")

    # 1) NIA
    src_b64 = load_image_base64(TEST_IMAGE)
    _tick("NIA")
    preds = step1_nia(src_b64)
    _tock("NIA")

    # 2) Feedback
    _tick("Feedback")
    step2_feedback(Path("data/predictions.json"))
    _tock("Feedback")

    # 3) Product
    _tick("Product")
    step3_product(preds)
    _tock("Product")

    # 4) Style
    _tick("Style")
    style_b64 = step4_style(src_b64)
    _tock("Style")

    # 교대 로딩 충돌 완화(대기시간은 총 소요시간에 포함)
    print("\n⏳ GPU 메모리 정리 대기 (10초)...")
    time.sleep(10)

    # 5) Makeup
    _tick("Makeup")
    makeup_b64 = step5_makeup(src_b64, style_b64)
    _tock("Makeup")

    # 6) Customization
    _tick("Customization")
    step6_custom(makeup_b64)
    _tock("Customization")

    # 총 시간 종료
    _tock("TOTAL")

    print("\n" + "="*60)
    print("전체 Pipeline 테스트 완료 (모든 필수 체인 통과)!")
    print("="*60)
    print("생성/확인 파일:")
    print("  - data/predictions.json (NIA)")
    print("  - data/output/makeup_result.png (Makeup)")
    print("  - data/output/final_result.png (Customization)")

    # 시간 요약 출력
    _print_time_summary()

if __name__ == "__main__":
    main()

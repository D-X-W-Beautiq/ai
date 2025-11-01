"""
AI Pipeline 연결 테스트 (스펙 검증 강화 버전)
NIA → Feedback → Product → Style → Makeup → Customization

✅ 추가된 기능:
- 모든 API 타임아웃 설정
- 필드 누락/추가/타입 검증
- 파일 저장 확인
- 상세한 에러 핸들링
"""

import requests
import json
import base64
import time
from pathlib import Path
from typing import Any, Dict, List

# ============================================================================
# 설정
# ============================================================================
BASE_URL = "http://127.0.0.1:8000"

# 타임아웃 설정 (초 단위) - 충분히 길게 설정하여 실제 에러 확인
TIMEOUT_NIA = 60
TIMEOUT_FEEDBACK = 60
TIMEOUT_PRODUCT = 120
TIMEOUT_STYLE = 90
TIMEOUT_MAKEUP = 600  # 10분 (Stable Diffusion - 실제 에러 확인용)
TIMEOUT_CUSTOM = 600  # 10분 (SegFormer - 실제 에러 확인용)


# ============================================================================
# 검증 함수들
# ============================================================================

def validate_nia_response(response: Dict[str, Any]) -> List[str]:
    """NIA API 응답 검증"""
    errors = []

    if "status" not in response:
        errors.append("❌ 필수 필드 누락: status")
        return errors

    if response["status"] == "success":
        if "predictions" not in response:
            errors.append("❌ 필수 필드 누락: predictions")
        else:
            pred = response["predictions"]
            required_fields = ["moisture_reg", "elasticity_reg", "wrinkle_reg",
                              "pigmentation_reg", "pore_reg"]
            for field in required_fields:
                if field not in pred:
                    errors.append(f"❌ predictions.{field} 누락")
                elif not isinstance(pred[field], int):
                    errors.append(f"❌ predictions.{field} 타입 오류: {type(pred[field]).__name__} (expected: int)")
                elif not (0 <= pred[field] <= 100):
                    errors.append(f"❌ predictions.{field} 범위 오류: {pred[field]} (expected: 0-100)")

        if "message" in response and response["message"] is not None:
            errors.append(f"⚠️  스펙 외 필드: message = '{response['message']}' (무시 가능)")

    elif response["status"] == "error":
        if "message" not in response:
            errors.append("❌ 필수 필드 누락: message")

    return errors


def validate_feedback_response(response: Dict[str, Any]) -> List[str]:
    """Feedback API 응답 검증"""
    errors = []

    if "status" not in response:
        errors.append("❌ 필수 필드 누락: status")
        return errors

    if response["status"] == "success":
        if "feedback" not in response:
            errors.append("❌ 필수 필드 누락: feedback")
        elif response["feedback"] is None:
            errors.append("❌ feedback가 null")
        elif not isinstance(response["feedback"], str):
            errors.append(f"❌ feedback 타입 오류: {type(response['feedback']).__name__} (expected: str)")

        if "message" in response and response["message"] is not None:
            errors.append(f"⚠️  스펙 외 필드: message (무시 가능)")

    elif response["status"] == "error":
        if "message" not in response:
            errors.append("❌ 필수 필드 누락: message")

    return errors


def validate_product_response(response: Dict[str, Any]) -> List[str]:
    """Product API 응답 검증"""
    errors = []

    if "status" not in response:
        errors.append("❌ 필수 필드 누락: status")
        return errors

    if response["status"] == "success":
        if "recommendations" not in response:
            errors.append("❌ 필수 필드 누락: recommendations")
        elif response["recommendations"] is None:
            errors.append("❌ recommendations가 null")
        elif not isinstance(response["recommendations"], list):
            errors.append(f"❌ recommendations 타입 오류: {type(response['recommendations']).__name__} (expected: list)")
        else:
            for i, rec in enumerate(response["recommendations"]):
                if "product_id" not in rec:
                    errors.append(f"❌ recommendations[{i}].product_id 누락")
                elif not isinstance(rec["product_id"], str):
                    errors.append(f"❌ recommendations[{i}].product_id 타입 오류: {type(rec['product_id']).__name__}")

                if "reason" not in rec:
                    errors.append(f"❌ recommendations[{i}].reason 누락")
                elif not isinstance(rec["reason"], str):
                    errors.append(f"❌ recommendations[{i}].reason 타입 오류: {type(rec['reason']).__name__}")

        if "message" in response and response["message"] is not None:
            errors.append(f"⚠️  스펙 외 필드: message (무시 가능)")
        if "error_code" in response and response["error_code"] is not None:
            errors.append(f"⚠️  스펙 외 필드: error_code (무시 가능)")

    elif response["status"] == "error":
        if "message" not in response:
            errors.append("❌ 필수 필드 누락: message")
        if "error_code" not in response:
            errors.append("❌ 필수 필드 누락: error_code")

    return errors


def validate_style_response(response: Dict[str, Any]) -> List[str]:
    """Style API 응답 검증"""
    errors = []

    if "status" not in response:
        errors.append("❌ 필수 필드 누락: status")
        return errors

    if response["status"] == "success":
        if "results" not in response:
            errors.append("❌ 필수 필드 누락: results")
        elif response["results"] is None:
            errors.append("❌ results가 null")
        elif not isinstance(response["results"], list):
            errors.append(f"❌ results 타입 오류: {type(response['results']).__name__} (expected: list)")
        else:
            if len(response["results"]) == 0:
                errors.append("⚠️  results 배열이 비어있음 (Top-3 예상)")

            for i, res in enumerate(response["results"]):
                if "style_id" not in res:
                    errors.append(f"❌ results[{i}].style_id 누락")
                elif not isinstance(res["style_id"], str):
                    errors.append(f"❌ results[{i}].style_id 타입 오류: {type(res['style_id']).__name__}")
                elif res["style_id"] == "":
                    errors.append(f"⚠️  results[{i}].style_id가 빈 문자열")

                if "style_image_base64" not in res:
                    errors.append(f"❌ results[{i}].style_image_base64 누락")
                elif not isinstance(res["style_image_base64"], str):
                    errors.append(f"❌ results[{i}].style_image_base64 타입 오류: {type(res['style_image_base64']).__name__}")

                # 스펙에 없는 필드 체크
                if "score" in res:
                    errors.append(f"⚠️  results[{i}]에 스펙 외 필드: score = {res['score']}")

        if "message" in response and response["message"] is not None:
            errors.append(f"⚠️  스펙 외 필드: message (무시 가능)")

    elif response["status"] == "error":
        if "message" not in response:
            errors.append("❌ 필수 필드 누락: message")

    return errors


def validate_makeup_response(response: Dict[str, Any]) -> List[str]:
    """Makeup API 응답 검증"""
    errors = []

    if "status" not in response:
        errors.append("❌ 필수 필드 누락: status")
        return errors

    if response["status"] == "success":
        if "result_image_base64" not in response:
            errors.append("❌ 필수 필드 누락: result_image_base64")
        elif response["result_image_base64"] is None:
            errors.append("❌ result_image_base64가 null")
        elif not isinstance(response["result_image_base64"], str):
            errors.append(f"❌ result_image_base64 타입 오류: {type(response['result_image_base64']).__name__}")

        if "message" in response and response["message"] is not None:
            # message는 저장 경로 정보이므로 유용함 (스펙 외이지만 경고 안 함)
            pass

    elif response["status"] == "error":
        if "message" not in response:
            errors.append("❌ 필수 필드 누락: message")

    return errors


def validate_customization_response(response: Dict[str, Any]) -> List[str]:
    """Customization API 응답 검증"""
    errors = []

    if "status" not in response:
        errors.append("❌ 필수 필드 누락: status")
        return errors

    if response["status"] == "success":
        if "result_image_base64" not in response:
            errors.append("❌ 필수 필드 누락: result_image_base64")
        elif response["result_image_base64"] is None:
            errors.append("❌ result_image_base64가 null")
        elif not isinstance(response["result_image_base64"], str):
            errors.append(f"❌ result_image_base64 타입 오류: {type(response['result_image_base64']).__name__}")

    elif response["status"] == "error":
        if "message" not in response:
            errors.append("❌ 필수 필드 누락: message")

    return errors


def print_validation_result(api_name: str, errors: List[str]):
    """검증 결과 출력"""
    if not errors:
        print(f"  ✅ {api_name} 스펙 준수 완료")
    else:
        print(f"  🔍 {api_name} 검증 결과:")
        for error in errors:
            print(f"     {error}")


# ============================================================================
# 유틸 함수
# ============================================================================

def load_image_base64(image_path):
    """이미지를 Base64로 인코딩"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def print_response(step, response):
    """응답 출력"""
    print(f"\n{'='*60}")
    print(f"[{step}] Status Code: {response.status_code}")
    print(f"{'='*60}")
    try:
        data = response.json()
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except:
        print(response.text)
    print()


# ============================================================================
# STEP 1: NIA - 피부 분석
# ============================================================================
print("\n" + "="*60)
print("STEP 1: NIA - 피부 분석")
print("="*60)

test_image = "data/inference.jpg"
image_b64 = load_image_base64(test_image)

nia_request = {
    "image_base64": image_b64
}

response_nia = requests.post(
    f"{BASE_URL}/nia/analyze",
    json=nia_request,
    timeout=TIMEOUT_NIA
)
print_response("NIA", response_nia)

nia_result = response_nia.json()

# 스펙 검증
validation_errors = validate_nia_response(nia_result)
print_validation_result("NIA", validation_errors)

if nia_result.get("status") == "success":
    predictions = nia_result["predictions"]
    print(f"피부 분석 완료!")
    print(f"  - 수분: {predictions['moisture_reg']}")
    print(f"  - 탄력: {predictions['elasticity_reg']}")
    print(f"  - 주름: {predictions['wrinkle_reg']}")
    print(f"  - 색소: {predictions['pigmentation_reg']}")
    print(f"  - 모공: {predictions['pore_reg']}")

    # 파일 저장 확인
    predictions_file = Path("data/predictions.json")
    if predictions_file.exists():
        print(f"  ✅ 결과 파일 저장됨: {predictions_file}")
    else:
        print(f"  ⚠️  결과 파일 저장 실패: {predictions_file}")
else:
    print("NIA 실패!")
    exit(1)


# ============================================================================
# STEP 2: Feedback - 피부 피드백 생성
# ============================================================================
print("\n" + "="*60)
print("STEP 2: Feedback - 피부 피드백 생성")
print("="*60)

feedback_request = {
    "predictions_json_path": "data/predictions.json"
}

response_feedback = requests.post(
    f"{BASE_URL}/feedback/generate",
    json=feedback_request,
    timeout=TIMEOUT_FEEDBACK
)
print_response("Feedback", response_feedback)

feedback_result = response_feedback.json()

# 스펙 검증
validation_errors = validate_feedback_response(feedback_result)
print_validation_result("Feedback", validation_errors)

if feedback_result.get("status") == "success":
    print("피드백 생성 완료!")
    feedback_text = feedback_result['feedback']
    print(f"  {feedback_text[:200]}..." if len(feedback_text) > 200 else f"  {feedback_text}")
else:
    print("Feedback 실패!")


# ============================================================================
# STEP 3: Product - 제품 추천 이유 생성
# ============================================================================
print("\n" + "="*60)
print("STEP 3: Product - 제품 추천 이유 생성")
print("="*60)

product_request = {
    "skin_analysis": predictions,
    "recommended_categories": ["moisture", "elasticity"],
    "filtered_products": [
        {
            "product_id": "SKU123",
            "product_name": "Hydra Boost Serum",
            "brand": "BrandA",
            "category": "moisture",
            "price": 32000,
            "review_score": 4.5,
            "review_count": 320,
            "ingredients": ["히알루론산", "글리세린", "판테놀"]
        },
        {
            "product_id": "SKU456",
            "product_name": "Firming Peptide Cream",
            "brand": "BrandB",
            "category": "elasticity",
            "price": 42000,
            "review_score": 4.3,
            "review_count": 210,
            "ingredients": ["펩타이드", "세라마이드", "나이아신아마이드"]
        }
    ],
    "locale": "ko-KR"
}

response_product = requests.post(
    f"{BASE_URL}/product/reason",
    json=product_request,
    timeout=TIMEOUT_PRODUCT
)
print_response("Product", response_product)

product_result = response_product.json()

# 스펙 검증
validation_errors = validate_product_response(product_result)
print_validation_result("Product", validation_errors)

if product_result.get("status") == "success":
    print(f"제품 추천 완료! ({len(product_result['recommendations'])}개)")
    for i, rec in enumerate(product_result["recommendations"]):
        reason_preview = rec['reason'][:100] + "..." if len(rec['reason']) > 100 else rec['reason']
        print(f"  [{i+1}] {rec['product_id']}: {reason_preview}")
else:
    print("Product 실패!")


# ============================================================================
# STEP 4: Style - 스타일 추천
# ============================================================================
print("\n" + "="*60)
print("STEP 4: Style - 스타일 추천")
print("="*60)

style_request = {
    "source_image_base64": image_b64,
    "keywords": ["natural", "pink blush", "soft"]
}

response_style = requests.post(
    f"{BASE_URL}/style/recommend",
    json=style_request,
    timeout=TIMEOUT_STYLE
)
print_response("Style", response_style)

style_result = response_style.json()

# 스펙 검증
validation_errors = validate_style_response(style_result)
print_validation_result("Style", validation_errors)

if style_result.get("status") == "success":
    results = style_result.get("results", [])
    if not results:
        print("⚠️  스타일 추천 결과가 비어있습니다!")
        style_image_b64 = image_b64
    else:
        print(f"스타일 추천 완료! ({len(results)}개)")
        for i, res in enumerate(results):
            style_id = res.get("style_id", "")
            if not style_id:
                print(f"  [{i+1}] ⚠️  style_id 누락 또는 빈 문자열")
            else:
                print(f"  [{i+1}] style_id: {style_id}")

        # style_image_base64 확인
        if results[0].get("style_image_base64"):
            style_image_b64 = results[0]["style_image_base64"]
            print(f"  ✅ 첫 번째 스타일 이미지 사용")
        else:
            print("  ⚠️  style_image_base64가 없어서 원본 이미지를 사용합니다.")
            style_image_b64 = image_b64
else:
    print("Style 실패!")
    style_image_b64 = image_b64

# GPU 메모리 정리 대기 (Style → Makeup 전환 시)
print("\n⏳ GPU 메모리 정리 대기 중 (10초)...")
print("   (Style API의 CLIP 모델과 Makeup API의 Stable Diffusion 동시 로딩 방지)")
time.sleep(10)

# ============================================================================
# STEP 5: Makeup - 메이크업 시뮬레이션
# ============================================================================
print("\n" + "="*60)
print("STEP 5: Makeup - 메이크업 시뮬레이션")
print("="*60)

makeup_request = {
    "source_image_base64": image_b64,
    "style_image_base64": style_image_b64
}

try:
    print(f"⏳ Makeup API 호출 중 (최대 {TIMEOUT_MAKEUP}초 대기)...")
    response_makeup = requests.post(
        f"{BASE_URL}/makeup/simulate",
        json=makeup_request,
        timeout=TIMEOUT_MAKEUP
    )
    print_response("Makeup", response_makeup)

    makeup_result = response_makeup.json()

    # 스펙 검증
    validation_errors = validate_makeup_response(makeup_result)
    print_validation_result("Makeup", validation_errors)

    if makeup_result.get("status") == "success":
        print("✅ 메이크업 시뮬레이션 완료!")
        makeup_result_b64 = makeup_result["result_image_base64"]

        # 저장 경로 확인
        message = makeup_result.get("message", "")
        if "saved:" in message:
            saved_path = message.split("saved:")[1].strip()
            if Path(saved_path).exists():
                print(f"  ✅ 파일 저장됨: {saved_path}")
            else:
                print(f"  ⚠️  파일 없음: {saved_path}")

        # 결과 이미지를 로컬에도 저장
        result_path = Path("data/output/makeup_result.png")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_bytes(base64.b64decode(makeup_result_b64))
        print(f"  ✅ 로컬 저장: {result_path}")
    else:
        print("❌ Makeup API 서버 에러 발생!")
        error_msg = makeup_result.get("message", "알 수 없는 에러")
        print(f"   에러 메시지: {error_msg}")
        print(f"\n   → 서버 터미널에서 상세 스택트레이스를 확인하세요")
        makeup_result_b64 = image_b64

except requests.exceptions.Timeout:
    print(f"❌ Makeup API 타임아웃 ({TIMEOUT_MAKEUP}초 초과)")
    print("   Stable Diffusion 추론 시간이 너무 오래 걸립니다.")
    print("   해결 방법:")
    print("   1. 서버 GPU 사용 확인")
    print("   2. num_inference_steps 줄이기 (30 → 20)")
    print("   3. 타임아웃 늘리기")
    makeup_result = {"status": "error", "message": "Timeout"}
    makeup_result_b64 = image_b64

except requests.exceptions.ConnectionError as e:
    print(f"❌ Makeup API 연결 오류: {e}")
    print("   서버가 응답 중 연결을 끊었습니다.")
    makeup_result = {"status": "error", "message": str(e)}
    makeup_result_b64 = image_b64

except Exception as e:
    print(f"❌ Makeup API 예상치 못한 오류: {e}")
    makeup_result = {"status": "error", "message": str(e)}
    makeup_result_b64 = image_b64


# ============================================================================
# STEP 6: Customization - 메이크업 커스터마이징
# ============================================================================
print("\n" + "="*60)
print("STEP 6: Customization - 메이크업 커스터마이징")
print("="*60)

custom_request = {
    "base_image_base64": makeup_result_b64,
    "edits": [
        {"region": "lip", "intensity": 70},
        {"region": "blush", "intensity": 60}
    ]
}

try:
    print(f"⏳ Customization API 호출 중 (최대 {TIMEOUT_CUSTOM}초 대기)...")
    response_custom = requests.post(
        f"{BASE_URL}/custom/apply",
        json=custom_request,
        timeout=TIMEOUT_CUSTOM
    )
    print_response("Customization", response_custom)

    custom_result = response_custom.json()

    # 스펙 검증
    validation_errors = validate_customization_response(custom_result)
    print_validation_result("Customization", validation_errors)

    if custom_result.get("status") == "success":
        print("✅ 커스터마이징 완료!")
        custom_result_b64 = custom_result["result_image_base64"]

        # 저장 경로 확인
        message = custom_result.get("message", "")
        if "saved:" in message:
            saved_path = message.split("saved:")[1].strip()
            if Path(saved_path).exists():
                print(f"  ✅ 파일 저장됨: {saved_path}")
            else:
                print(f"  ⚠️  파일 없음: {saved_path}")

        # 최종 결과 저장
        final_path = Path("data/output/final_result.png")
        final_path.write_bytes(base64.b64decode(custom_result_b64))
        print(f"  ✅ 최종 결과 저장: {final_path}")
    else:
        print("❌ Customization API 서버 에러 발생!")
        error_msg = custom_result.get("message", "알 수 없는 에러")
        print(f"   에러 메시지: {error_msg}")
        print(f"\n   → 서버 터미널에서 상세 스택트레이스를 확인하세요")

except requests.exceptions.Timeout:
    print(f"❌ Customization API 타임아웃 ({TIMEOUT_CUSTOM}초 = {TIMEOUT_CUSTOM//60}분 초과)")
    print("   실제로 이렇게 오래 걸리면 서버에 문제가 있습니다.")
    print("   → 서버 터미널 로그를 확인하세요!")
    custom_result = {"status": "error", "message": "Timeout"}

except requests.exceptions.ConnectionError as e:
    print(f"❌ Customization API 연결 오류: {e}")
    print("   서버가 응답 중 연결을 끊었습니다.")
    custom_result = {"status": "error", "message": str(e)}

except Exception as e:
    print(f"❌ Customization API 예상치 못한 오류: {e}")
    custom_result = {"status": "error", "message": str(e)}


# ============================================================================
# 전체 요약
# ============================================================================
print("\n" + "="*60)
print("전체 Pipeline 테스트 완료!")
print("="*60)

results = {
    "NIA": "성공" if nia_result.get("status") == "success" else "실패",
    "Feedback": "성공" if feedback_result.get("status") == "success" else "실패",
    "Product": "성공" if product_result.get("status") == "success" else "실패",
    "Style": "성공" if style_result.get("status") == "success" else "실패",
    "Makeup": "성공" if makeup_result.get("status") == "success" else "실패",
    "Customization": "성공" if custom_result.get("status") == "success" else "실패",
}

for step, status in results.items():
    emoji = "✅" if status == "성공" else "❌"
    print(f"  {emoji} {step}: {status}")

all_success = all(status == "성공" for status in results.values())
if all_success:
    print("\n🎉 모든 API가 정상 작동합니다!")
else:
    print("\n⚠️  일부 API에 문제가 있습니다. 위 로그를 확인하세요.")

print(f"\n📁 생성된 파일:")
print(f"  - data/predictions.json (NIA 결과)")
print(f"  - data/output/makeup_*.png (서버 저장)")
print(f"  - data/output/makeup_result.png (로컬 저장)")
print(f"  - data/output/custom_*.png (서버 저장)")
print(f"  - data/output/final_result.png (최종 결과)")

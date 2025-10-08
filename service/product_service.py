import google.generativeai as genai
import json
import os
from pathlib import Path

def _get_model():
    """Gemini 모델 인스턴스 반환 (런타임 검증)"""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash-exp")

def run_inference(request: dict) -> dict:
    try:
        # 입력 검증
        if "skin_analysis" not in request:
            raise ValueError("Missing required field: skin_analysis")
        if "recommended_categories" not in request:
            raise ValueError("Missing required field: recommended_categories")
        if "filtered_products" not in request:
            raise ValueError("Missing required field: filtered_products")

        skin_analysis = request["skin_analysis"]
        recommended_categories = request["recommended_categories"]
        filtered_products = request["filtered_products"]
        locale = request.get("locale", "ko-KR")

        recommendations = []

        # 모델 가져오기
        model = _get_model()

        # 각 제품별로 추천 이유 생성
        for product in filtered_products:
            try:
                # 프롬프트 생성
                prompt = generate_recommendation_prompt(
                    skin_analysis,
                    recommended_categories,
                    product,
                    locale
                )

                # Gemini API 호출 (재시도 로직 포함)
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = model.generate_content(
                            prompt,
                            generation_config={
                                "temperature": 0.7,
                                "max_output_tokens": 500,
                            },
                            request_options={"timeout": 30}
                        )

                        # 응답 검증
                        if not response or not response.text:
                            raise ValueError("Empty response from Gemini API")

                        reason = response.text.strip()
                        break

                    except Exception as api_error:
                        if attempt == max_retries - 1:
                            raise api_error
                        continue

                recommendations.append({
                    "product_id": product["product_id"],
                    "reason": reason
                })

            except Exception as product_error:
                # 개별 제품 처리 실패 시 기본 메시지
                recommendations.append({
                    "product_id": product["product_id"],
                    "reason": f"추천 이유 생성 중 오류가 발생했습니다: {str(product_error)}"
                })

        return {
            "status": "success",
            "recommendations": recommendations
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "error_code": "LLM_GENERATION_FAILED"
        }

def generate_recommendation_prompt(skin_analysis, recommended_categories, product, locale):
    """추천 이유 생성을 위한 프롬프트 작성"""

    # 카테고리 매핑
    category_map = {
        "moisture": "수분",
        "elasticity": "탄력",
        "wrinkle": "주름",
        "pigmentation": "색소침착",
        "pore": "모공"
    }

    # 카테고리별 기준값 (BE와 동일)
    thresholds = {
        'moisture': 65,
        'pigmentation': 70,
        'elasticity': 60,
        'wrinkle': 50,
        'pore': 55
    }

    # 부족한 영역 정보 추출
    concern_details = []

    for category in recommended_categories:
        korean_name = category_map.get(category, category)
        threshold = thresholds.get(category, 70)

        # BE와 동일한 로직으로 대표 점수 계산
        if category == "moisture":
            score = min(
                skin_analysis.get("dryness", 100),
                skin_analysis.get("moisture_reg", 100)
            )
            concern_details.append(f"{korean_name} (대표점수: {score}/100, 기준: {threshold}점 미만)")

        elif category == "elasticity":
            score = min(
                skin_analysis.get("sagging", 100),
                skin_analysis.get("elasticity_reg", 100)
            )
            concern_details.append(f"{korean_name} (대표점수: {score}/100, 기준: {threshold}점 미만)")

        elif category == "wrinkle":
            score = min(
                skin_analysis.get("wrinkle", 100),
                skin_analysis.get("wrinkle_reg", 100)
            )
            concern_details.append(f"{korean_name} (대표점수: {score}/100, 기준: {threshold}점 미만)")

        elif category == "pigmentation":
            score = min(
                skin_analysis.get("pigmentation", 100),
                skin_analysis.get("pigmentation_reg", 100)
            )
            concern_details.append(f"{korean_name} (대표점수: {score}/100, 기준: {threshold}점 미만)")

        elif category == "pore":
            score = min(
                skin_analysis.get("pore", 100),
                skin_analysis.get("pore_reg", 100)
            )
            concern_details.append(f"{korean_name} (대표점수: {score}/100, 기준: {threshold}점 미만)")

    # 프롬프트 작성
    if locale == "ko-KR":
        prompt = f"""
당신은 피부 전문가입니다. 아래 피부 분석 결과를 바탕으로 제품 추천 이유를 작성해주세요.

[피부 고민]
{', '.join(concern_details)}

[추천 제품]
- 브랜드: {product['brand']}
- 제품명: {product['product_name']}
- 카테고리: {category_map.get(product['category'], product['category'])}
- 가격: {product['price']:,}원
- 리뷰 점수: {product['review_score']}점
- 리뷰 수: {product['review_count']:,}건
- 주요 성분: {', '.join(product.get('ingredients', [])) if product.get('ingredients') else '정보 없음'}

요구사항:
1. 피부 분석 결과를 바탕으로 왜 이 제품이 적합한지 구체적으로 설명
2. 주요 성분의 효능을 간단히 언급
3. 리뷰 점수와 리뷰 수를 활용해 신뢰성 강조
4. 친근하고 전문적인 톤으로 2-3문장으로 작성
5. 불필요한 인사말이나 마무리 멘트 없이 본문만 작성

추천 이유:
"""
    else:
        prompt = f"""
You are a skin care expert. Write a product recommendation reason based on the skin analysis results.

[Skin Concerns]
{', '.join(concern_details)}

[Recommended Product]
- Brand: {product['brand']}
- Product Name: {product['product_name']}
- Category: {product['category']}
- Price: {product['price']:,} KRW
- Review Score: {product['review_score']}
- Review Count: {product['review_count']:,}
- Key Ingredients: {', '.join(product.get('ingredients', [])) if product.get('ingredients') else 'Not available'}

Requirements:
1. Explain specifically why this product is suitable based on skin analysis
2. Briefly mention the efficacy of key ingredients
3. Emphasize reliability using review scores and counts
4. Write in a friendly and professional tone, 2-3 sentences
5. Write only the main content without unnecessary greetings or closing remarks

Recommendation Reason:
"""

    return prompt
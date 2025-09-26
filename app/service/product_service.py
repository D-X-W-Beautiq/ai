import google.generativeai as genai
import json
import os
from pathlib import Path

# Gemini API 설정
GEMINI_API_KEY = ""  # 실제 키로 교체 또는 환경 변수 사용
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-exp")

def run_inference(request: dict) -> dict:
    """
    제품별 추천 이유 생성
    
    Args:
        request: {
            "predictions": {...},           # NIA 분석 결과
            "needs": ["moisture", "pore"],  # 부족한 카테고리 (영문)
            "filtered_products": [          # 백엔드에서 필터링된 제품들
                {
                    "product_id": "string",
                    "product_name": "string",
                    "brand": "string",
                    "category": "string",
                    "price": "integer",
                    "review_score": "float",
                    "review_count": "integer",
                    "ingredients": ["string"]
                }
            ],
            "locale": "string"
        }
    
    Returns:
        {
            "status": "success",
            "recommendations": [
                {
                    "product_id": "string",
                    "reason": "string"
                }
            ]
        }
    """
    try:
        # 입력 검증
        if "predictions" not in request:
            raise ValueError("Missing required field: predictions")
        if "needs" not in request:
            raise ValueError("Missing required field: needs")
        if "filtered_products" not in request:
            raise ValueError("Missing required field: filtered_products")
        
        predictions = request["predictions"]
        needs = request["needs"]
        filtered_products = request["filtered_products"]
        locale = request.get("locale", "ko-KR")
        
        recommendations = []
        
        # 각 제품별로 추천 이유 생성
        for product in filtered_products:
            # 프롬프트 생성
            prompt = generate_recommendation_prompt(
                predictions, 
                needs, 
                product, 
                locale
            )
            
            # Gemini API 호출
            response = model.generate_content(prompt)
            reason = response.text.strip()
            
            recommendations.append({
                "product_id": product["product_id"],
                "reason": reason
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
def generate_recommendation_prompt(predictions, needs, product, locale):
    """추천 이유 생성을 위한 프롬프트 작성"""
    
    # 카테고리 매핑
    category_map = {
        "moisture": "수분",
        "elasticity": "탄력",
        "wrinkle": "주름",
        "pigmentation": "색소침착",
        "pore": "모공"
    }
    
    # 부족한 영역 정보 추출
    concern_details = []
    
    for need in needs:
        korean_name = category_map.get(need, need)
        
        if need == "moisture":
            avg_moisture = (
                predictions.get("forehead_moisture", 0) +
                predictions.get("cheek_moisture", 0) +
                predictions.get("chin_moisture", 0)
            ) / 3
            concern_details.append(f"{korean_name} (평균 {avg_moisture:.0f}점, 기준 70점 미만)")
        
        elif need == "elasticity":
            avg_elasticity = (
                predictions.get("forehead_elasticity_R2", 0) +
                predictions.get("cheek_elasticity_R2", 0) +
                predictions.get("chin_elasticity_R2", 0)
            ) / 3
            concern_details.append(f"{korean_name} (평균 {avg_elasticity:.2f}, 기준 0.7 미만)")
        
        elif need == "wrinkle":
            wrinkle_value = predictions.get("perocular_wrinkle_Ra", 0)
            concern_details.append(f"{korean_name} (눈가 주름 {wrinkle_value}, 기준 30 이상)")
        
        elif need == "pigmentation":
            forehead_pig = predictions.get("forehead_pigmentation", 0)
            cheek_pig = predictions.get("cheek_pigmentation", 0)
            concern_details.append(f"{korean_name} (이마 {forehead_pig}, 볼 {cheek_pig}, 기준 250 이상)")
        
        elif need == "pore":
            pore_value = predictions.get("cheek_pore", 0)
            concern_details.append(f"{korean_name} (볼 모공 {pore_value}, 기준 1800 이상)")
    
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
- 주요 성분: {', '.join(product.get('ingredients', []))}

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
- Key Ingredients: {', '.join(product.get('ingredients', []))}

Requirements:
1. Explain specifically why this product is suitable based on skin analysis
2. Briefly mention the efficacy of key ingredients
3. Emphasize reliability using review scores and counts
4. Write in a friendly and professional tone, 2-3 sentences
5. Write only the main content without unnecessary greetings or closing remarks

Recommendation Reason:
"""
    
    return prompt
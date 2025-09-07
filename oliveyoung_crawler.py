import asyncio
import pandas as pd
from playwright.async_api import async_playwright
from urllib.parse import urlencode
from datetime import datetime
import re
import random

class OliveYoungScraper:
    def __init__(self):
        self.base_url = "https://www.oliveyoung.co.kr/store/search/getSearchMain.do"

    # --- URL 생성 ---
    def build_search_url(self, query: str, page: int = 1) -> str:
        if page == 1:
            params = {
                "query": query,
                "giftYn": "N",
                "page": "랭킹",
                "click": "검색창",
                "search_name": query,
            }
            return f"{self.base_url}?{urlencode(params, encoding='utf-8')}"

        start_count = (page - 1) * 24
        params = {
            "startCount": str(start_count),
            "sort": "RANK/DESC",
            "goods_sort": "WEIGHT/DESC,RANK/DESC",
            "collection": "ALL",
            "realQuery": query,
            "reQuery": "",
            "viewtype": "image",
            "listnum": "24",
            "query": query,
            "typeChk": "thum",
            "quickYn": "N",
        }
        return f"{self.base_url}?{urlencode(params, encoding='utf-8')}"

    # --- 상품 상세 페이지에서 리뷰 점수와 성분 추출 ---
    async def get_review_score_from_detail_page(self, page, product_url):
        """상품 상세 페이지에서 리뷰 점수와 리뷰수, 성분 추출"""
        detail_page = None
        try:
            if not product_url:
                print(f"    → 상품 URL 없음")
                return {"review_score": "", "review_count": "", "ingredients": ""}
            
            print(f"  상세페이지 접속: {product_url[:60]}...")
            
            # 새 탭에서 상품 상세 페이지 열기
            detail_page = await page.context.new_page()
            await detail_page.goto(product_url, wait_until='domcontentloaded', timeout=30000)
            await detail_page.wait_for_timeout(2000)  # 2초 대기
            
            # 페이지 로딩 확인
            title = await detail_page.title()
            print(f"    페이지 제목: {title[:50]}...")
            
            # 리뷰 점수와 리뷰수 추출
            review_score = ""
            review_count = ""
            
            # 방법 1: #repReview 요소 전체 확인
            try:
                review_area = await detail_page.query_selector('#repReview')
                if review_area:
                    rep_text = await review_area.inner_text()
                    print(f"    #repReview 찾음")
                    print(f"    텍스트: {rep_text[:100]}...")
                    
                    # <b> 태그에서 점수 추출
                    b_element = await review_area.query_selector('b')
                    if b_element:
                        b_text = await b_element.inner_text()
                        print(f"    <b> 태그 내용: '{b_text.strip()}'")
                        
                        # 숫자.숫자 패턴 찾기 (리뷰점수)
                        score_match = re.search(r'(\d+\.\d+)', b_text.strip())
                        if score_match:
                            review_score = score_match.group(1)
                            print(f"    → 점수 추출 성공: {review_score}")
                        else:
                            print(f"    → <b> 태그에서 점수 패턴 못찾음")
                    else:
                        print(f"    → <b> 태그 없음")
                    
                    # <em> 태그 또는 괄호에서 리뷰수 추출
                    em_element = await review_area.query_selector('em')
                    if em_element:
                        em_text = await em_element.inner_text()
                        print(f"    <em> 태그 내용: '{em_text.strip()}'")
                        
                        # 괄호 안의 숫자 추출 (5,435건)
                        count_match = re.search(r'\(([\d,]+)건?\)', em_text.strip())
                        if count_match:
                            review_count = count_match.group(1).replace(",", "")
                            print(f"    → 리뷰수 추출 성공: {review_count}")
                        else:
                            print(f"    → <em> 태그에서 리뷰수 패턴 못찾음")
                    else:
                        print(f"    → <em> 태그 없음, 전체 텍스트에서 검색")
                        # <em> 태그가 없으면 전체 텍스트에서 찾기
                        count_match = re.search(r'\(([\d,]+)건?\)', rep_text)
                        if count_match:
                            review_count = count_match.group(1).replace(",", "")
                            print(f"    → 전체 텍스트에서 리뷰수 추출 성공: {review_count}")
                        else:
                            print(f"    → 전체 텍스트에서도 리뷰수 패턴 못찾음")
                else:
                    print(f"    → #repReview 요소 없음")
                    
                    # 대안: 다른 셀렉터로 시도
                    alternatives = [
                        '.prd_social_info',
                        '[id*="review"]',
                        '[class*="review"]'
                    ]
                    
                    for alt_selector in alternatives:
                        alt_element = await detail_page.query_selector(alt_selector)
                        if alt_element:
                            alt_text = await alt_element.inner_text()
                            print(f"    대안 셀렉터 {alt_selector}: {alt_text[:50]}...")
                            break
                            
            except Exception as e:
                print(f"    리뷰 추출 중 오류: {e}")
            
            # 성분 정보 추출
            ingredients = ""
            try:
                print(f"    구매정보 탭으로 이동 중...")
                
                # 구매정보 탭 클릭
                buy_info_tab = await detail_page.query_selector('#buyInfo a.goods_buyinfo')
                if buy_info_tab:
                    await buy_info_tab.click()
                    await detail_page.wait_for_timeout(2000)  # 탭 로딩 대기
                    print(f"    → 구매정보 탭 클릭 완료")
                    
                    # 성분 정보 찾기
                    ingredient_elements = await detail_page.query_selector_all('.detail_info_list')
                    for element in ingredient_elements:
                        dt_element = await element.query_selector('dt')
                        if dt_element:
                            dt_text = await dt_element.inner_text()
                            print(f"    dt 텍스트: {dt_text[:50]}...")
                            
                            # "화장품법에 따라 기재해야 하는 모든 성분" 찾기
                            if "화장품법" in dt_text and "성분" in dt_text:
                                dd_element = await element.query_selector('dd')
                                if dd_element:
                                    ingredients = await dd_element.inner_text()
                                    print(f"    → 성분 추출 성공: {ingredients[:100]}...")
                                    break
                    
                    if not ingredients:
                        print(f"    → 성분 정보를 찾을 수 없음")
                else:
                    print(f"    → 구매정보 탭을 찾을 수 없음")
                    
            except Exception as e:
                print(f"    성분 추출 중 오류: {e}")
            
            await detail_page.close()
            
            if review_score:
                print(f"    → 최종 점수: {review_score}")
            else:
                print(f"    → 점수 추출 실패")
                
            if review_count:
                print(f"    → 최종 리뷰수: {review_count}")
            else:
                print(f"    → 리뷰수 추출 실패")
                
            if ingredients:
                print(f"    → 성분 추출 완료")
            else:
                print(f"    → 성분 추출 실패")
                
            return {"review_score": review_score, "review_count": review_count, "ingredients": ingredients}
            
        except Exception as e:
            print(f"    → 페이지 오류: {e}")
            if detail_page:
                try:
                    await detail_page.close()
                except:
                    pass
            return {"review_score": "", "review_count": "", "ingredients": ""}

    # --- 단일 상품 파싱 ---
    async def extract_product_info(self, page, li):
        try:
            info = {}

            a_name = await li.query_selector(".prd_name a")
            brand_el = await li.query_selector(".prd_name .tx_brand")
            name_el = await li.query_selector(".prd_name .tx_name")
            info["brand"] = (await brand_el.inner_text()).strip() if brand_el else ""
            info["name"] = (await name_el.inner_text()).strip() if name_el else ""

            # 가격
            org_el = await li.query_selector(".prd_price .tx_org .tx_num")
            cur_el = await li.query_selector(".prd_price .tx_cur .tx_num")
            info["original_price"] = (await org_el.inner_text()).replace(",", "") if org_el else ""
            info["current_price"] = (await cur_el.inner_text()).replace(",", "") if cur_el else ""

            # 리뷰수 추출 (검색 페이지에서 기본값)
            rp = await li.query_selector(".prd_point_area")
            if rp:
                m = re.search(r"\(([\d,]+)\)", await rp.inner_text())
                info["review_count"] = m.group(1).replace(",", "") if m else "0"
            else:
                info["review_count"] = "0"

            # 상품 링크 - JavaScript에서 실제 URL 추출
            href = ""
            if a_name:
                href_attr = await a_name.get_attribute("href")
                onclick_attr = await a_name.get_attribute("onclick")
                
                print(f"    원본 href: {href_attr[:100] if href_attr else 'None'}...")
                print(f"    onclick: {onclick_attr[:100] if onclick_attr else 'None'}...")
                
                # href가 javascript:로 시작하는 경우 onclick에서 상품번호 추출
                if href_attr and href_attr.startswith("javascript:"):
                    # moveGoodsDetailForSearch 함수에서 goodsNo 추출
                    goods_match = re.search(r"moveGoodsDetailForSearch\('([A-Z0-9]+)'", href_attr)
                    if goods_match:
                        goods_no = goods_match.group(1)
                        href = f"https://www.oliveyoung.co.kr/store/goods/getGoodsDetail.do?goodsNo={goods_no}"
                        print(f"    추출된 상품번호: {goods_no}")
                        print(f"    생성된 URL: {href}")
                    else:
                        print(f"    상품번호 추출 실패")
                        
                # onclick에서도 시도
                elif onclick_attr:
                    goods_match = re.search(r"moveGoodsDetailForSearch\('([A-Z0-9]+)'", onclick_attr)
                    if goods_match:
                        goods_no = goods_match.group(1)
                        href = f"https://www.oliveyoung.co.kr/store/goods/getGoodsDetail.do?goodsNo={goods_no}"
                        print(f"    onclick에서 추출된 상품번호: {goods_no}")
                        
                # 일반 URL인 경우
                elif href_attr:
                    if href_attr.startswith("/"):
                        href = "https://www.oliveyoung.co.kr" + href_attr
                    elif href_attr.startswith("http"):
                        href = href_attr
                        
            info["product_url"] = href or ""

            # 상품 상세 페이지에서 리뷰 점수와 리뷰수, 성분 추출
            review_data = await self.get_review_score_from_detail_page(page, info["product_url"])
            
            # 반환값이 딕셔너리인지 확인
            if isinstance(review_data, dict):
                info["review_score"] = review_data.get("review_score", "")
                info["ingredients"] = review_data.get("ingredients", "")
                # 상세 페이지에서 추출한 리뷰수가 있으면 업데이트, 없으면 검색 페이지 리뷰수 유지
                if review_data.get("review_count"):
                    info["review_count"] = review_data["review_count"]
            else:
                # 예외 상황에서 문자열이 반환된 경우 (기존 호환성)
                info["review_score"] = review_data if review_data else ""
                info["ingredients"] = ""

            # 태그
            flags = []
            for el in await li.query_selector_all(".prd_flag .icon_flag"):
                flags.append((await el.inner_text()).strip())
            info["flags"] = ", ".join(flags)

            # 베스트/신상
            tf = await li.query_selector(".thumb_flag")
            info["thumb_flag"] = (await tf.inner_text()).strip() if tf else ""

            # 이미지
            img = await li.query_selector(".prd_thumb img")
            src = await img.get_attribute("src") if img else ""
            if src and src.startswith("//"):
                src = "https:" + src
            info["image_url"] = src or ""

            return info
        except Exception as e:
            print("extract error:", e)
            return None

    # --- 페이지 단위 크롤 ---
    async def scrape_page(self, browser, query, page_num: int):
        context = None
        try:
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                extra_http_headers={
                    "Referer": "https://www.oliveyoung.co.kr/",
                    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
                },
            )
            context.set_default_timeout(60000)
            page = await context.new_page()

            url = self.build_search_url(query, page_num)
            print(f"{page_num}페이지 크롤링 중... URL: {url}")
            await page.goto(url, wait_until="networkidle", timeout=60000)

            # 올리브영 실제 구조에 맞는 셀렉터로 변경
            await page.wait_for_selector("li.flag.li_result", timeout=15000)
            items = await page.query_selector_all("li.flag.li_result")
            
            if not items:
                print(f"{page_num}페이지 결과 없음")
                await context.close()
                return []

            print(f"{page_num}페이지에서 {len(items)}개 상품 발견")
            results = []
            for i, li in enumerate(items):
                # 1) 인뷰+갱신 대기: "(0)" 초기값이 아닌 숫자 주입을 잠시 기다림
                await li.scroll_into_view_if_needed()
                try:
                    await page.wait_for_function(
                        """
                        li => {
                          const el = li.querySelector('.prd_point_area');
                          if (!el) return false;
                          const m = el.textContent.match(/\\((\\d[\\d,]*)\\)/);
                          return m && m[1] !== '0';
                        }""",
                        arg=li,
                        timeout=1500,  # 실제 0일 수도 있으니 과도하게 기다리지 않음
                    )
                except:
                    pass  # 그대로 진행

                info = await self.extract_product_info(page, li)  # page 파라미터 추가
                if not info:
                    continue
                info["page"] = page_num
                info["rank_in_page"] = i + 1
                info["overall_rank"] = (page_num - 1) * 24 + i + 1
                results.append(info)

            print(f"{page_num}페이지에서 {len(results)}개 상품 수집 완료")
            await context.close()
            return results

        except Exception as e:
            print(f"{page_num}페이지 오류:", e)
            if context:
                try:
                    await context.close()
                except:
                    pass
            return []

    # --- 여러 페이지 ---
    async def scrape_multiple_pages(self, query, max_pages: int = 5):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            all_products = []
            for page_num in range(1, max_pages + 1):
                batch = await self.scrape_page(browser, query, page_num)
                if not batch:
                    break
                all_products.extend(batch)
                await asyncio.sleep(1 + random.random())  # 간단한 우회
            await browser.close()
            return all_products

    # --- 여러 검색어 자동 크롤링 ---
    async def scrape_multiple_queries(self, query_config):
        """여러 검색어를 자동으로 크롤링"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            all_results = []
            
            total_queries = len(query_config)
            for idx, (query, max_pages) in enumerate(query_config.items(), 1):
                print(f"\n=== [{idx}/{total_queries}] '{query}' 검색 시작 ({max_pages}페이지) ===")
                
                query_products = []
                for page_num in range(1, max_pages + 1):
                    batch = await self.scrape_page(browser, query, page_num)
                    if not batch:
                        break
                    
                    # 각 상품에 카테고리 정보 추가
                    for product in batch:
                        product["category"] = query
                    
                    query_products.extend(batch)
                    await asyncio.sleep(1 + random.random())  # 간단한 우회
                
                print(f"'{query}' 카테고리에서 총 {len(query_products)}개 상품 수집 완료")
                all_results.extend(query_products)
            
            await browser.close()
            print(f"\n=== 전체 크롤링 완료: {len(all_results)}개 상품 ===")
            return all_results

    # --- Excel 저장 ---
    def save_to_excel(self, products, query: str):
        if not products:
            print("저장할 데이터가 없습니다.")
            return
        df = pd.DataFrame(products)

        # 숫자형 캐스팅
        for c in ["original_price", "current_price", "review_count",
                  "overall_rank", "page", "rank_in_page"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

        # 리뷰 점수는 float 타입으로 변환
        if "review_score" in df.columns:
            df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")

        columns_order = [
            "category", "overall_rank", "page", "rank_in_page", "brand", "name",
            "original_price", "current_price", "review_score", "review_count",
            "ingredients", "flags", "thumb_flag", "image_url", "product_url"
        ]
        df = df[[c for c in columns_order if c in df.columns]]

        column_mapping = {
            "category": "카테고리",
            "overall_rank": "전체순위",
            "page": "페이지",
            "rank_in_page": "페이지내순위",
            "brand": "브랜드",
            "name": "상품명",
            "original_price": "정가",
            "current_price": "판매가",
            "review_score": "리뷰점수",
            "review_count": "리뷰수",
            "ingredients": "성분",
            "flags": "태그",
            "thumb_flag": "베스트/신상",
            "image_url": "이미지URL",
            "product_url": "상품URL",
        }
        df = df.rename(columns=column_mapping)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"올리브영_{query}_{ts}.xlsx"
        df.to_excel(filename, index=False, engine="openpyxl")
        print(f"'{filename}' 저장. 총 {len(products)}개.")

# --- 실행부 ---
async def main():
    scraper = OliveYoungScraper()
    
    print("=== 올리브영 크롤링 프로그램 ===")
    print("1. 단일 검색어 크롤링")
    print("2. 자동 다중 검색어 크롤링 (수분, 탄력, 주름, 색소침착, 모공)")
    
    mode = input("모드를 선택하세요 (1 또는 2): ").strip()
    
    if mode == "1":
        # 기존 단일 검색어 모드
        query = input("검색어를 입력하세요: ").strip()
        if not query:
            print("검색어를 입력해주세요.")
            return
        try:
            max_pages = int(input("크롤링할 페이지 수(기본 3): ") or "3")
        except ValueError:
            max_pages = 3

        print(f"'{query}'로 {max_pages}페이지 수집 시작")
        products = await scraper.scrape_multiple_pages(query, max_pages)
        if products:
            scraper.save_to_excel(products, query)
            print("완료")
        else:
            print("결과 없음")
            
    elif mode == "2":
        # 자동 다중 검색어 모드
        print("\n=== 자동 다중 검색어 크롤링 설정 ===")
        
        # 기본 설정
        default_pages = {
            "수분": 3,
            "탄력": 3, 
            "주름": 3,
            "색소침착": 3,
            "모공": 3
        }
        
        query_config = {}
        
        for query, default_page in default_pages.items():
            try:
                pages = input(f"{query} - 페이지 수 (기본값: {default_page}): ").strip()
                pages = int(pages) if pages else default_page
                query_config[query] = pages
                print(f"  → {query}: {pages}페이지")
            except ValueError:
                query_config[query] = default_page
                print(f"  → {query}: {default_page}페이지 (기본값 적용)")
        
        print(f"\n총 {len(query_config)}개 카테고리로 자동 크롤링을 시작합니다...")
        
        # 다중 검색어 크롤링 실행
        all_products = await scraper.scrape_multiple_queries(query_config)
        
        if all_products:
            # 통합 엑셀 파일 저장
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"올리브영_통합_{ts}.xlsx"
            
            # save_to_excel 함수를 직접 호출하되 파일명 수정
            df = pd.DataFrame(all_products)

            # 숫자형 캐스팅
            for c in ["original_price", "current_price", "review_count",
                      "overall_rank", "page", "rank_in_page"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

            # 리뷰 점수는 float 타입으로 변환
            if "review_score" in df.columns:
                df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")

            columns_order = [
                "category", "overall_rank", "page", "rank_in_page", "brand", "name",
                "original_price", "current_price", "review_score", "review_count",
                "ingredients", "flags", "thumb_flag", "image_url", "product_url"
            ]
            df = df[[c for c in columns_order if c in df.columns]]

            column_mapping = {
                "category": "카테고리",
                "overall_rank": "전체순위",
                "page": "페이지",
                "rank_in_page": "페이지내순위",
                "brand": "브랜드",
                "name": "상품명",
                "original_price": "정가",
                "current_price": "판매가",
                "review_score": "리뷰점수",
                "review_count": "리뷰수",
                "ingredients": "성분",
                "flags": "태그",
                "thumb_flag": "베스트/신상",
                "image_url": "이미지URL",
                "product_url": "상품URL",
            }
            df = df.rename(columns=column_mapping)

            df.to_excel(filename, index=False, engine="openpyxl")
            print(f"'{filename}' 저장 완료! 총 {len(all_products)}개 상품")
            
            # 카테고리별 요약 출력
            category_summary = df['카테고리'].value_counts()
            print("\n=== 카테고리별 수집 결과 ===")
            for category, count in category_summary.items():
                print(f"{category}: {count}개")
        else:
            print("수집된 데이터가 없습니다.")
    
    else:
        print("올바른 모드를 선택해주세요 (1 또는 2)")

if __name__ == "__main__":
    asyncio.run(main())
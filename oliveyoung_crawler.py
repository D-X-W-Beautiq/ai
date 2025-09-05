import asyncio
import pandas as pd
from playwright.async_api import async_playwright
import re
from urllib.parse import quote
import time
from datetime import datetime

class OliveYoungCrawler:
    def __init__(self):
        self.products = []
        self.page_size = 24  # 한 페이지당 상품 수
        self.categories = ["수분", "탄력", "주름", "색소침착", "모공"]
        
    def build_search_url(self, query, page=1):
        """검색 URL 생성"""
        encoded_query = quote(query)
        
        if page == 1:
            # 첫 번째 페이지 URL - 단순화
            url = f"https://www.oliveyoung.co.kr/store/search/getSearchMain.do?query={encoded_query}"
        else:
            # 2페이지부터는 startCount 사용
            start_count = (page - 1) * self.page_size
            url = f"https://www.oliveyoung.co.kr/store/search/getSearchMain.do?startCount={start_count}&query={encoded_query}&sort=RANK%2FDESC&listnum=24"
            
        return url
    
    async def scroll_and_wait(self, page):
        """페이지를 스크롤하여 모든 상품이 로드되도록 함"""
        await page.evaluate("""
            () => {
                return new Promise((resolve) => {
                    let totalHeight = 0;
                    const distance = 100;
                    const timer = setInterval(() => {
                        const scrollHeight = document.body.scrollHeight;
                        window.scrollBy(0, distance);
                        totalHeight += distance;

                        if(totalHeight >= scrollHeight){
                            clearInterval(timer);
                            resolve();
                        }
                    }, 100);
                });
            }
        """)
        await page.wait_for_timeout(2000)  # 추가 대기시간
    
    def extract_number(self, text):
        """텍스트에서 숫자만 추출"""
        if not text:
            return ""
        # 괄호 안의 내용은 먼저 제거
        text = re.sub(r'\([^)]*\)', '', text)
        numbers = re.findall(r'[\d,]+\.?\d*', text)
        return numbers[0] if numbers else ""
    
    async def extract_products_from_page(self, page, category):
        """현재 페이지에서 상품 정보 추출"""
        products = []
        
        # 상품 리스트 대기
        await page.wait_for_selector('.flag.li_result', timeout=10000)
        
        # 모든 상품 요소 찾기
        product_elements = await page.query_selector_all('.flag.li_result')
        
        for idx, element in enumerate(product_elements):
            try:
                # 브랜드명
                brand_element = await element.query_selector('.tx_brand')
                brand = await brand_element.inner_text() if brand_element else ""
                
                # 상품명
                name_element = await element.query_selector('.tx_name')
                product_name = await name_element.inner_text() if name_element else ""
                
                # 가격
                price_element = await element.query_selector('.tx_cur .tx_num')
                price = await price_element.inner_text() if price_element else ""
                price = self.extract_number(price)
                
                # 리뷰 점수 및 개수 - 개선된 추출
                review_score = 0.0
                review_count = 0
                
                review_element = await element.query_selector('.prd_point_area')
                if review_element:
                    review_text = await review_element.inner_text()
                    
                    # 점수 추출 (숫자.숫자 형태)
                    score_match = re.search(r'(\d+\.\d+)', review_text)
                    if score_match:
                        review_score = float(score_match.group(1))
                    
                    # 리뷰 개수 추출 (괄호 안의 숫자)
                    count_match = re.search(r'\((\d+)[건)]*\)', review_text)
                    if count_match:
                        review_count = int(count_match.group(1))
                
                # 상품 링크
                link_element = await element.query_selector('.prd_thumb')
                product_link = ""
                product_id = f"OY_{category}_{idx}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                if link_element:
                    href = await link_element.get_attribute('href')
                    if href:
                        product_link = href if href.startswith('http') else f"https://www.oliveyoung.co.kr{href}"
                        # URL에서 상품 ID 추출 시도
                        id_match = re.search(r'goodsNo=(\w+)', href)
                        if id_match:
                            product_id = id_match.group(1)
                
                # 이미지 URL
                img_element = await element.query_selector('.prd_thumb img')
                image_url = ""
                if img_element:
                    image_url = await img_element.get_attribute('src')
                
                # 배송 정보
                delivery_element = await element.query_selector('.icon_flag.delivery')
                delivery_info = await delivery_element.inner_text() if delivery_element else ""
                
                # 베스트/신상 플래그
                flag_element = await element.query_selector('.thumb_flag')
                flag = await flag_element.inner_text() if flag_element else ""
                
                product_info = {
                    '카테고리': category,
                    '상품ID': product_id,
                    '브랜드': brand,
                    '상품명': product_name,
                    '가격': int(price.replace(',', '')) if price else 0,
                    '리뷰점수': review_score,
                    '리뷰수': review_count,
                    '배송정보': delivery_info,
                    '플래그': flag,
                    '상품링크': product_link,
                    '이미지URL': image_url,
                    '크롤링시간': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                products.append(product_info)
                
            except Exception as e:
                print(f"상품 정보 추출 중 오류: {e}")
                continue
        
        return products
    
    async def crawl_search_results(self, query, max_pages=10):
        """검색 결과 크롤링"""
        print(f"\n'{query}' 검색 결과 크롤링 시작...")
        category_products = []
        
        async with async_playwright() as p:
            # 브라우저 실행 - 더 많은 옵션 추가
            browser = await p.chromium.launch(
                headless=False,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox',
                    '--disable-setuid-sandbox'
                ]
            )
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                viewport={'width': 1920, 'height': 1080}
            )
            page = await context.new_page()
            
            # 자동화 감지 방지
            await page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            """)
            
            try:
                page_num = 1
                total_products = 0
                
                while page_num <= max_pages:
                    print(f"  {page_num}페이지 크롤링 중...")
                    
                    # 페이지 URL 생성 및 이동
                    url = self.build_search_url(query, page_num)
                    
                    try:
                        # 더 관대한 타임아웃과 대기 조건
                        await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                        await page.wait_for_timeout(5000)  # 추가 대기시간
                        
                        # 페이지 로딩 확인 - 여러 시도
                        for attempt in range(3):
                            try:
                                await page.wait_for_selector('.flag.li_result, .search_result, .prd_info', timeout=15000)
                                break
                            except:
                                if attempt == 2:
                                    print(f"    페이지 로딩 실패 - 다음 시도")
                                    break
                                await page.wait_for_timeout(3000)
                        
                    except Exception as e:
                        print(f"  페이지 로딩 오류: {e}")
                        if page_num == 1:
                            # 첫 페이지도 실패하면 다른 방법 시도
                            print("  직접 검색 페이지로 이동 시도...")
                            try:
                                await page.goto("https://www.oliveyoung.co.kr", timeout=30000)
                                await page.wait_for_timeout(3000)
                                
                                # 검색창 찾기 및 검색어 입력
                                search_input = await page.query_selector('input[name="query"], .search_input, input[placeholder*="검색"]')
                                if search_input:
                                    await search_input.fill(query)
                                    await page.keyboard.press('Enter')
                                    await page.wait_for_timeout(5000)
                                else:
                                    print("  검색창을 찾을 수 없습니다.")
                                    break
                            except Exception as e2:
                                print(f"  대체 방법도 실패: {e2}")
                                break
                        else:
                            print(f"  {page_num}페이지 로딩 실패, 크롤링 종료")
                            break
                    
                    # 스크롤하여 모든 상품 로드
                    await self.scroll_and_wait(page)
                    
                    # 상품이 있는지 확인
                    products_exist = await page.query_selector('.flag.li_result')
                    if not products_exist:
                        print(f"  {page_num}페이지에 상품이 없습니다. 크롤링 종료.")
                        break
                    
                    # 현재 페이지의 상품 정보 추출
                    page_products = await self.extract_products_from_page(page, query)
                    
                    if not page_products:
                        print(f"  {page_num}페이지에서 상품을 찾을 수 없습니다. 크롤링 종료.")
                        break
                    
                    category_products.extend(page_products)
                    total_products += len(page_products)
                    
                    print(f"    → {len(page_products)}개 상품 수집 완료 (총 {total_products}개)")
                    
                    # 다음 페이지로
                    page_num += 1
                    await page.wait_for_timeout(3000)  # 페이지 간 대기시간 증가
                
            except Exception as e:
                print(f"크롤링 중 오류 발생: {e}")
            finally:
                await browser.close()
        
        print(f"'{query}' 크롤링 완료! 총 {len(category_products)}개 상품 수집")
        return category_products
    
    def save_to_excel(self, filename=None):
        """수집한 데이터를 엑셀 파일로 저장"""
        if not self.products:
            print("저장할 데이터가 없습니다.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"oliveyoung_5categories_{timestamp}.xlsx"
        
        df = pd.DataFrame(self.products)
        
        # 엑셀 파일로 저장 - 전체 시트와 카테고리별 시트
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 전체 데이터 시트
            df.to_excel(writer, sheet_name='전체상품', index=False)
            
            # 카테고리별 시트
            for category in self.categories:
                category_df = df[df['카테고리'] == category]
                if not category_df.empty:
                    category_df.to_excel(writer, sheet_name=category, index=False)
            
            # 컬럼 너비 조정
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"\n엑셀 파일 저장 완료: {filename}")
        print(f"총 {len(self.products)}개 상품 정보가 저장되었습니다.")
        
        # 카테고리별 통계 출력
        print("\n=== 카테고리별 수집 통계 ===")
        for category in self.categories:
            category_count = len(df[df['카테고리'] == category])
            if category_count > 0:
                avg_price = df[df['카테고리'] == category]['가격'].mean()
                avg_review = df[df['카테고리'] == category]['리뷰점수'].mean()
                print(f"{category}: {category_count}개 상품 (평균가격: {avg_price:,.0f}원, 평균평점: {avg_review:.1f})")

async def main():
    """메인 실행 함수"""
    crawler = OliveYoungCrawler()
    
    print("\n올리브영 5대 카테고리 제품 크롤링")
    print("="*50)
    print("크롤링할 카테고리: 수분, 탄력, 주름, 색소침착, 모공")
    print("="*50)
    
    # 각 카테고리별 페이지 수 입력
    category_pages = {}
    print("\n각 카테고리별 크롤링할 페이지 수를 입력하세요:")
    
    for category in crawler.categories:
        try:
            pages = int(input(f"{category}: ").strip())
            category_pages[category] = pages
        except ValueError:
            category_pages[category] = 5
            print(f"  → 잘못된 입력. {category}는 5페이지로 설정됩니다.")
    
    print(f"\n크롤링 시작...")
    print("설정된 페이지 수:")
    for cat, pages in category_pages.items():
        print(f"  - {cat}: {pages}페이지")
    
    # 각 카테고리별 크롤링 실행
    all_products = []
    for category in crawler.categories:
        products = await crawler.crawl_search_results(category, max_pages=category_pages[category])
        all_products.extend(products)
        
        # 카테고리 간 대기 (마지막 카테고리 제외)
        if category != crawler.categories[-1]:
            print(f"\n다음 카테고리 크롤링 전 3초 대기...")
            await asyncio.sleep(3)
    
    crawler.products = all_products
    
    if crawler.products:
        # 엑셀 파일로 저장
        crawler.save_to_excel()
        
        # 결과 요약 출력
        print(f"\n=== 전체 크롤링 결과 요약 ===")
        print(f"총 수집된 상품 수: {len(crawler.products)}개")
    else:
        print("수집된 상품이 없습니다.")

# 실행
if __name__ == "__main__":
    # 필요한 패키지 설치 안내
    print("필요한 패키지가 설치되어 있는지 확인하세요:")
    print("pip install playwright pandas openpyxl")
    print("playwright install")
    print()
    
    # 프로그램 실행
    asyncio.run(main())
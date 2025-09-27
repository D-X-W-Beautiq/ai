import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from urllib.parse import urlparse

def create_folder(path):
    """폴더가 없으면 생성"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"폴더 생성: {path}")

def download_image(url, save_path, image_name):
    """이미지 URL에서 이미지 다운로드"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            # 확장자 결정
            content_type = response.headers.get('content-type', '')
            if 'jpeg' in content_type or 'jpg' in content_type:
                ext = '.jpg'
            elif 'png' in content_type:
                ext = '.png'
            elif 'gif' in content_type:
                ext = '.gif'
            else:
                ext = '.jpg'  # 기본값
            
            file_path = os.path.join(save_path, f"{image_name}{ext}")
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"이미지 저장 완료: {file_path}")
            return True
    except Exception as e:
        print(f"이미지 다운로드 실패: {e}")
        return False

def crawl_google_images(person_name, save_base_path, num_images=15):
    """구글 이미지 검색 및 크롤링"""
    
    # 저장 폴더 생성
    person_folder = os.path.join(save_base_path, person_name)
    create_folder(person_folder)
    
    # Chrome 옵션 설정
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    # WebDriver 시작
    driver = webdriver.Chrome(options=options)
    driver.maximize_window()
    
    try:
        # 검색어 구성
        search_query = f"{person_name} 얼굴 고화질"
        search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
        
        print(f"검색 중: {search_query}")
        driver.get(search_url)
        
        # 페이지 로딩 대기
        wait = WebDriverWait(driver, 10)
        time.sleep(2)
        
        # 이미지 썸네일 찾기
        thumbnails = driver.find_elements(By.CSS_SELECTOR, "div[jsname='dTDiAc']")
        
        if not thumbnails:
            print("이미지를 찾을 수 없습니다.")
            return
        
        print(f"총 {len(thumbnails)}개의 이미지 발견")
        
        downloaded_count = 0
        
        for i in range(min(num_images, len(thumbnails))):
            try:
                # 썸네일 클릭
                thumbnails[i].click()
                time.sleep(2)
                
                # 큰 이미지 찾기 (여러 선택자 시도)
                large_image = None
                
                # 선택자 1: 클래스명으로 찾기
                try:
                    large_image = driver.find_element(By.CSS_SELECTOR, "img.sFlh5c.FyHeAf.iPVvYb")
                except:
                    pass
                
                # 선택자 2: jsname으로 찾기
                if not large_image:
                    try:
                        large_image = driver.find_element(By.CSS_SELECTOR, "img[jsname='kn3ccd']")
                    except:
                        pass
                
                # 선택자 3: 더 일반적인 선택자
                if not large_image:
                    try:
                        large_images = driver.find_elements(By.CSS_SELECTOR, "img.iPVvYb")
                        if large_images:
                            large_image = large_images[0]
                    except:
                        pass
                
                if large_image:
                    # 이미지 URL 가져오기
                    img_url = large_image.get_attribute('src')
                    
                    # data URL이 아닌 경우만 다운로드
                    if img_url and not img_url.startswith('data:'):
                        image_name = f"{person_name}_{downloaded_count + 1:03d}"
                        if download_image(img_url, person_folder, image_name):
                            downloaded_count += 1
                            print(f"진행 상황: {downloaded_count}/{num_images}")
                    else:
                        print(f"이미지 {i+1}: data URL이거나 URL을 가져올 수 없음")
                else:
                    print(f"이미지 {i+1}: 큰 이미지를 찾을 수 없음")
                
                # 다음 이미지로 이동하기 전 잠시 대기
                time.sleep(1)
                
            except Exception as e:
                print(f"이미지 {i+1} 처리 중 오류: {e}")
                continue
        
        print(f"\n크롤링 완료: {person_name}")
        print(f"총 {downloaded_count}개의 이미지 저장됨")
        
    except Exception as e:
        print(f"크롤링 중 오류 발생: {e}")
    
    finally:
        driver.quit()

def main():
    # 저장 경로
    save_path = 'image'
    
    # 크롤링할 인물 리스트
    people_list = [
    "김지원", "김유정",
    "블랙핑크 제니", "블랙핑크 지수", "블랙핑크 로제", "블랙핑크 리사",
    "아이브 장원영", "아이브 안유진", "에스파 카리나", "에스파 윈터", "뉴진스 다니엘", "아이브 리즈", "르세라핌 홍은채", "마마무 화사",
    "전소미", "위키미키 김도연", "위키미키 최유정", "여자친구 유주", "여자친구 은하", "여자친구 신비",
    "르세라핌 김채원", "르세라핌 카즈하", "소녀시대 윤아", "소녀시대 태연",
    "트와이스 나연", "트와이스 사나", "트와이스 쯔위", "레드벨벳 아이린", "레드벨벳 슬기",
    "여자아이들 미연", "여자아이들 소연", "뉴진스 민지", "뉴진스 해린", "뉴진스 하니",
    "오마이걸 아린", "오마이걸 유아", "오마이걸 승희",
    "있지 예지", "있지 류진", "있지 유나", "스테이씨 윤", "스테이씨 시은",
    "프로미스나인 이나경", "프로미스나인 이새롬", "엔믹스 설윤", "엔믹스 해원",
    "여자아이들 우기", "여자아이들 슈화", "레드벨벳 조이", "레드벨벳 웬디",
    "르세라핌 허윤진", "권은비",
    ]
    
    
    # 각 인물에 대해 크롤링 실행
    for person in people_list:
        print(f"\n{'='*50}")
        print(f"{person} 이미지 크롤링 시작")
        print(f"{'='*50}")
        
        crawl_google_images(
            person_name=person,
            save_base_path=save_path,
            num_images=20
        )
        
        # 다음 검색 전 잠시 대기
        time.sleep(3)
    
    print("\n모든 크롤링 작업 완료!")

if __name__ == "__main__":
    main()
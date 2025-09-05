import errno
import json 
import cv2
import os
from tqdm import tqdm

# 📂 실제 데이터셋 경로
LABEL_DIR = "/home/work/conf/NIA/dataset/validation/02.라벨링"
IMG_DIR   = "/home/work/conf/NIA/dataset/validation/01.원천데이터"
CROP_DIR  = "/home/work/conf/NIA/dataset/validation/03.crop"

def mkdir(path):
    if path == "":
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# 🔁 라벨 디렉토리 순회
for equ in os.listdir(LABEL_DIR):   # 장치/카테고리 단위
    equ_path = os.path.join(LABEL_DIR, equ)
    for sub in tqdm(os.listdir(equ_path), desc=f"Processing {equ}"):
        sub_path = os.path.join(equ_path, sub)
        for anno_filename in os.listdir(sub_path):
            anno_f_path = os.path.join(sub_path, anno_filename)
            with open(anno_f_path, "r") as f:
                anno = json.load(f)
                
                # 원본 이미지 경로
                img_path = os.path.join(IMG_DIR, equ, sub, anno["info"]["filename"])
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[Warning] Image not found: {img_path}")
                    continue
                
                if anno["images"]["bbox"] is None:
                    continue
                
                bbox = list(map(int, anno["images"]["bbox"]))
                center_bbox = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                center_bbox = list(map(int, center_bbox))

                # 크롭 이미지 저장 경로
                save_dir = os.path.join(CROP_DIR, equ, sub)
                mkdir(save_dir)

                if anno["images"]["facepart"] == 0:
                    cropped_img = img
                else:
                    width = bbox[3] - bbox[1]
                    height = bbox[2] - bbox[0]
                    crop_length = int(max(width, height) / 2)

                    x1 = max(center_bbox[0] - crop_length, 0)
                    x2 = min(center_bbox[0] + crop_length, img.shape[1])
                    y1 = max(center_bbox[1] - crop_length, 0)
                    y2 = min(center_bbox[1] + crop_length, img.shape[0])

                    cropped_img = img[y1:y2, x1:x2]
                
                # 리사이즈 및 저장
                resized_img = cv2.resize(cropped_img, (256, 256))
                save_path = os.path.join(
                    save_dir,
                    anno["info"]["filename"][:-4] + f'_{str(anno["images"]["facepart"]).zfill(2)}.jpg'
                )
                cv2.imwrite(save_path, resized_img)

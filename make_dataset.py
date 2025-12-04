import os
import glob
import shutil
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ================= [사용자 설정 구역] =================
# 1. 외부 데이터 경로 (리사이징된 원본들)
SOURCE_TRAIN_DIR = r'D:\x-ray_data\resized_train_dataset'  # 학습용 폴더명 확인 필요
SOURCE_TEST_DIR = r'D:\x-ray_data\resized_eval_dataset'  # 평가용 폴더명 확인 필요

# 2. 프로젝트 내부에 생성될 최종 데이터셋 경로
DEST_ROOT = r'D:\x-ray\dataset'

# 3. 클래스 목록 (순서 중요! 이전과 동일하게 유지)
CLASSES = [
    'Aerosol', 'Alcohol', 'Bat', 'Battery', 'Bullet',
    'Electronic cigarettes', 'Gun', 'Hammer', 'HDD', 'Knife', 'LapTop', 'Lighter',
    'Liquid', 'NailClippers', 'SmartPhone', 'USB'
]


# ====================================================

def convert_box(size, box):
    """ XML 좌표 -> YOLO 좌표 변환 """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)


def process_file(xml_path, save_img_dir, save_lbl_dir):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        # 이미지 찾기 (XML과 같은 경로에 있다고 가정)
        base_path = os.path.splitext(xml_path)[0]
        image_found = None
        for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
            if os.path.exists(base_path + ext):
                image_found = base_path + ext
                break

        if image_found is None: return False

        # YOLO 라벨 생성
        yolo_lines = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in CLASSES: continue
            cls_id = CLASSES.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                 float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert_box((w, h), b)
            yolo_lines.append(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}")

        if not yolo_lines: return False

        # 파일 저장
        filename = os.path.basename(image_found)
        txt_filename = os.path.splitext(filename)[0] + '.txt'

        with open(os.path.join(save_lbl_dir, txt_filename), 'w') as f:
            f.write('\n'.join(yolo_lines))
        shutil.copy2(image_found, os.path.join(save_img_dir, filename))
        return True
    except Exception:
        return False


def main():
    # 폴더 초기화 (이미 있으면 삭제 후 재생성 방지하거나, 비우고 시작)
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(DEST_ROOT, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(DEST_ROOT, split, 'labels'), exist_ok=True)

    # 1. Train Data 처리 (Train:Valid = 9:1 분할)
    print("🚀 학습 데이터 처리 중...")
    train_xmls = glob.glob(os.path.join(SOURCE_TRAIN_DIR, '**', '*.xml'), recursive=True)
    random.shuffle(train_xmls)

    split_idx = int(len(train_xmls) * 0.9)
    train_set = train_xmls[:split_idx]
    valid_set = train_xmls[split_idx:]

    for xml in tqdm(train_set, desc="Train"):
        process_file(xml, os.path.join(DEST_ROOT, 'train/images'), os.path.join(DEST_ROOT, 'train/labels'))

    for xml in tqdm(valid_set, desc="Valid"):
        process_file(xml, os.path.join(DEST_ROOT, 'valid/images'), os.path.join(DEST_ROOT, 'valid/labels'))

    # 2. Test(Eval) Data 처리
    print("🚀 평가 데이터 처리 중...")
    test_xmls = glob.glob(os.path.join(SOURCE_TEST_DIR, '**', '*.xml'), recursive=True)
    for xml in tqdm(test_xmls, desc="Test"):
        process_file(xml, os.path.join(DEST_ROOT, 'test/images'), os.path.join(DEST_ROOT, 'test/labels'))

    print("\n✅ 데이터셋 준비 완료!")


if __name__ == '__main__':
    main()
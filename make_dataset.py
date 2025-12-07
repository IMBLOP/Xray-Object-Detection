import os
import glob
import shutil
import random
import xml.etree.ElementTree as ET

import cv2
from tqdm import tqdm

# ======================= [사용자 설정] =======================

# 리사이즈된 이미지 + XML이 들어있는 루트 폴더
SOURCE_DIR = r'D:\x-ray_data\resized_train_dataset'

# 최종 YOLO 데이터셋이 생성될 경로
DEST_ROOT = r'D:\x-ray\dataset_v2'  # 새 버전이라면 폴더 이름 다르게 추천

# train/valid/test 비율 (합이 1.0이 되도록)
SPLIT_RATIOS = {
    "train": 0.8,
    "valid": 0.1,
    "test": 0.1,
}

# 클래스 목록 (XML의 <name>과 동일한 문자열이어야 함)
# 필요에 따라 LapTop 빼거나, 이름 수정해서 사용하면 됨
CLASSES = [
    'Aerosol', 'Axe', 'Bat', 'Battery', 'Gun',
    'Hammer', 'HDD', 'Knife', 'LapTop', 'MetalPipe', 'Scissors',
    'SmartPhone', 'Spanner', 'TabletPC', 'USB'
]

# 랜덤 시드 (재현성을 위해 고정)
RANDOM_SEED = 42

# ============================================================


def ensure_dirs():
    """DEST_ROOT 안에 train/valid/test 하위 폴더 생성"""
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(DEST_ROOT, split, 'images')
        lbl_dir = os.path.join(DEST_ROOT, split, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)


def find_image_for_xml(xml_path):
    """XML과 같은 이름의 이미지를 찾는다 (확장자만 다를 수 있음)."""
    base_path = os.path.splitext(xml_path)[0]
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        img_path = base_path + ext
        if os.path.exists(img_path):
            return img_path
    return None


def voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    """VOC 좌표 -> YOLO 좌표 변환 (정규화된 cx, cy, w, h)"""
    # 혹시 모를 좌표 이상치 방지용 클램핑
    xmin = max(0, min(xmin, img_w - 1))
    xmax = max(0, min(xmax, img_w - 1))
    ymin = max(0, min(ymin, img_h - 1))
    ymax = max(0, min(ymax, img_h - 1))

    bw = xmax - xmin
    bh = ymax - ymin
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0

    return cx / img_w, cy / img_h, bw / img_w, bh / img_h


def process_one_xml(xml_path, img_save_dir, lbl_save_dir):
    """
    XML 1개를:
      - 이미지 실제 크기 기반으로 YOLO txt 생성
      - 이미지 복사
    """
    try:
        img_path = find_image_for_xml(xml_path)
        if img_path is None:
            return False

        img = cv2.imread(img_path)
        if img is None:
            return False

        img_h, img_w = img.shape[:2]

        tree = ET.parse(xml_path)
        root = tree.getroot()

        yolo_lines = []

        for obj in root.iter('object'):
            cls_name = obj.find('name').text.strip()
            if cls_name not in CLASSES:
                # 정의되지 않은 클래스는 스킵
                continue

            cls_id = CLASSES.index(cls_name)

            bnd = obj.find('bndbox')
            xmin = float(bnd.find('xmin').text)
            ymin = float(bnd.find('ymin').text)
            xmax = float(bnd.find('xmax').text)
            ymax = float(bnd.find('ymax').text)

            cx, cy, bw, bh = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)

            # 너무 작은/이상한 박스는 스킵 (원하면 조건 완화/삭제 가능)
            if bw <= 0 or bh <= 0:
                continue

            yolo_lines.append(
                f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            )

        if not yolo_lines:
            # 유효한 객체가 하나도 없으면 사용 X
            return False

        # 파일 이름은 이미지 파일 기준으로
        base_name = os.path.basename(img_path)
        txt_name = os.path.splitext(base_name)[0] + '.txt'

        # 라벨 저장
        lbl_out_path = os.path.join(lbl_save_dir, txt_name)
        with open(lbl_out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(yolo_lines))

        # 이미지 복사
        img_out_path = os.path.join(img_save_dir, base_name)
        if img_out_path != img_path:
            shutil.copy2(img_path, img_out_path)

        return True

    except Exception as e:
        # 디버깅 원하면 여기서 print(e) 추가
        return False


def main():
    random.seed(RANDOM_SEED)

    ensure_dirs()

    # 1) 전체 XML 파일 수집
    xml_files = glob.glob(os.path.join(SOURCE_DIR, '**', '*.xml'), recursive=True)
    xml_files = [x for x in xml_files if os.path.isfile(x)]

    print(f"총 XML 개수: {len(xml_files)}")

    if len(xml_files) == 0:
        print("❌ XML 파일이 없습니다. SOURCE_DIR 경로를 확인하세요.")
        return

    # 2) 셔플 후 train/valid/test 분할
    random.shuffle(xml_files)

    n_total = len(xml_files)
    n_train = int(n_total * SPLIT_RATIOS["train"])
    n_valid = int(n_total * SPLIT_RATIOS["valid"])
    # 나머지는 test
    n_test = n_total - n_train - n_valid

    train_xmls = xml_files[:n_train]
    valid_xmls = xml_files[n_train:n_train + n_valid]
    test_xmls = xml_files[n_train + n_valid:]

    print(f"분할 결과 -> train: {len(train_xmls)}, valid: {len(valid_xmls)}, test: {len(test_xmls)}")

    # 3) 각 split별로 처리
    stats = {
        "train": {"total": len(train_xmls), "ok": 0},
        "valid": {"total": len(valid_xmls), "ok": 0},
        "test": {"total": len(test_xmls), "ok": 0},
    }

    # --- train ---
    train_img_dir = os.path.join(DEST_ROOT, 'train', 'images')
    train_lbl_dir = os.path.join(DEST_ROOT, 'train', 'labels')
    for xml in tqdm(train_xmls, desc="Processing train"):
        if process_one_xml(xml, train_img_dir, train_lbl_dir):
            stats["train"]["ok"] += 1

    # --- valid ---
    valid_img_dir = os.path.join(DEST_ROOT, 'valid', 'images')
    valid_lbl_dir = os.path.join(DEST_ROOT, 'valid', 'labels')
    for xml in tqdm(valid_xmls, desc="Processing valid"):
        if process_one_xml(xml, valid_img_dir, valid_lbl_dir):
            stats["valid"]["ok"] += 1

    # --- test ---
    test_img_dir = os.path.join(DEST_ROOT, 'test', 'images')
    test_lbl_dir = os.path.join(DEST_ROOT, 'test', 'labels')
    for xml in tqdm(test_xmls, desc="Processing test"):
        if process_one_xml(xml, test_img_dir, test_lbl_dir):
            stats["test"]["ok"] += 1

    print("\n=========== 완료 요약 ==========")
    for split in ["train", "valid", "test"]:
        print(f"{split}: XML {stats[split]['total']}개 중 {stats[split]['ok']}개 사용됨")
    print(f"\n✅ 최종 YOLO 데이터셋 경로: {DEST_ROOT}")


if __name__ == '__main__':
    main()

import os
import cv2
import shutil
import pathlib
from tqdm import tqdm

# ================= 설정 =================
# 1. 원본 이미지 폴더 경로 (Smith 폴더가 시작점)
IMAGE_ROOT = r'D:\x-ray_data\smith'

# 2. 원본 XML 폴더 경로 (여기 안에 XML들이 쭉 있다고 하셨죠)
XML_ROOT = r'D:\x-ray_data\Annotation\train\Smith'

# 3. 결과물이 저장될 새로운 폴더 (이 폴더를 압축해서 서버로 보낼 겁니다)
OUTPUT_ROOT = r'D:\x-ray_data\resized_train_dataset'

# 4. 이미지 크기 및 압축 품질
IMG_SIZE = 640  # YOLO 학습용 크기
JPG_QUALITY = 85


# =======================================

def get_xml_map(xml_root_path):
    """
    XML 폴더를 미리 뒤져서 {파일명(확장자X): 전체경로} 형태의 사전을 만듭니다.
    Annotation 폴더 구조가 이미지와 달라도 파일명만 같으면 찾을 수 있게 합니다.
    """
    print("🔍 XML 파일 위치를 파악하는 중...", end='')
    xml_map = {}
    # 재귀적으로 모든 xml 검색
    for root, dirs, files in os.walk(xml_root_path):
        for file in files:
            if file.lower().endswith('.xml'):
                filename_no_ext = os.path.splitext(file)[0]
                xml_map[filename_no_ext] = os.path.join(root, file)
    print(f" 완료! (총 {len(xml_map)}개 XML 발견)")
    return xml_map


def main():
    # 1. XML 위치 매핑
    xml_mapping = get_xml_map(XML_ROOT)

    # 2. 이미지 폴더 순회
    print(f"🚀 이미지 리사이징 및 데이터 병합 시작...")

    # 처리할 이미지 확장자
    valid_ext = ['.png', '.jpg', '.jpeg', '.bmp']

    image_files = []
    for root, dirs, files in os.walk(IMAGE_ROOT):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_ext:
                image_files.append(os.path.join(root, file))

    success_count = 0
    fail_count = 0
    missing_xml_count = 0

    for img_path in tqdm(image_files):
        try:
            # 파일명 추출 (확장자 제외)
            file_name = os.path.basename(img_path)
            file_name_no_ext = os.path.splitext(file_name)[0]

            # 짝꿍 XML 찾기
            if file_name_no_ext not in xml_mapping:
                missing_xml_count += 1
                continue  # 라벨이 없으면 학습에 못 쓰니 건너뜁니다.

            src_xml_path = xml_mapping[file_name_no_ext]

            # 저장할 경로 생성 (IMAGE_ROOT 이하의 폴더 구조를 유지)
            # 예: D:\x-ray_data\Smith\Aerosol\Multiple... -> Aerosol\Multiple...
            rel_path = os.path.relpath(os.path.dirname(img_path), IMAGE_ROOT)
            save_dir = os.path.join(OUTPUT_ROOT, rel_path)
            os.makedirs(save_dir, exist_ok=True)

            # --- [이미지 처리] ---
            # 한글 경로 등으로 인한 오류 방지를 위해 numpy로 읽기
            import numpy as np
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                fail_count += 1
                continue

            # 리사이징 (비율 유지하면서 긴 변 기준 축소)
            h, w = img.shape[:2]
            scale = IMG_SIZE / max(h, w)
            if scale < 1:
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # 저장 (JPG로 변환하여 용량 최소화)
            save_img_name = file_name_no_ext + ".jpg"
            save_img_path = os.path.join(save_dir, save_img_name)

            # cv2.imwrite는 한글 경로 인식 못하므로 인코딩하여 저장
            result, encoded_img = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), JPG_QUALITY])
            if result:
                with open(save_img_path, mode='w+b') as f:
                    encoded_img.tofile(f)

            # --- [XML 처리] ---
            # 찾은 XML 파일을 이미지 바로 옆에 복사
            save_xml_path = os.path.join(save_dir, file_name_no_ext + ".xml")
            shutil.copy2(src_xml_path, save_xml_path)

            success_count += 1

        except Exception as e:
            print(f"Error: {img_path} - {e}")
            fail_count += 1

    print("\n" + "=" * 50)
    print(f"✅ 작업 완료!")
    print(f"총 처리 성공: {success_count}장")
    print(f"XML 매칭 실패(건너뜀): {missing_xml_count}장")
    print(f"이미지 읽기/저장 실패: {fail_count}장")
    print(f"저장된 폴더: {OUTPUT_ROOT}")
    print("=" * 50)
    print("👉 이제 생성된 폴더를 압축해서 서버로 전송하세요.")


if __name__ == '__main__':
    main()
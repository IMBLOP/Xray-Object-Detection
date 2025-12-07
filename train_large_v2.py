#train_and_eval.py
from ultralytics import YOLO
import torch
import os

def main():
    # --- 1. 설정 (사용자 수정 영역) ---
    # 로컬에서 FTP로 올린 data.yaml의 절대 경로 (지침서 p.8: /scratch 사용 권장) [cite: 237-238]
    DATA_PATH = os.path.abspath("/scratch/e1430a19/x-ray_project/data_v2.yaml") 
    
    PROJECT_NAME = "yolo_project"  # 결과가 저장될 폴더 이름
    RUN_NAME = "train_large_v2"          # 실행 이름
    MODEL_NAME = "yolov8l.pt"      # 사용할 모델 (n, s, m, l, x)
    
    EPOCHS = 100
    BATCH_SIZE = 16
    IMG_SIZE = 640

    # --- 2. GPU 설정 자동 감지 ---
    gpu_count = torch.cuda.device_count()
    devices = list(range(gpu_count))
    
    print(f"✅ Detected {gpu_count} GPUs. Training on devices: {devices}")
    print(f"✅ Data Path: {DATA_PATH}")

    # ==========================================
    # [STEP 1] 모델 학습 (Train)
    # ==========================================
    print("\n🚀 [STEP 1] Starting Training...")
    model = YOLO(MODEL_NAME)

    # 학습 시작
    model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=devices,      # 멀티 GPU 분산 학습
        project=PROJECT_NAME,
        name=RUN_NAME,
        exist_ok=True,       # 덮어쓰기 허용
        pretrained=True,
        workers=8,           # CPU 데이터 로더 프로세스 수
        val=True,             # Epoch마다 Validation 세트로 검증 수행 (기본값)
        copy_paste=0.3,      # 작은 물체를 복사해서 붙여넣기 (학습 기회 증대)
        mixup=0.1,           # 겹침 상황 학습
        degrees=10.0,        # 회전 (다양한 각도 학습)
        patience=50,         # 50 에폭 동안 성능 향상 없으면 조기 종료 (시간 절약)
        cos_lr=True          # Cosine Learning Rate Scheduler 사용 (수렴 안정성 향상)
    )
    print("✅ Training Finished.")

    # ==========================================
    # [STEP 2] 최종 성능 평가 (Test Evaluation)
    # ==========================================
    print("\n🚀 [STEP 2] Starting Final Evaluation on TEST set...")
    
    # 학습된 최적 모델 경로 (자동으로 best.pt가 생성됨)
    # 경로: yolo_project/train_v1/weights/best.pt
    best_model_path = os.path.join(PROJECT_NAME, RUN_NAME, "weights", "best.pt")
    
    if os.path.exists(best_model_path):
        # 최적 모델 다시 로드
        best_model = YOLO(best_model_path)
        
        # Test 데이터셋으로 평가 (split='test')
        # data.yaml에 'test:' 경로가 설정되어 있어야 합니다.
        metrics = best_model.val(
            data=DATA_PATH,
            split='test',    # ⭐️ 중요: Validation셋이 아닌 Test셋으로 최종 평가
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=devices,
            project=PROJECT_NAME,
            name=f"{RUN_NAME}_eval" # 결과는 yolo_project/train_v1_eval 에 저장됨
        )
        
        print("\n" + "="*40)
        print(f"🏆 Final Test Results (mAP):")
        print(f"   - mAP50    : {metrics.box.map50:.4f}")
        print(f"   - mAP50-95 : {metrics.box.map:.4f}")
        print("="*40 + "\n")
        
    else:
        print(f"⚠️ Error: Best model not found at {best_model_path}")

if __name__ == '__main__':
    main()

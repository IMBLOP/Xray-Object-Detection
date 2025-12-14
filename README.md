#  AI HUB 위해물품 X-ray 이미지 분석 모델

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)](https://github.com/ultralytics/ultralytics)

본 프로젝트는 AI HUB의 '위해물품 엑스레이 이미지' 데이터를 활용하여, 딥러닝 컴퓨터 비전 기술을 통해 X-ray 이미지 내 위해물품을 자동으로 탐지하는 지도학습(Supervised Learning) 기반 객체 탐지(Object Detection) 모델을 개발합니다.

---

##  목차

1.  [프로젝트 개요](#1-프로젝트-개요)
2.  [데이터](#2-데이터)
3.  [기술 스택 및 모델링](#3-기술-스택-및-모델링)
4.  [프로젝트 고도화 과정](#4-프로젝트-고도화-과정)
5.  [실험 결과 및 모델 성능](#5-실험-결과-및-모델-성능)
6.  [결론](#6-결론)

---

## 1. 프로젝트 개요

### 1.1. 문제 정의
X-ray 보안 검색은 현재 공항, 항만 등에서 수동 판독에 크게 의존하고 있습니다. 이로 인해 발생하는 판독자의 높은 피로도, 일관성 부족, 인적 오류 가능성은 보안 검색의 신뢰도를 저하시키는 주요 원인입니다.

### 1.2. 프로젝트 목표
* **정량적 목표:** AI HUB 위해물품 X-ray 데이터를 활용하여, **총 16종의 핵심 위해물품 및 저장매체**를 탐지하고 분류하며, 전체 클래스에 대해 **mAP 80% 이상**을 달성하는 자동 탐지 모델 개발.
* **정성적 목표:** 보안 검색 프로세스의 효율성 증대, 인적 오류 감소, 표준화된 판독 기준 제시.

### 1.3. 프로젝트 범위
* **In-Scope (포함 범위):**
    * AI HUB 데이터 전처리 (로컬 리사이징 및 포맷 변환)
    * 데이터 증강 파이프라인 구축
    * 객체 탐지 모델 학습 및 하이퍼파라미터 튜닝
    * 테스트셋을 이용한 정량적 성능 검증
* **Out-of-Scope (제외 범위):**
    * 실시간 X-ray 하드웨어 장비 연동
    * 실제 운영 환경 배포
    * 웹/모바일 애플리케이션 개발

### 1.4. 기대 효과
* 개발된 모델을 통해 보안 검색 요원의 판독 업무를 보조하여 피로도를 낮추고, 의심 영역을 선별적으로 제시함으로써 검색 효율과 정확도를 향상시킬 수 있습니다.
* 자동화된 탐지를 통해 일관성 있는 보안 검색 품질을 유지할 수 있습니다.

---

## 2. 데이터

### 2.1. 데이터 소스
* **[AI HUB - 위해물품 엑스레이 이미지 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EC%9C%84%ED%95%B4%EB%AC%BC%ED%92%88%20%EC%97%91%EC%8A%A4%EB%A0%88%EC%9D%B4%20%EC%9D%B4%EB%AF%B8%EC%A7%80&aihubDataSe=data&dataSetSn=233)**
<img width="1905" height="929" alt="대표도면_1" src="https://github.com/user-attachments/assets/8cd475c7-559e-4c70-8a0f-c2d6ff9d4d8c" />

* 데이터 구조
<img width="1280" height="720" alt="업로드_데이터셋_폴더구조_엠폴시스템_0" src="https://github.com/user-attachments/assets/bdf48e05-42bc-49e8-846d-ebaaaf23618c" />


* **본 데이터셋은 보안 검색 강화 및 X-ray 이미지 자동 판독 기술 개발을 위해 구축되었습니다.**

#### 데이터 선정 전략
본 프로젝트는 데이터의 일관성과 학습 효율성을 위해 **'Smiths Detection'** 장비의 데이터를 메인으로 사용하며, 총 16개의 핵심 클래스를 선별하여 학습을 진행합니다.

* **선별 클래스 (16 -> 15 Classes):**
> ~~Aerosol, Alcohol, Bat, Battery, Bullet, Electronic cigarettes, Gun, Hammer, HDD, Knife, LapTop, Lighter, Liquid, NailClippers, SmartPhone, USB~~
> Aerosol, Axe, Bat, Battery, Gun, Hammer, HDD, Knife, LapTop, MetalPipe, Scissors, SmartPhone, Spanner, TabletPC, USB

* **데이터 구조 (폴더):**
* `Single_Default`: 단일 물체, 배경 깨끗함 (기초 학습용)
* `Single_Other`: 단일 물체, 복잡한 배경 (실전 적응용)
* `Multiple_Categories`: 다중 물체 혼합 (심화 학습용)
* `Multiple_Other`: 다중 물체 + 복잡한 배경 (심화 학습용)

### 2.2. 데이터 전처리
대용량(약 250GB) 데이터를 효율적으로 처리하기 위해 **Two-Step 전처리 파이프라인**을 구축했습니다.

1. **Local Preprocessing (경량화):**
* 로컬 환경에서 원본 이미지를 `640px` (YOLO 입력 크기)로 리사이징하여 용량을 최적화.
* 폴더별로 흩어진 XML 어노테이션 파일을 이미지 경로에 맞춰 병합.
* 데이터 용량 최적화 후 서버 전송 (전송 효율 90% 이상 향상).
2. **Server Preprocessing (포맷 변환):**
* **Format Conversion:** Pascal VOC(`xml`) 형식을 YOLO(`txt`) 형식으로 변환.
* **Normalization:** 바운딩 박스 좌표를 0~1 사이의 상대 좌표로 정규화.
* **Data Split:** 전체 데이터를 `Train(90%)` / `Validation(10%)`로 계층적 분할.
* **Train Data:** AI-HUB에서 제공해주는 Eval Dataset 사용.

### 2.3. 데이터 입출력 정의
본 프로젝트 모델의 데이터 처리 흐름과 입출력 정보는 다음과 같습니다.

#### 입력 데이터
* **형태:** Smiths Detection 장비의 X-ray 스캔 이미지 (.jpg/.png, RGB 3채널).
* **특징:** 물체의 밀도에 따라 가짜 색상이 적용되어 있으며, 객체 간 겹침이 존재함.
* **모델 입력:** 전처리를 거쳐 640x640 픽셀로 리사이징 및 정규화된 텐서.

#### 출력 데이터
모델은 이미지 내 탐지된 모든 객체에 대해 다음 3가지 핵심 정보를 반환합니다.
1. **Bounding Box:** 탐지된 물체의 위치 좌표 $(x, y, w, h)$.
2. **Class Label:** 탐지된 물체의 종류 (타겟 클래스 중 하나).
3. **Confidence Score:** 해당 물체일 확률 (0.0 ~ 1.0 사이의 신뢰도).

---

## 3. 기술 스택 및 모델링

### 3.1. 핵심 기술
`Computer Vision` `Supervised Learning` `Object Detection`

### 3.2. 개발 환경
| 구분 | 기술/라이브러리 | 버전 | 비고 |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.9+ | |
| **Framework** | PyTorch | 1.12+ |  |
| **Object Detection** | Ultralytics YOLO | v8 | |
| **Core Libs** | OpenCV | 4.x | 이미지 처리 |
| **Infra** | **Neuron** | - | High-Performance Server Environment |

### 3.3. 사용 모델 및 동작 메커니즘

본 프로젝트는 실시간성과 정확도의 균형을 위해 **1-Stage Detector**인 **YOLOv8**을 베이스라인 모델로 선정하였습니다. 
특히 기존의 수동적인 딥러닝 구현 방식과 달리, 데이터 중심의 엔지니어링이 가능한 YOLOv8의 **동적 아키텍처 구성** 및 **최신 탐지 기술**을 적극 활용했습니다.

#### 모델 작동 방식 비교 (Traditional vs. YOLOv8)

| 특징 | 전통적 딥러닝 모델 | Ultralytics YOLOv8 |
| :--- | :--- | :--- |
| **모델 정의** | `class Net(nn.Module)` 등 코드 레벨에서 레이어를 직접 하드코딩하여 정의. | `*.pt` 파일 내의 아키텍처 설정(Config)을 파싱하여 **동적으로 모델을 조립**하고 구성. |
| **앵커 박스** | 데이터셋에 맞춰 K-means로 앵커 크기를 미리 계산하고 고정값으로 설정 (**Anchor-Based**). | 물체의 중심점(Center)과 거리를 예측하는 **Anchor-Free** 방식을 채택하여 기형적인 물체 탐지에 유연함. |
| **입출력 설계** | 입력 이미지 크기와 출력 클래스 수(`nc`)에 맞춰 모델의 입출력 레이어를 수동으로 수정. | `data.yaml`의 정보를 읽어 학습 시작 시 Head Layer를 자동으로 교체(Hot-swap)하고 입력을 동적 리사이징함. |
| **정답 할당** | 고정된 IoU 임계값(예: 0.5)을 넘으면 정답으로 간주 (**Static Assigner**). | 위치 정확도(IoU)와 클래스 확신도(Score)를 종합하여 학습 기여도가 높은 샘플을 동적으로 선별 (**Task Aligned Assigner**). |

#### YOLOv8 핵심 기술

본 프로젝트의 X-ray 데이터 특성(물체 겹침, 모호한 경계)을 극복하기 위해 YOLOv8의 다음 핵심 기술들이 적용되었습니다.

1.  **Decoupled Head (분리된 헤드 구조)**
    * 물체의 **종류**와 **위치**를 예측하는 연산을 분리하여 수행합니다. 이를 통해 각 태스크의 학습 충돌을 방지하고 수렴 속도와 정확도를 동시에 향상시켰습니다.

2.  **Distribution Focal Loss (DFL)**
    * X-ray 이미지 특성상 물체가 겹쳐 있어 경계선이 흐릿한 경우가 많습니다. YOLOv8은 박스의 좌표를 단일 값이 아닌 **확률 분포**로 예측하여, 경계가 모호한 물체에 대해서도 강건한 탐지 성능을 보여줍니다.

3.  **Mosaic & Mixup Augmentation**
    * 내장된 데이터 파이프라인을 통해 학습 시 이미지를 4장씩 합치거나(Mosaic), 서로 겹쳐서(Mixup) 입력합니다. 이는 다양한 배경과 겹침 상황을 시뮬레이션하여 모델의 일반화 성능을 극대화합니다.

**[Image of YOLOv8 architecture diagram](https://github.com/user-attachments/assets/76262e41-9e30-43ad-a78c-451468c6e934)**

---

## 4. 프로젝트 고도화 과정

본 프로젝트는 초기 베이스라인 모델(V0)의 한계를 분석하고, 데이터 분포 조정(V1)과 전처리 로직 실험(V2)을 거쳐, **모델 규모 확장 및 최적화(V3)**를 통해 최종 성능을 완성하였습니다.

### 4.1. 버전별 실험 요약

| 구분 | V0 (Baseline) | V1 (Distribution Fix) | V2 (Experiment) | **V3 (Final Optimization)** |
| :--- | :--- | :--- | :--- | :--- |
| **데이터 구성** | 기존 Split (불균형) | **전체 병합 후 8:1:1** | 8:1:1 (클래스 재구성) | **전체 병합 후 8:1:1** |
| **모델 규모** | YOLOv8m | YOLOv8m | YOLOv8m | **YOLOv8 Large (확장)** |
| **좌표 정규화** | XML 헤더 기준 | **XML 헤더 기준** | 실제 이미지 크기 (실패) | **XML 헤더 기준 (롤백)** |
| **결과 (mAP50)** | Test 35.5% | Test 99.2% | Test 92.0% | **Test 99.5% (SOTA)** |

### 4.2. 상세 진행 및 분석

#### V0: 초기 베이스라인
* **내용:** AI HUB 기본 데이터셋 구조를 그대로 사용하여 학습.
* **문제점:** 학습(Train) mAP는 99%에 달했으나, 평가(Test) mAP는 **35.5%로** 실사용 불가능 수준.
* **원인 분석:** Train 데이터와 Test 데이터 간의 **심각한 데이터 분포 불일치(Domain Shift)** 확인.

#### V1: 데이터 분포 재구성 및 증강 (Performance Leap)
* **해결 전략:**
* **데이터 재분할:** Test 데이터를 제외한 Train/Validation 데이터를 모두 합친 후, 무작위 셔플을 통해 **8 : 1 : 1 비율로 재분할**하여 데이터 분포의 균형을 맞춤.
* **증강 기법 도입:** X-ray 특성(겹침, 회전)을 반영하여 `CopyPaste(0.3)`, `Mixup(0.1)`, `Degrees(10.0)` 옵션 적용.
* **결과:**
* **Test mAP50 99.2% 달성**하며 과적합 문제를 해결하고 일반화 성능 확보.
* 데이터 파이프라인의 유효성을 입증했으나, 더 높은 정밀도 확보를 위해 추가 연구 진행.

#### V2: 전처리 정밀화 실험 (Experimental Failure)
* **목표:** '실제 이미지 크기'를 기반으로 좌표를 정규화하여 정밀도를 극대화하고자 함.
* **시도:** 리사이징 과정에서 XML 헤더 정보 대신, `cv2.imread().shape`를 사용하여 좌표 정규화 로직 변경.
* **결과 및 분석:**
* mAP50이 92.0%로 V1 대비 하락하였으며, 특히 **mAP50-95가 76.8%로 급락.**
* **원인:** 리사이징 시 발생하는 Padding과 좌표 계산 로직 간의 미세한 불일치(Coordinate Shift) 발생. 이로 인해 **USB(-53%), Battery(-27%) 등 소형 객체의 탐지 성능이 붕괴**됨을 확인.
* **교훈:** 전처리 로직을 안정적인 V1 방식(XML 메타데이터 기준)으로 롤백 결정.

#### V3: 최종 모델 최적화 및 확정 (Final Selection)
* **전략:**
* **전처리 롤백:** V2의 실패를 교훈 삼아 **XML 메타데이터 기반 절대 좌표 정규화** 방식을 확정 적용.
* **Model Scale-Up:** 복잡하게 겹친 물체와 소형 객체(USB, 배터리 등)의 탐지율을 극대화하기 위해 모델을 **YOLOv8 Large**로 확장.
* **Hyperparameter Tuning:** `Mosaic(1.0)` 및 `Mixup` 비율 최적화를 통해 겹침 상황(Occlusion) 대응력 강화.
* **최종 성과:**
* **mAP50 99.5%, mAP50-95 96.2%** 달성 (KPI 초과 달성).
* 소형 객체(USB, Battery) 및 겹친 물체에 대해서도 완벽에 가까운 탐지 성능을 보여 **최종 배포 모델로 선정.**

---

## 5. 실험 결과 및 모델 성능

본 프로젝트는 V0(베이스라인), V1(데이터 분포 개선), V2(전처리 실험) 단계를 거쳐, 최종적으로 모델 규모를 확장하고 전처리 로직을 최적화한 **V3 모델**을 최종 선정하였습니다.

### 5.1. 정량적 평가 
최종 선정된 **V3 모델(YOLOv8l + Augmentation + Re-split)** 의 테스트 데이터셋 평가 결과입니다.

**[전체 성능 요약]**

| Model Version | mAP@50 | mAP@50-95 | 비고 |
| :--- | :--- | :--- | :--- |
| **V0 (Baseline)** | 35.5% | 27.1% | 데이터 분포 불일치로 인한 실패 |
| **V1 (Distribution Fix)** | 99.2% | 92.8% | 일반화 성능 확보 성공 |
| **V2 (Experimental)** | 92.0% | 76.8% | 소형 객체 탐지 성능 저하 (기각) |
| **V3 (Final Selected)** | **99.5%** | **96.2%** | **Large 모델 도입으로 정밀도(Precision) 극대화** |

**[성능 분석 시각화]**

| Metric | Graph | 분석 |
| :---: | :---: | :--- |
| **Class별 정확도** | <img src="https://github.com/user-attachments/assets/050b45f1-dfc6-4759-806a-bc90b8930803" width="400" alt="Confusion Matrix"> | **Normalized Confusion Matrix:**<br>대부분의 클래스에서 대각선(정답)이 **1.00에 근접**하며, V1 대비 클래스 간 오분류가 더욱 감소하여 완벽에 가까운 분류 성능을 보입니다. |
| **최적 임계값** | <img src="https://github.com/user-attachments/assets/65257a46-ffad-4976-9c81-ced07db16eb6" width="400" alt="F1 Curve"> | **F1-Confidence Curve:**<br>Confidence Score 0.5 부근에서 모든 클래스의 F1 Score가 최고점에 도달하며, 이는 모델이 예측에 대해 매우 높은 확신을 가지고 있음을 의미합니다. |
| **탐지 신뢰성** | <img src="https://github.com/user-attachments/assets/d59b5fde-a310-461a-a127-a84d6c1a950d" width="400" alt="PR Curve"> | **Precision-Recall Curve:**<br>mAP@50 **99.5%**라는 수치가 보여주듯, PR 곡선이 우상단 모서리에 완벽하게 밀착되어 있어 False Positive와 False Negative가 거의 없습니다. |

**[V3 모델 클래스별 성능 상세]**
YOLOv8 Large 모델 적용 결과, 대형 물체뿐만 아니라 V2에서 문제가 되었던 소형 물체까지 완벽하게 탐지해냈습니다.

* **Perfect Detection (100% ~ 99.7%):**
* `Gun` (**1.00**), `TabletPC` (**0.997**), `HDD` (**0.999**), `Aerosol` (**0.998**)
* 형태가 뚜렷하고 특징적인 물체들은 오차가 전혀 없는 완벽한 탐지율을 기록했습니다.

* **Small Objects Optimization:**
* `USB` (**0.996**), `Battery` (**0.998**), `Lighter` (**0.992**)
* 육안 식별이 어려운 초소형 객체들도 V3의 고해상도 특징 추출 능력 덕분에 99% 이상의 탐지율을 달성했습니다.

* **Precision Improvement:**
* `Battery`의 경우 mAP50은 99.8%이나, mAP50-95(정밀도)는 **0.90**을 기록했습니다. 이는 물체는 확실히 찾지만(Find), 박스의 크기를 아주 미세하게 조정하는 데(Fit) 약간의 여지가 있음을 의미하나, 실사용에는 전혀 무리가 없는 수준입니다.

### 5.2. 정성적 평가 
테스트 데이터셋에 대한 실제 모델 추론(Inference) 결과 시각화입니다.
복잡하게 겹쳐진 수하물 이미지 내에서도 물체의 종류와 위치를 정확히 식별합니다.
![val_batch0_pred](https://github.com/user-attachments/assets/fc59b179-c277-4f8e-ac44-ab1966163a3d)
![val_batch1_pred](https://github.com/user-attachments/assets/a801b9af-ddfa-42c3-a423-60b660a535e2)

---

## 6. 결론

### 6.1. 프로젝트 성과 요약
* **데이터 중심의 성능 혁신:** 초기 모델(V0)의 실패 원인을 '데이터 분포의 불균형(Domain Shift)'에서 정확히 진단하고, 재분할 전략과 전처리 최적화를 통해 성능을 **35.5% → 99.5%로** 극대화했습니다.
* **기술적 난제 해결:** 소형 객체(USB, 배터리)의 탐지 성능 저하 원인이 '좌표 연산 오차(Coordinate Shift)'임을 실험(V2)을 통해 규명하고, **XML 메타데이터 기반 정규화**와 **Large 모델 도입(V3)**으로 이를 완벽하게 극복했습니다.
* **고신뢰성 모델 확보:** 최종 mAP50-95(정밀도) **96.2%**를 달성하여, 현업 보안 검색대에 즉시 투입 가능한 수준의 정밀한 AI 모델을 구축했습니다.

### 6.2. 한계점 및 향후 과제
* **모델의 연산 비용:** 탐지 정확도를 극대화하기 위해 **YOLOv8 Large** 모델을 채택함에 따라, Small/Nano 모델 대비 추론 속도와 메모리 사용량이 증가하였습니다. 저사양 엣지 디바이스 탑재를 위해서는 최적화가 필요합니다.
* **장비 특성 종속성:** 현재 모델은 Smiths Detection 장비 데이터에 고도로 최적화되어 있습니다. 타사(Rapiscan, L3 등) 장비의 X-ray 이미지에 적용할 경우 도메인 적응 과정이 필요할 수 있습니다.
* **향후 계획:**
* **모델 경량화:** 현재 확보한 고성능 Large 모델의 지식을 작은 모델에 전이하는 **지식 증류** 기법을 적용하여, 성능은 유지하되 속도를 높이는 연구 진행.
* **멀티 도메인 확장:** 다양한 제조사의 X-ray 장비 데이터를 추가 수집하여, 장비 종류에 상관없이 범용적으로 작동하는 **Global Model**로 고도화.

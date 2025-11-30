#  AI HUB 위해물품 X-ray 이미지 분석 모델

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org/)

본 프로젝트는 AI HUB의 '위해물품 엑스레이 이미지' 데이터를 활용하여, 딥러닝 컴퓨터 비전 기술을 통해 X-ray 이미지 내 위해물품을 자동으로 탐지하는 지도학습(Supervised Learning) 기반 객체 탐지(Object Detection) 모델을 개발합니다.

---

##  목차 (Table of Contents)

1.  [프로젝트 개요](#1-프로젝트-개요)
2.  [데이터](#2-데이터)
3.  [기술 스택 및 모델링](#3-기술-스택-및-모델링)

---

## 1. 프로젝트 개요

### 1.1. 문제 정의
X-ray 보안 검색은 현재 공항, 항만 등에서 수동 판독에 크게 의존하고 있습니다. 이로 인해 발생하는 판독자의 높은 피로도, 일관성 부족, 인적 오류(Human Error) 가능성은 보안 검색의 신뢰도를 저하시키는 주요 원인입니다.

### 1.2. 프로젝트 목표
* **정량적 목표:** AI HUB 위해물품 X-ray 데이터를 활용하여, **총 18종의 핵심 위해물품 및 저장매체**를 탐지하고 분류하며, 전체 클래스에 대해 **mAP 80% 이상**을 달성하는 자동 탐지 모델 개발.
* **정성적 목표:** 보안 검색 프로세스의 효율성 증대, 인적 오류 감소, 표준화된 판독 기준 제시.

### 1.3. 프로젝트 범위
* **In-Scope (포함 범위):**
    * AI HUB 데이터 전처리 (로컬 리사이징 및 포맷 변환)
    * 데이터 증강(Augmentation) 파이프라인 구축
    * 객체 탐지 모델 학습 및 하이퍼파라미터 튜닝
    * 테스트셋을 이용한 정량적 성능 검증
* **Out-of-Scope (제외 범위):**
    * 실시간 X-ray 하드웨어 장비 연동
    * 실제 운영 환경 배포 (SaaS/On-premise)
    * 웹/모바일 애플리케이션 개발

### 1.4. 기대 효과
* 개발된 모델을 통해 보안 검색 요원의 판독 업무를 보조하여 피로도를 낮추고, 의심 영역을 선별적으로 제시함으로써 검색 효율과 정확도를 향상시킬 수 있습니다.
* 자동화된 탐지를 통해 일관성 있는 보안 검색 품질을 유지할 수 있습니다.

---

## 2. 데이터

### 2.1. 데이터 소스
* **[AI HUB - 위해물품 엑스레이(X-ray) 이미지 데이터](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&searchKeyword=%EC%9C%84%ED%95%B4%EB%AC%BC%ED%92%88%20%EC%97%91%EC%8A%A4%EB%A0%88%EC%9D%B4%20%EC%9D%B4%EB%AF%B8%EC%A7%80&aihubDataSe=data&dataSetSn=233)**
<img width="1905" height="929" alt="대표도면_1" src="https://github.com/user-attachments/assets/8cd475c7-559e-4c70-8a0f-c2d6ff9d4d8c" />

* 데이터 구조
<img width="1280" height="720" alt="업로드_데이터셋_폴더구조_엠폴시스템_0" src="https://github.com/user-attachments/assets/bdf48e05-42bc-49e8-846d-ebaaaf23618c" />


* **본 데이터셋은 보안 검색 강화 및 X-ray 이미지 자동 판독 기술 개발을 위해 구축되었습니다.**

#### 데이터 선정 전략 (Data Selection)
본 프로젝트는 데이터의 일관성과 학습 효율성을 위해 **'Smiths Detection'** 장비의 데이터를 메인으로 사용하며, 총 18개의 핵심 클래스를 선별하여 학습을 진행합니다.

* **선별 클래스 (18 Classes):**
> Aerosol, Alcohol, Battery, SSD, Axe, Bat, Bullet, Electronic cigarettes, Gun, Hammer, Knife, LapTop, Lighter, Liquid, NailClippers, SmartPhone, SupplymentaryBattery, TabletPC

* **데이터 구조 (폴더):**
* `Single_Default`: 단일 물체, 배경 깨끗함 (기초 학습용)
* `Single_Other`: 단일 물체, 복잡한 배경 (실전 적응용)
* `Multiple_Categories`: 다중 물체 혼합 (심화 학습용)
* `Multiple_Other`: 다중 물체 + 복잡한 배경 (최종 성능 평가용)

### 2.2. 데이터 전처리 (Preprocessing)
대용량(약 250GB) 데이터를 효율적으로 처리하기 위해 **Two-Step 전처리 파이프라인**을 구축했습니다.

1. **Local Preprocessing (경량화):**
* 로컬 환경에서 원본 이미지를 `640px` (YOLO 입력 크기)로 리사이징하여 용량을 최적화.
* 폴더별로 흩어진 XML 어노테이션 파일을 이미지 경로에 맞춰 병합.
* 데이터 용량 최적화 후 서버 전송 (전송 효율 90% 이상 향상).
2. **Server Preprocessing (포맷 변환):**
* **Format Conversion:** Pascal VOC(`xml`) 형식을 YOLO(`txt`) 형식으로 변환.
* **Normalization:** 바운딩 박스 좌표를 0~1 사이의 상대 좌표로 정규화.
* **Data Split:** 전체 데이터를 `Train(90%)` / `Validation(10%)`로 계층적 분할(Stratified Split).

---

## 3. 기술 스택 및 모델링

### 3.1. 핵심 기술
`Computer Vision` `Supervised Learning` `Object Detection`

### 3.2. 개발 환경
| 구분 | 기술/라이브러리 | 버전 | 비고 |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.9+ | |
| **Framework** | PyTorch | 1.12+ | 
| **Object Detection** | Ultralytics YOLO | v8 | |
| **Core Libs** | OpenCV | 4.x | 이미지 처리 |
| **Infra** | **Neuron** | - | High-Performance Server Environment |

### 3.3. 사용 모델

본 프로젝트는 실시간성과 정확도의 균형을 위해 **1-Stage Detector**인 **YOLOv8**을 베이스라인 모델로 선정하여 학습을 진행합니다.

* **모델 선정 이유:**
* 보안 검색 환경 특성상 빠른 추론 속도(Real-time Inference)가 필수적임.
* YOLOv8은 이전 버전 대비 작은 물체(Small Object) 탐지 성능이 개선되어, X-ray 내의 작은 위해물품(라이터, 총알 등) 탐지에 유리할 것으로 판단됨.

---

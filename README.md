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
* **정량적 목표:** AI HUB 위해물품 X-ray 데이터를 활용하여, **총 34종의 품목 (위해물품 27종, 저장매체 7종)**을 탐지하고 분류하며, 전체 클래스에 대해 mAP 80% 이상 (또는 프로젝트 목표치)을 달성하는 자동 탐지 모델 개발.
* **정성적 목표:** 보안 검색 프로세스의 효율성 증대, 인적 오류 감소, 표준화된 판독 기준 제시.

### 1.3. 프로젝트 범위
* **In-Scope (포함 범위):**
    * AI HUB 데이터 전처리 및 분석 (EDA)
    * 데이터 증강(Augmentation) 파이프라인 구축
    * 객체 탐지 모델(YOLO, Faster R-CNN 등) 학습 및 하이퍼파라미터 튜닝
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

### 2.2. 데이터 분석 (EDA)
* **주요 분석 항목:**
    * 이미지 해상도 및 크기 분포
    * 채널 정보 (흑백/컬러)
    * 위해물품 클래스별 샘플 수 (클래스 불균형 확인)
    * 바운딩 박스 크기 및 종횡비(Aspect Ratio) 분석

### 2.3. 데이터 전처리
* **이미지 정규화:** 이미지 크기 통일(Resizing) 및 픽셀 값 정규화(Normalization).
* **데이터 증강 (Augmentation):** 밝기/대비 조절, 회전, 이동 (X-ray 특성 고려).

---

## 3. 기술 스택 및 모델링

### 3.1. 핵심 기술
`Computer Vision` `Supervised Learning` `Object Detection`

### 3.2. 개발 환경
| 구분 | 기술/라이브러리 | 버전 | 비고 |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.9+ | |
| **Framework** | PyTorch | 1.12+ | 
| **Object Detection** | MMDetection | - |  |
| **Core Libs** | OpenCV | 4.x | 이미지 처리 |
| **Infra** | Google Colab | T4 | (CUDA 11.x) |

### 3.3. 사용 모델


본 프로젝트는 속도와 정확도의 트레이드오프(Trade-off)를 고려하여 1-Stage와 2-Stage 모델을 모두 실험하고 비교합니다.

* **1-Stage Detector (속도 중심):**
    * **YOLO (You Only Look Once)**: 특히 **YOLOv8** 모델을 사용하여 빠른 추론 속도와 준수한 정확도를 목표로 합니다. 실시간성이 요구되는 환경에 적합합니다.
* **2-Stage Detector (정확도 중심):**
    * **Faster R-CNN**: 복잡하게 겹쳐진(Occluded) 물체나 크기가 매우 작은 물체를 정밀하게 탐지해야 할 경우, 높은 정확도를 보장하기 위해 사용합니다.

---

# bbox 이미지 SAM 세그멘테이션 가이드

## 📋 개요
기존에 저장된 bbox 크롭 이미지들을 SAM(Segment Anything Model)으로 세그멘테이션하여 배경을 제거한 누끼따기 이미지를 생성합니다. 마네킹 합성을 위한 투명 배경 이미지를 준비합니다.

## 🎯 목적
- **배경 제거**: bbox 크롭 이미지의 배경을 정확히 제거
- **마네킹 합성 준비**: 투명 배경 이미지로 마네킹과 합성 가능
- **일괄 처리**: 기존 bbox 이미지들을 자동으로 일괄 처리

## 🗂️ 파일 구조

### 입력 구조
```
API/processed_images/
├── top/                   # 상의 bbox 이미지들
│   ├── item_181_top.jpg
│   ├── item_178_top.jpg
│   └── ...
├── bottom/                # 하의 bbox 이미지들
│   ├── item_181_bottom.jpg
│   └── ...
├── outer/                 # 아우터 bbox 이미지들
│   ├── item_26_outer.jpg
│   └── ...
└── dress/                 # 드레스 bbox 이미지들
    └── ...
```

### 출력 구조
```
segmented_bbox_images/
├── top_segmented/         # 상의 누끼따기 결과
│   ├── item_181_top_segmented.png
│   ├── item_178_top_segmented.png
│   └── ...
├── bottom_segmented/      # 하의 누끼따기 결과
│   ├── item_181_bottom_segmented.png
│   └── ...
├── outer_segmented/       # 아우터 누끼따기 결과
│   ├── item_26_outer_segmented.png
│   └── ...
├── dress_segmented/       # 드레스 누끼따기 결과
│   └── ...
├── segmentation_report.json    # 처리 결과 요약
└── detailed_results.json       # 상세 처리 결과
```

## 🚀 실행 방법

### 1. 환경 설정
```bash
# segment-anything 설치
pip install git+https://github.com/facebookresearch/segment-anything.git

# 기타 필요한 패키지
pip install torch torchvision opencv-python pillow matplotlib tqdm
```

### 2. 실행
```bash
# 방법 1: 직접 실행
cd c
python 13_bbox_sam_segmentation.py

# 방법 2: 래퍼 스크립트 사용
cd c
python run_bbox_sam.py
```

## 🔧 주요 기능

### 1. 자동 bbox 세그멘테이션
```python
# bbox 이미지 전체를 객체로 인식
bbox = [0, 0, width, height]  # 전체 이미지 영역

# 중앙점을 포그라운드 포인트로 사용
center_point = [width//2, height//2]

# SAM으로 정밀 세그멘테이션
mask, score = sam.predict(box=bbox, point_coords=center_point)
```

### 2. 투명 배경 생성
- **PNG 형식**: 알파 채널 지원
- **RGBA**: 완전한 투명도
- **고품질**: 원본 해상도 유지

### 3. 카테고리별 일괄 처리
- **top**: 상의 이미지들
- **bottom**: 하의 이미지들
- **outer**: 아우터 이미지들
- **dress**: 드레스 이미지들

## 📊 처리 결과

### 1. 파일명 규칙
```
{원본파일명}_segmented.png
예: item_181_top.jpg -> item_181_top_segmented.png
```

### 2. 요약 리포트 (segmentation_report.json)
```json
{
  "summary": {
    "total_files": 150,
    "successful_files": 145,
    "failed_files": 5,
    "success_rate": 96.7
  },
  "category_stats": {
    "top": {
      "total": 50,
      "success": 48,
      "failed": 2,
      "success_rate": 96.0,
      "average_mask_score": 0.892
    }
  }
}
```

### 3. 상세 결과 (detailed_results.json)
```json
[
  {
    "input_path": "API/processed_images/top/item_181_top.jpg",
    "output_path": "segmented_bbox_images/top_segmented/item_181_top_segmented.png",
    "category": "top",
    "mask_score": 0.95,
    "success": true
  }
]
```

## ⚙️ 설정 옵션

### 1. SAM 모델 타입
```python
model_type = "vit_h"  # vit_h (정확), vit_l (균형), vit_b (빠름)
```

### 2. 중앙점 사용 여부
```python
use_center_point = True  # 중앙점을 포그라운드로 사용
```

### 3. 투명도 설정
```python
alpha = 1.0  # 투명도 (0.0 ~ 1.0)
```

## 🎨 마네킹 합성 활용

### 1. 합성 준비
- **투명 배경**: 마네킹 이미지와 자연스러운 합성
- **고품질**: 원본 해상도 유지
- **일관성**: 모든 의류 아이템 동일한 처리

### 2. 합성 예시
```python
# 마네킹 이미지 로드
mannequin = cv2.imread("mannequin.png", cv2.IMREAD_UNCHANGED)

# 세그멘테이션된 의류 로드
clothing = cv2.imread("item_181_top_segmented.png", cv2.IMREAD_UNCHANGED)

# 알파 블렌딩으로 합성
result = blend_with_alpha(mannequin, clothing)
```

### 3. 활용 방안
- **가상 피팅룸**: 사용자 아바타에 의류 합성
- **상품 이미지**: 마네킹 모델에 의류 착용
- **스타일 시뮬레이션**: 다양한 의류 조합 시각화

## 📈 성능 최적화

### 1. GPU 사용
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### 2. 배치 처리
- 이미지별 순차 처리
- 메모리 효율적 관리
- 진행률 표시

### 3. 품질 설정
- **vit_h**: 가장 정확한 세그멘테이션
- **중앙점 사용**: 더 정확한 객체 인식
- **멀티마스크**: 최적 마스크 자동 선택

## 🔍 문제 해결

### 1. segment-anything 설치 오류
```bash
# PyTorch 버전 확인
pip install torch torchvision

# segment-anything 재설치
pip uninstall segment-anything
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 2. CUDA 메모리 부족
```python
# CPU 사용으로 변경
device = "cpu"

# 또는 이미지 크기 줄이기
image = cv2.resize(image, (512, 512))
```

### 3. 세그멘테이션 품질 문제
```python
# 중앙점 사용 활성화
use_center_point = True

# 더 큰 모델 사용
model_type = "vit_h"
```

## 📝 주의사항

1. **파일 형식**: 입력은 JPG, 출력은 PNG (투명도 지원)
2. **메모리 사용**: 큰 이미지 처리 시 GPU 메모리 고려
3. **처리 시간**: 이미지 수와 크기에 따라 시간 소요
4. **품질**: 복잡한 배경일수록 세그멘테이션 품질 향상

## 🎯 다음 단계

1. **마네킹 합성**: 투명 배경 이미지를 마네킹과 합성
2. **포즈 변환**: 다양한 포즈의 마네킹에 적용
3. **스타일 조합**: 여러 의류 아이템 조합
4. **실시간 합성**: 웹/앱에서 실시간 합성 기능

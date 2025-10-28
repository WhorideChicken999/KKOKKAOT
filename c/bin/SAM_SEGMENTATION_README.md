# SAM 세그멘테이션 가이드

## 📋 개요
YOLO로 감지된 bbox 영역을 SAM(Segment Anything Model)으로 정확히 세그멘테이션하여 배경을 제거한 누끼따기 이미지를 생성합니다.

## 🎯 목적
- **배경 제거**: YOLO bbox 영역을 정확히 세그멘테이션하여 배경 제거
- **합성 준비**: 추후 이미지 합성을 위한 투명 배경 이미지 생성
- **정확한 마스킹**: SAM의 고성능 세그멘테이션으로 정밀한 객체 분리

## 🗂️ 파일 구조
```
API/pre_trained_weights/
├── yolo_best.pt          # YOLO 모델 (4개 카테고리: top, bottom, outer, dress)
└── sam_best.pt           # SAM 모델 (세그멘테이션)

segmented_images/         # 출력 디렉토리
├── all_images/           # converted_data/all_images 처리 결과
├── uploaded_images/      # API/uploaded_images 처리 결과
└── default_items/        # API/default_items 처리 결과
    ├── image1_top_0.png
    ├── image1_bottom_1.png
    └── segmentation_results.json
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
python 12_sam_segmentation.py

# 방법 2: 래퍼 스크립트 사용
cd c
python run_sam_segmentation.py
```

## 🔧 주요 기능

### 1. YOLO + SAM 파이프라인
```python
# 1단계: YOLO로 객체 감지
detections = yolo_model(image)
# 결과: bbox 좌표, 클래스, 신뢰도

# 2단계: SAM으로 정밀 세그멘테이션
mask, score = sam_segmenter.segment_bbox(image, bbox)
# 결과: 정확한 객체 마스크

# 3단계: 투명 배경 이미지 생성
transparent_image = create_transparent_image(image, mask)
```

### 2. 클래스별 처리
- **top**: 상의 (티셔츠, 셔츠, 블라우스 등)
- **bottom**: 하의 (청바지, 스커트, 반바지 등)
- **outer**: 아우터 (자켓, 코트, 가디건 등)
- **dress**: 드레스 (원피스, 원피스 등)

### 3. 출력 형식
- **PNG**: 투명 배경 지원
- **RGBA**: 알파 채널 포함
- **고품질**: 원본 해상도 유지

## 📊 처리 결과

### 1. 파일명 규칙
```
{원본이미지명}_{클래스명}_{인덱스}.png
예: 1000005_top_0.png, 1000005_bottom_1.png
```

### 2. 결과 JSON
```json
{
  "class_name": "top",
  "class_id": 0,
  "confidence": 0.95,
  "bbox": [100, 200, 300, 400],
  "mask_score": 0.98,
  "output_path": "segmented_images/all_images/1000005_top_0.png"
}
```

### 3. 통계 정보
- 처리된 이미지 수
- 생성된 세그멘테이션 수
- 클래스별 통계
- 평균 신뢰도

## ⚙️ 설정 옵션

### 1. 신뢰도 임계값
```python
confidence_threshold = 0.3  # YOLO 감지 신뢰도
```

### 2. SAM 모델 타입
```python
sam_model_type = "vit_h"  # vit_h, vit_l, vit_b
```

### 3. 출력 형식
```python
format = "PNG"  # PNG (투명), JPG (검은 배경)
```

## 🎨 사용 예시

### 1. 단일 이미지 처리
```python
from c.sam_segmentation import YOLOSAMProcessor

# 프로세서 초기화
processor = YOLOSAMProcessor(
    yolo_model_path="API/pre_trained_weights/yolo_best.pt",
    sam_model_path="API/pre_trained_weights/sam_best.pt"
)

# 단일 이미지 처리
results = processor.detect_and_segment(
    "path/to/image.jpg",
    "output/directory",
    confidence_threshold=0.3
)
```

### 2. 배치 처리
```python
# 여러 이미지 일괄 처리
results = processor.process_batch(
    "input/directory",
    "output/directory",
    confidence_threshold=0.3
)
```

### 3. 커스텀 세그멘테이션
```python
from c.sam_segmentation import SAMSegmentation

# SAM만 사용
sam = SAMSegmentation("path/to/sam_model.pt")

# bbox로 세그멘테이션
mask, score = sam.segment_bbox(image, bbox)

# 투명 이미지 생성
transparent = sam.create_transparent_image(image, mask)
```

## 📈 성능 최적화

### 1. GPU 사용
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### 2. 배치 크기 조정
- GPU 메모리에 따라 조정
- 일반적으로 1-4개 이미지 동시 처리

### 3. 모델 선택
- **vit_h**: 가장 정확하지만 느림
- **vit_l**: 균형잡힌 성능
- **vit_b**: 빠르지만 정확도 낮음

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

# 또는 배치 크기 줄이기
batch_size = 1
```

### 3. 모델 로딩 실패
```python
# 모델 경로 확인
yolo_path = "API/pre_trained_weights/yolo_best.pt"
sam_path = "API/pre_trained_weights/sam_best.pt"

# 파일 존재 여부 확인
assert Path(yolo_path).exists(), "YOLO 모델을 찾을 수 없습니다"
assert Path(sam_path).exists(), "SAM 모델을 찾을 수 없습니다"
```

## 🎯 활용 방안

### 1. 이미지 합성
- 배경 제거된 의류 이미지
- 새로운 배경과 합성
- 가상 피팅룸 구현

### 2. 데이터 증강
- 다양한 배경과 합성
- 스타일 변환
- 포즈 변경

### 3. 상품 이미지 처리
- 쇼핑몰 상품 이미지
- 일관된 배경 제거
- 브랜드 이미지 통일

## 📝 주의사항

1. **라이선스**: SAM 모델 사용 시 Facebook의 라이선스 확인
2. **저작권**: 처리된 이미지의 저작권 고려
3. **품질**: 세그멘테이션 품질이 합성 결과에 직접 영향
4. **저장공간**: PNG 파일은 JPG보다 용량이 큼

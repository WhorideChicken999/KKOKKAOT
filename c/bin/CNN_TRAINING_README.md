# CNN 라벨 학습 가이드

## 📋 개요
`converted_data/cnn/` 폴더의 JSON 라벨과 `converted_data/all_images/` 폴더의 이미지를 사용하여 패션 속성 분류 모델을 학습합니다.

## 🗂️ 데이터 구조
```
converted_data/
├── cnn/                    # CNN 라벨 JSON 파일들
│   ├── cnn_1000005.json
│   ├── cnn_1000015.json
│   └── ...
└── all_images/             # 이미지 파일들
    ├── 1000005.jpg
    ├── 1000015.jpg
    └── ...
```

## 📊 JSON 라벨 구조
```json
{
  "image_id": 1000005,
  "file_name": "1000005.jpg",
  "items": {
    "상의": {
      "카테고리": "티셔츠",
      "색상": "화이트",
      "핏": "오버사이즈",
      "소재": ["저지"],
      "프린트": ["레터링"],
      "디테일": ["드롭숄더"]
    },
    "하의": {
      "카테고리": "청바지",
      "색상": "네이비",
      "핏": "노멀",
      "소재": ["데님"],
      "프린트": ["무지"],
      "디테일": ["포켓", "디스트로이드"]
    }
  }
}
```

## 🚀 실행 방법

### 1. 환경 설정
```bash
# 필요한 패키지 설치
pip install torch torchvision torchaudio
pip install scikit-learn matplotlib seaborn tqdm pillow pandas numpy
```

### 2. 학습 실행
```bash
# 방법 1: 직접 실행
cd c
python 11_cnn_label_training.py

# 방법 2: 래퍼 스크립트 사용
cd c
python run_cnn_training.py
```

## 🎯 학습 대상 속성
- **상의_category**: 티셔츠, 셔츠, 블라우스 등
- **하의_category**: 청바지, 스커트, 반바지 등
- **상의_color**: 화이트, 블랙, 네이비 등
- **하의_color**: 화이트, 블랙, 네이비 등
- **상의_fit**: 노멀, 타이트, 오버사이즈 등
- **하의_fit**: 노멀, 타이트, 스키니 등

## 📈 모델 아키텍처
- **백본**: ResNet50 (ImageNet 사전 훈련)
- **분류기**: 512차원 → 클래스 수
- **정규화**: Dropout (0.5, 0.3)
- **옵티마이저**: Adam (lr=0.001, weight_decay=1e-4)
- **스케줄러**: StepLR (step_size=20, gamma=0.1)

## 📊 출력 결과
```
c/trained_models/
├── best_model.pth          # 최고 성능 모델
├── model_info.json         # 모델 정보
├── training_curves.png     # 학습 곡선
└── confusion_matrix.png    # 혼동 행렬
```

## 🔧 주요 기능

### 1. 데이터 로딩
- JSON 라벨 파일 자동 파싱
- 이미지 파일 존재 여부 확인
- 유효한 데이터만 필터링

### 2. 데이터 전처리
- 이미지 리사이징 (224x224)
- 데이터 증강 (회전, 플립, 색상 조정)
- 정규화 (ImageNet 평균/표준편차)

### 3. 모델 학습
- 자동 하이퍼파라미터 튜닝
- 조기 종료 (최고 성능 모델 저장)
- 학습 곡선 시각화

### 4. 모델 평가
- 분류 리포트 생성
- 혼동 행렬 시각화
- 클래스별 성능 분석

## 📝 사용 예시

```python
from c.cnn_label_training import FashionLabelTrainer

# 학습기 초기화
trainer = FashionLabelTrainer()

# 데이터 로드
trainer.load_data()

# 특정 속성 학습
X_train, X_test, y_train, y_test = trainer.prepare_training_data('상의_category')
train_loader, test_loader = trainer.create_data_loaders(X_train, X_test, y_train, y_test)

# 모델 학습
best_acc = trainer.train_model(train_loader, test_loader, num_classes=10, epochs=50)

# 모델 평가
trainer.evaluate_model(test_loader, trainer.label_encoders['상의_category'])
```

## ⚠️ 주의사항
1. **GPU 메모리**: 배치 크기를 GPU 메모리에 맞게 조정
2. **데이터 불균형**: 클래스별 데이터 분포 확인 필요
3. **과적합 방지**: 드롭아웃, 정규화, 데이터 증강 활용
4. **학습 시간**: 에포크 수와 데이터 크기에 따라 시간 소요

## 🎉 결과 활용
학습된 모델은 다음과 같이 활용할 수 있습니다:
- 패션 아이템 자동 분류
- 스타일 추천 시스템
- 쇼핑몰 상품 태깅
- 개인화된 패션 컨설팅

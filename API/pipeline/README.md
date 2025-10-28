# 🎯 패션 분석 파이프라인

AI 기반 패션 아이템 분석 시스템 - 깔끔하고 모듈화된 구조

---

## 📁 구조

```
pipeline/
├── __init__.py        # 패키지 초기화
├── models.py          # AI 모델 정의 (Gender, Style, Attribute)
├── loader.py          # 모델 로더
├── predictor.py       # 예측 파이프라인
├── database.py        # PostgreSQL 저장
├── main.py            # 전체 통합
└── README.md          # 이 파일
```

---

## 🚀 사용 방법

### 방법 1: 간단한 사용 (추천)

```python
from pipeline import FashionPipeline

# 파이프라인 초기화
pipeline = FashionPipeline()

# 이미지 분석
result = pipeline.process("photo.jpg", user_id=1)

# 결과 확인
print(f"성별: {result['gender']}")              # 'male' or 'female'
print(f"스타일: {result['style']}")              # '스트리트', '캐주얼' 등
print(f"감지된 카테고리: {result['detected_categories']}")  # ['top', 'bottom']
print(f"아이템 ID: {result['item_id']}")         # PostgreSQL에 저장된 ID

# 속성 확인
if 'top' in result['attributes']:
    top_attrs = result['attributes']['top']
    print(f"상의 카테고리: {top_attrs['category']['value']}")
    print(f"상의 색상: {top_attrs['color']['value']}")
    print(f"상의 성별: {top_attrs['gender']['value']}")  # ✅ gender 포함!
```

### 방법 2: 더 간단한 사용 (일회성)

```python
from pipeline import analyze_fashion_item

result = analyze_fashion_item("photo.jpg", user_id=1)
```

### 방법 3: 세부 제어

```python
from pipeline import ModelLoader, FashionPredictor, DatabaseManager

# 1. 모델 로드
loader = ModelLoader()
loader.load_all()

# 2. 예측만 (DB 저장 안 함)
predictor = FashionPredictor(loader)
prediction = predictor.process_image("photo.jpg", user_id=1)

# 3. 나중에 DB 저장
db = DatabaseManager(db_config={...})
item_id = db.save_prediction_result(user_id=1, image_path="photo.jpg", prediction_result=prediction)
db.close()
```

---

## 🔄 전체 흐름

```
사진 입력
  ↓
1️⃣ Gender 예측 (전체 이미지) → gender='male'
  ↓
2️⃣ Style 예측 (전체 이미지) → style='스트리트'
  ↓
3️⃣ YOLO 디텍팅 → 상의/하의/아우터/원피스 bbox
  ↓
4️⃣ Crop & 저장
  ↓
5️⃣ 각 Crop별 속성 예측
  ↓
6️⃣ PostgreSQL 저장
     - wardrobe_items: gender, style 저장
     - top_attributes_new: gender='male' 복사 ✅
     - bottom_attributes_new: gender='male' 복사 ✅
     - outer_attributes_new: gender='male' 복사 ✅
     - dress_attributes_new: gender='male' 복사 ✅
```

**핵심**: 1단계에서 예측한 `gender`를 모든 속성 테이블에 복사합니다!

---

## 📊 결과 구조

```python
{
    'success': True,
    'item_id': 123,
    'gender': 'male',
    'gender_confidence': 0.95,
    'style': '스트리트',
    'style_confidence': 0.89,
    'detected_categories': ['top', 'bottom'],
    'attributes': {
        'top': {
            'category': {'value': 'T셔츠', 'confidence': 0.92},
            'color': {'value': '검정', 'confidence': 0.88},
            'fit': {'value': '오버핏', 'confidence': 0.85},
            'material': {'value': '면', 'confidence': 0.91},
            'print': {'value': '무지', 'confidence': 0.93},
            'style': {'value': '캐주얼', 'confidence': 0.87},
            'sleeve': {'value': '반팔', 'confidence': 0.94},
            'gender': {'value': 'male', 'confidence': 1.0}  # ✅
        },
        'bottom': {
            'category': {'value': '청바지', 'confidence': 0.96},
            'color': {'value': '파랑', 'confidence': 0.89},
            'gender': {'value': 'male', 'confidence': 1.0},  # ✅
            ...
        }
    }
}
```

---

## ⚙️ 설정

### 모델 경로 변경

```python
pipeline = FashionPipeline(
    gender_model_path="path/to/gender_model.pth",  # 성별 모델 (없으면 None)
    style_model_path="path/to/style_model.pth",    # 스타일 모델
    yolo_model_path="path/to/yolo.pt",             # YOLO 모델
    top_model_path="path/to/top.pth",              # 상의 속성 모델
    bottom_model_path="path/to/bottom.pth",        # 하의 속성 모델
    outer_model_path="path/to/outer.pth",          # 아우터 속성 모델
    dress_model_path="path/to/dress.pth"           # 원피스 속성 모델
)
```

### DB 설정 변경

```python
pipeline = FashionPipeline(
    db_config={
        'host': 'localhost',
        'port': 5432,
        'database': 'kkokkaot_closet',
        'user': 'postgres',
        'password': 'your_password'
    }
)
```

---

## 📝 PostgreSQL 스키마

이 파이프라인을 사용하기 전에 PostgreSQL에 다음 테이블들이 있어야 합니다:

- `users` - 사용자 정보
- `wardrobe_items` - 옷장 아이템 메인 테이블 (gender, style 포함)
- `top_attributes_new` - 상의 속성 (gender 포함)
- `bottom_attributes_new` - 하의 속성 (gender 포함)
- `outer_attributes_new` - 아우터 속성 (gender 포함)
- `dress_attributes_new` - 원피스 속성 (gender 포함)

스키마 생성 SQL은 프로젝트 루트의 `schema.sql` 참고

---

## 🆚 기존 `_main_pipeline.py`와 비교

| 항목 | 기존 | 새 파이프라인 |
|------|------|-------------|
| 파일 수 | 1개 (1,450줄) | 6개 (각 200-400줄) |
| 구조 | 모든 게 한 파일 | 모듈별로 분리 |
| 가독성 | ❌ 어려움 | ✅ 쉬움 |
| 유지보수 | ❌ 어려움 | ✅ 쉬움 |
| 테스트 | ❌ 어려움 | ✅ 쉬움 |
| Gender 예측 | ❌ 없음 | ✅ 있음 |

---

## 🎓 예제

### 예제 1: 기본 사용

```python
from pipeline import FashionPipeline

pipeline = FashionPipeline()
result = pipeline.process("tshirt.jpg", user_id=1)

if result['success']:
    print(f"분석 성공! (ID: {result['item_id']})")
else:
    print(f"분석 실패: {result['error']}")
```

### 예제 2: 여러 이미지 처리

```python
from pipeline import FashionPipeline
from pathlib import Path

pipeline = FashionPipeline()

images = Path("./photos").glob("*.jpg")
for image_path in images:
    result = pipeline.process(str(image_path), user_id=1)
    print(f"{image_path.name}: {result['gender']} - {result['style']}")

pipeline.close()
```

### 예제 3: DB 저장 안 하고 예측만

```python
from pipeline import ModelLoader, FashionPredictor

loader = ModelLoader()
loader.load_all()

predictor = FashionPredictor(loader)
result = predictor.process_image("photo.jpg", user_id=1)

print(result['gender'])  # {'gender': 'male', 'confidence': 0.95}
print(result['style'])   # {'style': '스트리트', 'confidence': 0.89}
```

---

## 🐛 문제 해결

### 모델 로드 실패
```
❌ 스타일 모델 로드 실패: FileNotFoundError
```
→ 모델 파일 경로 확인

### DB 연결 실패
```
❌ PostgreSQL 연결 실패
```
→ PostgreSQL 실행 중인지 확인
→ db_config 설정 확인

### YOLO 감지 실패
```
❌ 의류 감지 실패
```
→ 이미지에 의류가 명확한지 확인
→ 이미지 품질 확인

---

## ✨ 특징

1. **모듈화**: 각 기능이 독립적인 파일로 분리
2. **깔끔한 구조**: 기존 1,450줄 → 6개 파일로 분리
3. **Gender 예측**: 새롭게 추가된 성별 예측 기능
4. **Gender 전파**: 한 번 예측한 성별을 모든 속성에 자동 복사
5. **쉬운 사용**: `pipeline.process()` 한 줄로 완료
6. **유지보수 용이**: 각 모듈을 독립적으로 수정 가능

---

**작성일**: 2025-01-XX  
**작성자**: 꼬까옷 팀


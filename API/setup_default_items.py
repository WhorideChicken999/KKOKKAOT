"""
기본 아이템 자동 설정 스크립트 (폴더 구조 기반)

폴더 구조:
D:\kkokkaot\API\default_items\
├── female/
│   ├── top/
│   ├── bottom/
│   ├── outer/
│   └── dress/
└── male/
    ├── top/
    ├── bottom/
    └── outer/

사용 방법:
1. 위 폴더 구조에 이미지 파일 넣기
2. 터미널에서 실행: python setup_default_items.py

기능:
- 폴더 구조로 성별/카테고리 자동 인식
- 속성만 예측 (YOLO 탐지 불필요)
- DB에 저장 (user_id=0, is_default=TRUE)
"""

from pathlib import Path
import sys
import shutil

# 프로젝트 루트를 PYTHONPATH에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import MODEL_PATHS, DB_CONFIG, IMAGE_PATHS
from pipeline.loader import ModelLoader
from pipeline.database import DatabaseManager
from pipeline.models import STYLE_CLASSES
import torch
from PIL import Image
from torchvision import transforms


def clear_existing_default_items(db):
    """기존 기본 아이템 완전 삭제"""
    print("\n" + "="*60)
    print("🗑️ 기존 기본 아이템 삭제 중...")
    print("="*60)
    
    try:
        with db.conn.cursor() as cur:
            # 1. 기존 기본 아이템 수 확인
            cur.execute("SELECT COUNT(*) FROM wardrobe_items WHERE is_default = TRUE")
            count = cur.fetchone()[0]
            
            if count == 0:
                print("✅ 삭제할 기본 아이템이 없습니다.\n")
                return 0
            
            print(f"📦 기존 기본 아이템: {count}개")
            
            # 2. 속성 테이블들 먼저 삭제
            cur.execute("DELETE FROM top_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE is_default = TRUE)")
            cur.execute("DELETE FROM bottom_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE is_default = TRUE)")
            cur.execute("DELETE FROM outer_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE is_default = TRUE)")
            cur.execute("DELETE FROM dress_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE is_default = TRUE)")
            
            # 3. 메인 테이블 삭제
            cur.execute("DELETE FROM wardrobe_items WHERE is_default = TRUE")
            
            db.conn.commit()
            
        print(f"✅ 기존 기본 아이템 {count}개 삭제 완료\n")
        return count
        
    except Exception as e:
        print(f"❌ 삭제 실패: {e}\n")
        db.conn.rollback()
        return 0


def predict_style(loader, image_path):
    """스타일 예측"""
    try:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(loader.device)
        
        with torch.no_grad():
            outputs = loader.style_model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            
        style = STYLE_CLASSES[pred_idx.item()]
        return style, confidence.item()
    except Exception as e:
        print(f"  ⚠️ 스타일 예측 실패: {e}")
        import traceback
        traceback.print_exc()
        return "캐주얼", 0.5


def predict_attributes(loader, category, gender, image_path):
    """카테고리별 속성 예측"""
    try:
        # 모델 선택
        if category not in loader.attribute_models:
            print(f"  ⚠️ {category} 모델이 로드되지 않음")
            return {}
        
        model_info = loader.attribute_models[category]
        model = model_info['model']
        encoders = model_info['encoders']
        
        # 이미지 전처리
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(loader.device)
        
        # 예측
        with torch.no_grad():
            outputs = model(image_tensor)
        
        # 결과 파싱
        attributes = {}
        for attr_name, output in outputs.items():
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, 1)
            
            if attr_name in encoders:
                value = encoders[attr_name].inverse_transform([pred_idx.item()])[0]
                attributes[attr_name] = {
                    'value': value,
                    'confidence': confidence.item()
                }
        
        # 성별 정보 추가 (폴더 기반)
        attributes['gender'] = {
            'value': gender,
            'confidence': 1.0
        }
        
        return attributes
        
    except Exception as e:
        print(f"  ⚠️ 속성 예측 실패: {e}")
        import traceback
        traceback.print_exc()
        return {}


def process_default_items():
    """폴더 구조 기반 기본 아이템 처리"""
    
    print("\n" + "="*60)
    print("🌱 기본 아이템 자동 설정 시작 (폴더 구조 기반)")
    print("="*60 + "\n")
    
    # 1. 폴더 구조 확인
    default_items_dir = IMAGE_PATHS['default_items']
    
    if not default_items_dir.exists():
        print(f"❌ default_items 폴더가 없습니다: {default_items_dir}")
        return
    
    # 2. 이미지 파일 수집
    image_list = []
    
    for gender in ['male', 'female']:
        gender_dir = default_items_dir / gender
        if not gender_dir.exists():
            print(f"⚠️ {gender} 폴더가 없습니다.")
            continue
        
        categories = ['top', 'bottom', 'outer']
        if gender == 'female':
            categories.append('dress')
        
        for category in categories:
            category_dir = gender_dir / category
            if not category_dir.exists():
                continue
            
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                for img_path in sorted(category_dir.glob(ext)):
                    image_list.append({
                        'path': img_path,
                        'gender': gender,
                        'category': category
                    })
    
    if not image_list:
        print("❌ 이미지가 없습니다!")
        return
    
    print(f"📁 발견된 이미지: {len(image_list)}개")
    for gender in ['male', 'female']:
        count = len([x for x in image_list if x['gender'] == gender])
        if count > 0:
            print(f"  - {gender}: {count}개")
    print()
    
    # 3. 모델 로더 초기화
    print("🤖 AI 모델 로딩 중...")
    loader = ModelLoader()
    
    # 각 모델 로드
    loader.load_style_model(MODEL_PATHS['style_model'])
    loader.load_attribute_models(
        top_path=MODEL_PATHS['top_model'],
        bottom_path=MODEL_PATHS['bottom_model'],
        outer_path=MODEL_PATHS['outer_model'],
        dress_path=MODEL_PATHS['dress_model']
    )
    
    print("✅ AI 모델 로딩 완료\n")
    
    # 4. 데이터베이스 초기화
    print("💾 데이터베이스 연결 중...")
    db = DatabaseManager(DB_CONFIG)
    print("✅ 데이터베이스 연결 완료\n")
    
    try:
        # 5. 시스템 사용자 생성 (user_id=0)
        print("="*60)
        print("👤 시스템 사용자 확인/생성 (user_id=0)")
        print("="*60)
        
        with db.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (user_id, username, email, password_hash)
                VALUES (0, 'system_default', 'system@kkokkaot.com', 'system_no_login')
                ON CONFLICT (user_id) DO NOTHING
            """)
            db.conn.commit()
        print("✅ 시스템 사용자 준비 완료\n")
        
        # 6. 기존 기본 아이템 삭제
        clear_existing_default_items(db)
        
        # 7. 각 이미지 처리
        print("="*60)
        print("🎨 AI 속성 예측 시작")
        print("="*60 + "\n")
        
        success_count = 0
        fail_count = 0
        
        for idx, item in enumerate(image_list, 1):
            image_path = item['path']
            gender = item['gender']
            category = item['category']
            
            print(f"[{idx}/{len(image_list)}] 📸 {gender}/{category}/{image_path.name}")
            
            try:
                # 스타일 예측
                style, style_conf = predict_style(loader, str(image_path))
                
                # 속성 예측
                attributes = predict_attributes(loader, category, gender, str(image_path))
                
                if not attributes:
                    print(f"  ❌ 속성 예측 실패\n")
                    fail_count += 1
                    continue
                
                # DB에 저장
                with db.conn.cursor() as cur:
                    # 1. wardrobe_items 삽입
                    cur.execute("""
                        INSERT INTO wardrobe_items 
                        (user_id, original_image_path, style, style_confidence, gender, gender_confidence,
                         has_top, has_bottom, has_outer, has_dress, is_default, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                        RETURNING item_id
                    """, (
                        0,  # user_id
                        str(image_path.name),  # 파일명만
                        style,
                        style_conf,
                        gender,
                        1.0,  # gender_confidence (폴더로 확정)
                        category == 'top',
                        category == 'bottom',
                        category == 'outer',
                        category == 'dress',
                        True  # is_default
                    ))
                    
                    item_id = cur.fetchone()[0]
                    
                    # 2. 속성 테이블에 삽입
                    table_name = f"{category}_attributes_new"
                    
                    # 모델 속성명 → DB 컬럼명 매핑
                    attr_name_mapping = {
                        'print': 'print_pattern',
                        'sleeve': 'sleeve_length'
                    }
                    
                    # confidence 컬럼이 실제로 존재하는 속성만
                    attrs_with_confidence = ['category', 'color', 'fit', 'gender']
                    
                    # 속성 컬럼명과 값 준비
                    columns = ['item_id']
                    values = [item_id]
                    
                    for attr_name, attr_data in attributes.items():
                        # 속성명 매핑 (모델 이름 → DB 컬럼명)
                        db_column_name = attr_name_mapping.get(attr_name, attr_name)
                        
                        columns.append(db_column_name)
                        values.append(attr_data['value'])
                        
                        # confidence 컬럼이 있는 속성만 confidence 저장
                        if attr_name in attrs_with_confidence:
                            columns.append(f"{db_column_name}_confidence")
                            values.append(1.0)
                    
                    placeholders = ', '.join(['%s'] * len(values))
                    columns_str = ', '.join(columns)
                    
                    cur.execute(f"""
                        INSERT INTO {table_name} ({columns_str})
                        VALUES ({placeholders})
                    """, values)
                    
                    db.conn.commit()
                
                # 3. 이미지 파일 저장
                # user_0/{category}/item_{item_id}_{category}.jpg
                category_dir = IMAGE_PATHS['processed'] / f"user_0" / category
                category_dir.mkdir(parents=True, exist_ok=True)
                category_image_path = category_dir / f"item_{item_id}_{category}.jpg"
                shutil.copy(image_path, category_image_path)
                
                # user_0/full/item_{item_id}_full.jpg
                full_dir = IMAGE_PATHS['processed'] / f"user_0" / "full"
                full_dir.mkdir(parents=True, exist_ok=True)
                full_image_path = full_dir / f"item_{item_id}_full.jpg"
                shutil.copy(image_path, full_image_path)
                
                # 결과 출력
                print(f"  ✅ 완료: item_id={item_id}")
                print(f"     👔 스타일: {style} ({style_conf:.1%})")
                print(f"     🎨 속성:")
                for attr_name, attr_data in attributes.items():
                    if attr_name != 'gender':
                        print(f"        - {attr_name}: {attr_data['value']} ({attr_data['confidence']:.1%})")
                print()
                
                success_count += 1
                
            except Exception as e:
                # 트랜잭션 롤백
                db.conn.rollback()
                print(f"  ❌ 처리 실패: {e}\n")
                fail_count += 1
                continue
        
        # 7. 결과 요약
        print("\n" + "="*60)
        print("🎉 기본 아이템 설정 완료!")
        print("="*60)
        print(f"  ✅ 성공: {success_count}개")
        print(f"  ❌ 실패: {fail_count}개")
        print("="*60 + "\n")
        
        if success_count > 0:
            print("💡 프론트엔드에서 기본 아이템을 확인하세요!")
            print("   옷장 > 기본 아이템 탭에서 볼 수 있습니다.\n")
        
    finally:
        db.close()
        print("✅ 데이터베이스 연결 종료\n")


if __name__ == '__main__':
    try:
        process_default_items()
    except KeyboardInterrupt:
        print("\n\n⚠️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

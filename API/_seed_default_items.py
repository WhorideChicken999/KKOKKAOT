import os
from pathlib import Path
from _main_pipeline import FashionPipeline

# 기본 제공할 대표 아이템들 (default_items 폴더의 모든 jpg 파일)
def get_all_images(folder_path):
    """폴더 내 모든 jpg 이미지 목록 가져오기"""
    images = []
    for file in sorted(Path(folder_path).glob("*.jpg")):
        images.append(file.name)
    return images


def get_users_without_items(pipeline):
    """옷이 없는 사용자 목록 가져오기"""
    with pipeline.db_conn.cursor() as cur:
        cur.execute("""
            SELECT u.user_id, u.username, u.email
            FROM users u
            LEFT JOIN wardrobe_items w ON u.user_id = w.user_id
            WHERE w.user_id IS NULL 
            AND u.user_id != 0  -- 시스템 사용자 제외
            ORDER BY u.user_id
        """)
        return cur.fetchall()


def recommend_default_items_to_user(pipeline, user_id, username):
    """특정 사용자에게 기본 아이템들 추천"""
    print(f"\n👤 사용자 {user_id} ({username})에게 기본 아이템 추천 중...")
    
    # 1. 기본 아이템들 가져오기
    with pipeline.db_conn.cursor() as cur:
        cur.execute("""
            SELECT item_id, original_image_path, has_top, has_bottom, has_outer, has_dress
            FROM wardrobe_items 
            WHERE is_default = TRUE
            ORDER BY item_id
        """)
        default_items = cur.fetchall()
    
    if not default_items:
        print(f"   ⚠️ 기본 아이템이 없습니다.")
        return 0
    
    print(f"   📦 {len(default_items)}개의 기본 아이템 발견")
    
    # 2. 사용자에게 추천 아이템 추가
    recommended_count = 0
    for item_id, image_path, has_top, has_bottom, has_outer, has_dress in default_items:
        try:
            with pipeline.db_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO user_recommendations (user_id, item_id, recommendation_type, created_at)
                    VALUES (%s, %s, 'default_item', NOW())
                    ON CONFLICT (user_id, item_id) DO NOTHING
                """, (user_id, item_id))
                pipeline.db_conn.commit()
                recommended_count += 1
        except Exception as e:
            print(f"   ❌ 추천 추가 실패 (item_id={item_id}): {e}")
    
    print(f"   ✅ {recommended_count}개 아이템 추천 완료")
    return recommended_count


def clear_existing_default_items(pipeline):
    """기존 기본 아이템들 삭제"""
    print("🗑️ 기존 기본 아이템들 삭제 중...")
    
    with pipeline.db_conn.cursor() as cur:
        # user_recommendations에서 기본 아이템 추천 삭제
        cur.execute("""
            DELETE FROM user_recommendations 
            WHERE item_id IN (
                SELECT item_id FROM wardrobe_items WHERE is_default = TRUE
            )
        """)
        
        # top_attributes, bottom_attributes에서 기본 아이템 속성 삭제
        cur.execute("""
            DELETE FROM top_attributes 
            WHERE item_id IN (
                SELECT item_id FROM wardrobe_items WHERE is_default = TRUE
            )
        """)
        
        cur.execute("""
            DELETE FROM bottom_attributes 
            WHERE item_id IN (
                SELECT item_id FROM wardrobe_items WHERE is_default = TRUE
            )
        """)
        
        # wardrobe_items에서 기본 아이템 삭제
        cur.execute("DELETE FROM wardrobe_items WHERE is_default = TRUE")
        
        pipeline.db_conn.commit()
        print("✅ 기존 기본 아이템들 삭제 완료\n")


def seed_default_items():
    """기본 아이템들을 DB에 삽입"""
    
    print("\n" + "="*60)
    print("🌱 기본 아이템 데이터 삽입 시작")
    print("="*60 + "\n")
    
    # 파이프라인 초기화 (4개 카테고리 지원, Pose 제외, Background Remover 포함)
    pipeline = FashionPipeline(
        yolo_pose_path=None,  # Pose 모델 불필요
        yolo_detection_path="D:/kkokkaot/API/pre_trained_weights/yolo_best.pt",  # 4개 카테고리 감지만 사용
        top_model_path="D:/kkokkaot/API/pre_trained_weights/fashion_top_model.pth",
        bottom_model_path="D:/kkokkaot/API/pre_trained_weights/fashion_bottom_model.pth",
        chroma_path="./chroma_db",
        db_config={
            'host': 'localhost',
            'port': 5432,
            'database': 'kkokkaot_closet',
            'user': 'postgres',
            'password': '000000'
        }
    )
    
    try:
        # 0. 기존 기본 아이템들 삭제
        clear_existing_default_items(pipeline)
        
        # 1. 시스템 사용자 생성 (user_id = 0)
        with pipeline.db_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (user_id, username, email, password_hash)
                VALUES (0, 'system_default', 'system@kkokkaot.com', 'system_no_login')
                ON CONFLICT (user_id) DO NOTHING
            """)
            pipeline.db_conn.commit()
            print("✅ 시스템 사용자 생성 완료 (user_id=0)")
        
        # 2. user_recommendations 테이블 생성 (없으면)
        with pipeline.db_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_recommendations (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    item_id INTEGER NOT NULL,
                    recommendation_type VARCHAR(50) DEFAULT 'default_item',
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(user_id, item_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                    FOREIGN KEY (item_id) REFERENCES wardrobe_items(item_id) ON DELETE CASCADE
                )
            """)
            pipeline.db_conn.commit()
            print("✅ user_recommendations 테이블 확인/생성 완료\n")
        
        # 3. default_items 폴더의 모든 이미지 가져오기
        base_path = Path("./API/default_items")
        image_files = get_all_images(base_path)
        
        print(f"📁 발견된 이미지: {len(image_files)}개\n")
        
        if len(image_files) == 0:
            print("⚠️  default_items 폴더에 이미지가 없습니다!")
            return
        
        # 4. 각 이미지 처리
        success_count = 0
        fail_count = 0
        
        for idx, image_file in enumerate(image_files, 1):
            image_path = base_path / image_file
            
            if not image_path.exists():
                print(f"⚠️  이미지 없음: {image_path}")
                fail_count += 1
                continue
            
            print(f"[{idx}/{len(image_files)}] 처리 중: {image_file}")
            
            try:
                # AI 파이프라인 실행 (YOLO Pose로 자동 분리)
                result = pipeline.process_image(
                    image_path=str(image_path),
                    user_id=0,  # 시스템 사용자
                    save_separated_images=True
                )
                
                if result['success']:
                    # is_default = TRUE로 업데이트하고 original_image_path를 원본 파일명으로 설정
                    with pipeline.db_conn.cursor() as cur:
                        cur.execute("""
                            UPDATE wardrobe_items 
                            SET is_default = TRUE, gender = 'unisex', original_image_path = %s
                            WHERE item_id = %s
                        """, (image_file, result['item_id']))
                        pipeline.db_conn.commit()
                    
                    print(f"   ✅ 완료: item_id={result['item_id']}, 원본파일: {image_file}")
                    
                    # 감지된 카테고리 정보 출력
                    detection_result = result.get('detection_result', {})
                    if detection_result.get('has_top'):
                        print(f"      👕 상의: {result.get('top_attributes', {}).get('category', 'unknown')} ({result.get('top_attributes', {}).get('color', 'unknown')})")
                    if detection_result.get('has_bottom'):
                        print(f"      👖 하의: {result.get('bottom_attributes', {}).get('category', 'unknown')} ({result.get('bottom_attributes', {}).get('color', 'unknown')})")
                    if detection_result.get('has_outer'):
                        print(f"      🧥 아우터: {result.get('outer_attributes', {}).get('category', 'unknown')} ({result.get('outer_attributes', {}).get('color', 'unknown')})")
                    if detection_result.get('has_dress'):
                        print(f"      👗 드레스: {result.get('dress_attributes', {}).get('category', 'unknown')} ({result.get('dress_attributes', {}).get('color', 'unknown')})")
                    
                    print()
                    success_count += 1
                else:
                    print(f"   ❌ 실패: {result.get('error')}\n")
                    fail_count += 1
                    
            except Exception as e:
                print(f"   ❌ 예외 발생: {e}\n")
                fail_count += 1
                continue
        
        print("\n" + "="*60)
        print("🎉 기본 아이템 삽입 완료!")
        print(f"   ✅ 성공: {success_count}개")
        print(f"   ❌ 실패: {fail_count}개")
        print("="*60)
        
        # 4. 옷이 없는 사용자들에게 기본 아이템 추천
        print("\n" + "="*60)
        print("🎯 옷이 없는 사용자들에게 기본 아이템 추천")
        print("="*60)
        
        users_without_items = get_users_without_items(pipeline)
        
        if not users_without_items:
            print("✅ 모든 사용자가 이미 옷을 가지고 있습니다!")
        else:
            print(f"👥 옷이 없는 사용자: {len(users_without_items)}명")
            
            total_recommended = 0
            for user_id, username, email in users_without_items:
                recommended_count = recommend_default_items_to_user(pipeline, user_id, username)
                total_recommended += recommended_count
            
            print(f"\n🎉 추천 완료!")
            print(f"   👥 대상 사용자: {len(users_without_items)}명")
            print(f"   📦 총 추천 아이템: {total_recommended}개")
        
        print("="*60)
        
    finally:
        pipeline.close()


if __name__ == '__main__':
    seed_default_items()
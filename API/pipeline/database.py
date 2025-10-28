"""
데이터베이스 저장
- PostgreSQL에 예측 결과 저장
- wardrobe_items + 각 속성 테이블
"""
import psycopg2
from pathlib import Path
from typing import Dict, Optional


class DatabaseManager:
    """PostgreSQL 데이터베이스 관리"""
    
    def __init__(self, db_config: dict):
        """
        Args:
            db_config: {'host': 'localhost', 'port': 5432, 'database': 'kkokkaot_closet', ...}
        """
        self.db_config = db_config
        self.conn = psycopg2.connect(**db_config)
        print("✅ PostgreSQL 연결 완료")
    
    def save_item(self, 
                  user_id: int,
                  image_path: str,
                  gender: str,
                  gender_confidence: float,
                  style: str,
                  style_confidence: float,
                  has_top: bool,
                  has_bottom: bool,
                  has_outer: bool,
                  has_dress: bool,
                  is_default: bool = False) -> int:
        """
        wardrobe_items 테이블에 메인 정보 저장
        
        Returns:
            item_id: 생성된 아이템 ID
        """
        print("\n[6/6] PostgreSQL 저장 중...")
        
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO wardrobe_items (
                        user_id, 
                        original_image_path,
                        gender,
                        gender_confidence,
                        style,
                        style_confidence,
                        has_top,
                        has_bottom,
                        has_outer,
                        has_dress,
                        is_default
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING item_id
                """, (
                    user_id,
                    image_path,
                    gender,
                    gender_confidence,
                    style,
                    style_confidence,
                    has_top,
                    has_bottom,
                    has_outer,
                    has_dress,
                    is_default
                ))
                
                item_id = cur.fetchone()[0]
                self.conn.commit()
                
                print(f"  ✅ wardrobe_items 저장 완료 (item_id: {item_id})")
                return item_id
                
        except Exception as e:
            self.conn.rollback()
            print(f"  ❌ wardrobe_items 저장 실패: {e}")
            raise
    
    def save_top_attributes(self, item_id: int, attributes: Dict):
        """상의 속성 저장"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO top_attributes_new (
                        item_id,
                        category,
                        color,
                        fit,
                        material,
                        print_pattern,
                        style,
                        sleeve_length,
                        gender,
                        category_confidence,
                        color_confidence,
                        fit_confidence,
                        gender_confidence
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    item_id,
                    attributes.get('category', {}).get('value', 'Unknown'),
                    attributes.get('color', {}).get('value', 'Unknown'),
                    attributes.get('fit', {}).get('value', 'Unknown'),
                    attributes.get('material', {}).get('value', 'Unknown'),
                    attributes.get('print', {}).get('value', 'Unknown'),
                    attributes.get('style', {}).get('value', 'Unknown'),
                    attributes.get('sleeve', {}).get('value', 'Unknown'),
                    attributes.get('gender', {}).get('value', 'Unknown'),  # ✅ gender 포함
                    attributes.get('category', {}).get('confidence', 0.0),
                    attributes.get('color', {}).get('confidence', 0.0),
                    attributes.get('fit', {}).get('confidence', 0.0),
                    attributes.get('gender', {}).get('confidence', 0.0)  # ✅ gender_confidence
                ))
                
                self.conn.commit()
                print(f"  ✅ top_attributes_new 저장 완료")
                
        except Exception as e:
            self.conn.rollback()
            print(f"  ❌ top_attributes_new 저장 실패: {e}")
            raise
    
    def save_bottom_attributes(self, item_id: int, attributes: Dict):
        """하의 속성 저장"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO bottom_attributes_new (
                        item_id,
                        category,
                        color,
                        fit,
                        material,
                        print_pattern,
                        style,
                        length,
                        gender,
                        category_confidence,
                        color_confidence,
                        fit_confidence,
                        gender_confidence
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    item_id,
                    attributes.get('category', {}).get('value', 'Unknown'),
                    attributes.get('color', {}).get('value', 'Unknown'),
                    attributes.get('fit', {}).get('value', 'Unknown'),
                    attributes.get('material', {}).get('value', 'Unknown'),
                    attributes.get('print', {}).get('value', 'Unknown'),
                    attributes.get('style', {}).get('value', 'Unknown'),
                    attributes.get('length', {}).get('value', 'Unknown'),
                    attributes.get('gender', {}).get('value', 'Unknown'),  # ✅ gender
                    attributes.get('category', {}).get('confidence', 0.0),
                    attributes.get('color', {}).get('confidence', 0.0),
                    attributes.get('fit', {}).get('confidence', 0.0),
                    attributes.get('gender', {}).get('confidence', 0.0)  # ✅ gender_confidence
                ))
                
                self.conn.commit()
                print(f"  ✅ bottom_attributes_new 저장 완료")
                
        except Exception as e:
            self.conn.rollback()
            print(f"  ❌ bottom_attributes_new 저장 실패: {e}")
            raise
    
    def save_outer_attributes(self, item_id: int, attributes: Dict):
        """아우터 속성 저장"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO outer_attributes_new (
                        item_id,
                        category,
                        color,
                        fit,
                        material,
                        print_pattern,
                        style,
                        sleeve_length,
                        gender,
                        category_confidence,
                        color_confidence,
                        fit_confidence,
                        gender_confidence
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    item_id,
                    attributes.get('category', {}).get('value', 'Unknown'),
                    attributes.get('color', {}).get('value', 'Unknown'),
                    attributes.get('fit', {}).get('value', 'Unknown'),
                    attributes.get('material', {}).get('value', 'Unknown'),
                    attributes.get('print', {}).get('value', 'Unknown'),
                    attributes.get('style', {}).get('value', 'Unknown'),
                    attributes.get('sleeve', {}).get('value', 'Unknown'),
                    attributes.get('gender', {}).get('value', 'Unknown'),  # ✅ gender
                    attributes.get('category', {}).get('confidence', 0.0),
                    attributes.get('color', {}).get('confidence', 0.0),
                    attributes.get('fit', {}).get('confidence', 0.0),
                    attributes.get('gender', {}).get('confidence', 0.0)  # ✅ gender_confidence
                ))
                
                self.conn.commit()
                print(f"  ✅ outer_attributes_new 저장 완료")
                
        except Exception as e:
            self.conn.rollback()
            print(f"  ❌ outer_attributes_new 저장 실패: {e}")
            raise
    
    def save_dress_attributes(self, item_id: int, attributes: Dict):
        """원피스 속성 저장"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO dress_attributes_new (
                        item_id,
                        category,
                        color,
                        material,
                        print_pattern,
                        style,
                        gender,
                        category_confidence,
                        color_confidence,
                        gender_confidence
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    item_id,
                    attributes.get('category', {}).get('value', 'Unknown'),
                    attributes.get('color', {}).get('value', 'Unknown'),
                    attributes.get('material', {}).get('value', 'Unknown'),
                    attributes.get('print', {}).get('value', 'Unknown'),
                    attributes.get('style', {}).get('value', 'Unknown'),
                    attributes.get('gender', {}).get('value', 'Unknown'),  # ✅ gender
                    attributes.get('category', {}).get('confidence', 0.0),
                    attributes.get('color', {}).get('confidence', 0.0),
                    attributes.get('gender', {}).get('confidence', 0.0)  # ✅ gender_confidence
                ))
                
                self.conn.commit()
                print(f"  ✅ dress_attributes_new 저장 완료")
                
        except Exception as e:
            self.conn.rollback()
            print(f"  ❌ dress_attributes_new 저장 실패: {e}")
            raise
    
    def save_all_attributes(self, item_id: int, attributes: Dict):
        """모든 속성 저장 (있는 것만)"""
        print(f"\n  💾 속성 저장 중... (item_id: {item_id})")
        
        if 'top' in attributes:
            self.save_top_attributes(item_id, attributes['top'])
        
        if 'bottom' in attributes:
            self.save_bottom_attributes(item_id, attributes['bottom'])
        
        if 'outer' in attributes:
            self.save_outer_attributes(item_id, attributes['outer'])
        
        if 'dress' in attributes:
            self.save_dress_attributes(item_id, attributes['dress'])
        
        print(f"  ✅ 모든 속성 저장 완료")
    
    def save_prediction_result(self,
                              user_id: int,
                              image_path: str,
                              prediction_result: Dict) -> int:
        """
        전체 예측 결과를 한 번에 저장
        
        Args:
            user_id: 사용자 ID
            image_path: 원본 이미지 경로
            prediction_result: predictor.process_image()의 결과
        
        Returns:
            item_id: 생성된 아이템 ID
        """
        if not prediction_result.get('success'):
            raise ValueError("예측 실패한 결과는 저장할 수 없습니다.")
        
        # 1. wardrobe_items 저장
        item_id = self.save_item(
            user_id=user_id,
            image_path=image_path,
            gender=prediction_result['gender']['gender'],
            gender_confidence=prediction_result['gender']['confidence'],
            style=prediction_result['style']['style'],
            style_confidence=prediction_result['style']['confidence'],
            has_top=prediction_result['detection']['has_top'],
            has_bottom=prediction_result['detection']['has_bottom'],
            has_outer=prediction_result['detection']['has_outer'],
            has_dress=prediction_result['detection']['has_dress'],
            is_default=False
        )
        
        # 2. 각 속성 테이블 저장 (gender 포함!)
        self.save_all_attributes(item_id, prediction_result['attributes'])
        
        print(f"\n{'='*60}")
        print(f"✅ 데이터베이스 저장 완료!")
        print(f"  - item_id: {item_id}")
        print(f"  - gender: {prediction_result['gender']['gender']}")
        print(f"  - style: {prediction_result['style']['style']}")
        print(f"{'='*60}\n")
        
        return item_id
    
    def close(self):
        """연결 종료"""
        if self.conn:
            self.conn.close()
            print("✅ PostgreSQL 연결 종료")


import psycopg2
from psycopg2.extras import Json, RealDictCursor
from datetime import datetime
import json

class WardrobeDB:
    """가상옷장 데이터베이스 관리"""
    
    def __init__(self, host='localhost', port=5432, database='fashion_db', 
                 user='postgres', password='your_password'):
        """PostgreSQL 연결"""
        self.conn_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.conn = None
        self.connect()
    
    def connect(self):
        """데이터베이스 연결"""
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            print("PostgreSQL 연결 성공")
        except Exception as e:
            print(f"연결 실패: {e}")
            raise
    
    def close(self):
        """연결 종료"""
        if self.conn:
            self.conn.close()
            print("연결 종료")
    
    def create_user(self, username, email):
        """사용자 생성"""
        with self.conn.cursor() as cur:
            try:
                cur.execute("""
                    INSERT INTO users (username, email)
                    VALUES (%s, %s)
                    RETURNING user_id
                """, (username, email))
                user_id = cur.fetchone()[0]
                self.conn.commit()
                print(f"사용자 생성: {username} (ID: {user_id})")
                return user_id
            except psycopg2.IntegrityError:
                self.conn.rollback()
                # 이미 존재하면 조회
                cur.execute("SELECT user_id FROM users WHERE username = %s", (username,))
                user_id = cur.fetchone()[0]
                return user_id
    
    def add_wardrobe_item(self, user_id, original_image_path, 
                         has_top=False, has_bottom=False,
                         top_image_path=None, bottom_image_path=None,
                         waist_y=None, chroma_embedding_id=None,
                         gender=None, style=None):
        """가상옷장 아이템 추가"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO wardrobe_items (
                    user_id, original_image_path, 
                    has_top, has_bottom,
                    top_image_path, bottom_image_path,
                    waist_y, chroma_embedding_id,
                    gender, style
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING item_id
            """, (user_id, original_image_path, 
                  has_top, has_bottom,
                  top_image_path, bottom_image_path,
                  waist_y, chroma_embedding_id,
                  gender, style))
            
            item_id = cur.fetchone()[0]
            self.conn.commit()
            print(f"아이템 추가: ID {item_id}")
            return item_id
    
    def add_top_attributes(self, item_id, category=None, color=None, 
                          sub_color=None, fit=None, length=None,
                          sleeve_length=None, neckline=None, collar=None,
                          materials=None, prints=None, details=None,
                          category_confidence=0.0, color_confidence=0.0, 
                          fit_confidence=0.0):
        """상의 속성 추가"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO top_attributes (
                    item_id, category, color, sub_color, fit, length,
                    sleeve_length, neckline, collar,
                    materials, prints, details,
                    category_confidence, color_confidence, fit_confidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING top_id
            """, (item_id, category, color, sub_color, fit, length,
                  sleeve_length, neckline, collar,
                  Json(materials or []), Json(prints or []), Json(details or []),
                  category_confidence, color_confidence, fit_confidence))
            
            top_id = cur.fetchone()[0]
            self.conn.commit()
            return top_id
    
    def add_bottom_attributes(self, item_id, category=None, color=None,
                             sub_color=None, fit=None, length=None,
                             materials=None, prints=None, details=None,
                             category_confidence=0.0, color_confidence=0.0,
                             fit_confidence=0.0):
        """하의 속성 추가"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bottom_attributes (
                    item_id, category, color, sub_color, fit, length,
                    materials, prints, details,
                    category_confidence, color_confidence, fit_confidence
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING bottom_id
            """, (item_id, category, color, sub_color, fit, length,
                  Json(materials or []), Json(prints or []), Json(details or []),
                  category_confidence, color_confidence, fit_confidence))
            
            bottom_id = cur.fetchone()[0]
            self.conn.commit()
            return bottom_id
    
    def get_user_wardrobe(self, user_id, limit=50):
        """사용자 옷장 조회"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM wardrobe_full_view
                WHERE user_id = %s
                ORDER BY upload_date DESC
                LIMIT %s
            """, (user_id, limit))
            
            items = cur.fetchall()
            return [dict(item) for item in items]
    
    def save_recommendation_history(self, user_id, query_item_id,
                                   recommended_items, filter_gender=None,
                                   filter_style=None, filter_category=None):
        """추천 기록 저장"""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO recommendation_history (
                    user_id, query_item_id, recommended_items,
                    filter_gender, filter_style, filter_category
                ) VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING rec_id
            """, (user_id, query_item_id, Json(recommended_items),
                  filter_gender, filter_style, filter_category))
            
            rec_id = cur.fetchone()[0]
            self.conn.commit()
            return rec_id
    
    def search_items(self, user_id=None, category=None, color=None, 
                    style=None, gender=None):
        """아이템 검색"""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = "SELECT * FROM wardrobe_full_view WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = %s"
                params.append(user_id)
            if category:
                query += " AND (top_category = %s OR bottom_category = %s)"
                params.extend([category, category])
            if color:
                query += " AND (top_color = %s OR bottom_color = %s)"
                params.extend([color, color])
            if style:
                query += " AND style = %s"
                params.append(style)
            if gender:
                query += " AND gender = %s"
                params.append(gender)
            
            query += " ORDER BY upload_date DESC LIMIT 50"
            
            cur.execute(query, params)
            items = cur.fetchall()
            return [dict(item) for item in items]


# 사용 예시
if __name__ == '__main__':
    # 데이터베이스 연결 (비밀번호 수정 필요)
    db = WardrobeDB(
        host='localhost',
        port=5432,
        database='kkokkaot_closet',
        user='postgres',
        password='000000'  # 실제 비밀번호로 변경
    )
    
    try:
        # 사용자 생성
        user_id = db.create_user('test_user', 'test@example.com')
        
        # 아이템 추가
        item_id = db.add_wardrobe_item(
            user_id=user_id,
            original_image_path='/path/to/image.jpg',
            has_top=True,
            has_bottom=True,
            top_image_path='/path/to/top.jpg',
            bottom_image_path='/path/to/bottom.jpg',
            waist_y=350,
            chroma_embedding_id='img_12345',
            gender='female',
            style='로맨틱'
        )
        
        # 상의 속성 추가
        db.add_top_attributes(
            item_id=item_id,
            category='블라우스',
            color='화이트',
            fit='루즈',
            materials=['우븐', '레이스'],
            prints=['도트'],
            details=['러플', '리본'],
            category_confidence=0.95,
            color_confidence=0.98
        )
        
        # 하의 속성 추가
        db.add_bottom_attributes(
            item_id=item_id,
            category='팬츠',
            color='블랙',
            fit='노멀',
            materials=['우븐'],
            category_confidence=0.92
        )
        
        # 옷장 조회
        wardrobe = db.get_user_wardrobe(user_id)
        print(f"\n총 {len(wardrobe)}개 아이템")
        for item in wardrobe[:3]:
            print(f"- {item['top_category']} + {item['bottom_category']}")
        
    finally:
        db.close()
import numpy as np
from typing import List, Dict, Optional
import chromadb
import psycopg2
from psycopg2.extras import RealDictCursor
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from pathlib import Path
import platform

# Matplotlib Korean font settings
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
else:  # Linux
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False


class FashionRecommender:
    """패션 코디 추천 시스템"""
    
    def __init__(self, chroma_path: str = "./chroma_db", db_config: dict = None):
        """초기화"""
        
        # ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.chroma_collection = self.chroma_client.get_collection(name="fashion_collection")
        
        # PostgreSQL
        if db_config is None:
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'kkokkaot_closet',
                'user': 'postgres',
                'password': '000000'
            }
        self.db_conn = psycopg2.connect(**db_config)
        
        # 색상 조화 룰
        self.color_harmony_rules = self._init_color_rules()
        
        # 스타일 매칭 룰
        self.style_matching_rules = self._init_style_rules()
    
    
    def _init_color_rules(self) -> Dict:
        """색상 조화 룰 정의"""
        return {
            # 무채색 (모든 색과 잘 어울림)
            'neutral': ['화이트', '블랙', '그레이', '베이지', '네이비', '아이보리'],
            
            # 보색 관계
            'complementary': {
                '레드': ['그린', '민트'],
                '블루': ['오렌지', '코랄'],
                '옐로우': ['퍼플', '바이올렛'],
                '핑크': ['그린', '민트'],
            },
            
            # 유사색 (인접색)
            'analogous': {
                '레드': ['핑크', '오렌지', '버건디'],
                '블루': ['퍼플', '민트', '네이비'],
                '옐로우': ['오렌지', '베이지', '골드'],
                '그린': ['민트', '카키', '올리브'],
                '핑크': ['레드', '퍼플', '코랄'],
            },
            
            # 톤온톤 (같은 색 계열)
            'monochromatic': {
                '블루': ['네이비', '스카이블루', '인디고'],
                '그린': ['카키', '올리브', '민트'],
                '레드': ['버건디', '와인', '핑크'],
            }
        }
    
    
    def _init_style_rules(self) -> Dict:
        """스타일 매칭 룰 정의"""
        return {
            # 각 스타일과 잘 어울리는 스타일들
            '로맨틱': ['페미닌', '로맨틱', '클래식', '프레피'],
            '페미닌': ['로맨틱', '페미닌', '클래식', '소피스트케이티드'],
            '캐주얼': ['스트리트', '스포티', '캐주얼', '젠더리스'],
            '스트리트': ['힙합', '스포티', '캐주얼', '톰보이'],
            '모던': ['미니멀', '소피스트케이티드', '모던', '클래식'],
            '클래식': ['프레피', '소피스트케이티드', '모던', '페미닌'],
            '스포티': ['캐주얼', '스트리트', '스포티', '액티브'],
            '섹시': ['소피스트케이티드', '모던', '섹시'],
            '빈티지': ['레트로', '클래식', '빈티지', '히피'],
            '힙합': ['스트리트', '스포티', '힙합', '펑크'],
        }
    
    
    def calculate_color_score(self, color1: str, color2: str) -> float:
        """색상 조화 점수 계산 (0.0 ~ 1.0)"""
        
        if not color1 or not color2:
            return 0.5
        
        color1 = color1.lower()
        color2 = color2.lower()
        
        # 같은 색상
        if color1 == color2:
            return 0.9
        
        # 무채색 포함 (항상 잘 어울림)
        neutral_colors = [c.lower() for c in self.color_harmony_rules['neutral']]
        if color1 in neutral_colors or color2 in neutral_colors:
            return 0.85
        
        # 보색 관계
        for main_color, complements in self.color_harmony_rules['complementary'].items():
            main_lower = main_color.lower()
            complements_lower = [c.lower() for c in complements]
            
            if (color1 == main_lower and color2 in complements_lower) or \
               (color2 == main_lower and color1 in complements_lower):
                return 0.8
        
        # 유사색 관계
        for main_color, analogs in self.color_harmony_rules['analogous'].items():
            main_lower = main_color.lower()
            analogs_lower = [c.lower() for c in analogs]
            
            if (color1 == main_lower and color2 in analogs_lower) or \
               (color2 == main_lower and color1 in analogs_lower):
                return 0.75
        
        # 톤온톤
        for main_color, tones in self.color_harmony_rules['monochromatic'].items():
            tones_lower = [c.lower() for c in tones]
            if color1 in tones_lower and color2 in tones_lower:
                return 0.7
        
        # 기본 점수
        return 0.5
    
    
    def calculate_style_score(self, style1: str, style2: str) -> float:
        """스타일 매칭 점수 계산 (0.0 ~ 1.0)"""
        
        if not style1 or not style2:
            return 0.5
        
        style1 = style1.lower()
        style2 = style2.lower()
        
        # 같은 스타일
        if style1 == style2:
            return 1.0
        
        # 매칭 룰 확인
        for main_style, compatible_styles in self.style_matching_rules.items():
            main_lower = main_style.lower()
            compatible_lower = [s.lower() for s in compatible_styles]
            
            if style1 == main_lower and style2 in compatible_lower:
                return 0.8
            if style2 == main_lower and style1 in compatible_lower:
                return 0.8
        
        return 0.3
    
    
    def get_item_details(self, item_id: int) -> Dict:
        """PostgreSQL에서 아이템 상세 정보 조회"""
        
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM wardrobe_full_view
                WHERE item_id = %s
            """, (item_id,))
            
            result = cur.fetchone()
            return dict(result) if result else None
    
    
    def recommend_matching_items(self, 
                                 query_item_id: int,
                                 recommend_type: str = 'bottom',  # 'top' or 'bottom'
                                 user_id: Optional[int] = None,
                                 gender_filter: Optional[str] = None,
                                 style_filter: Optional[str] = None,
                                 n_results: int = 10) -> List[Dict]:
        """
        코디 매칭 아이템 추천
        
        Args:
            query_item_id: 기준 아이템 ID
            recommend_type: 'top' (상의 추천) or 'bottom' (하의 추천)
            user_id: 사용자 ID (특정 사용자 옷장만 검색)
            gender_filter: 성별 필터
            style_filter: 스타일 필터
            n_results: 추천 개수
        """
        
        print(f"\n=== 추천 시작 ===")
        print(f"기준 아이템: {query_item_id}")
        print(f"추천 타입: {recommend_type}")
        
        # 1. 기준 아이템 정보 가져오기
        query_item = self.get_item_details(query_item_id)
        if not query_item:
            print(f"아이템을 찾을 수 없습니다: {query_item_id}")
            return []
        
        print(f"기준 아이템 정보:")
        if recommend_type == 'bottom':
            print(f"  - 상의 카테고리: {query_item.get('top_category')}")
            print(f"  - 상의 색상: {query_item.get('top_color')}")
        else:
            print(f"  - 하의 카테고리: {query_item.get('bottom_category')}")
            print(f"  - 하의 색상: {query_item.get('bottom_color')}")
        
        # 2. ChromaDB에서 임베딩 가져오기
        chroma_id = query_item.get('chroma_embedding_id')
        if not chroma_id:
            print("ChromaDB ID가 없습니다")
            return []
        
        # ChromaDB에서 해당 아이템의 임베딩 가져오기
        chroma_result = self.chroma_collection.get(
            ids=[chroma_id],
            include=['embeddings']
        )
        
        if not chroma_result['embeddings']:
            print("임베딩을 찾을 수 없습니다")
            return []
        
        query_embedding = chroma_result['embeddings'][0]
        
        # 3. 유사 아이템 검색 (많이 가져와서 필터링)
        where_filter = {}
        
        # 추천 타입에 따라 필터링
        if recommend_type == 'bottom':
            where_filter['has_bottom'] = 'True'
        else:
            where_filter['has_top'] = 'True'
        
        # 성별 필터
        if gender_filter:
            where_filter['gender'] = gender_filter
        
        # 사용자 필터
        if user_id:
            where_filter['user_id'] = str(user_id)
        
        print(f"\nChromaDB 검색 필터: {where_filter}")
        
        similar_results = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=min(50, n_results * 5),  # 많이 가져와서 필터링
            where=where_filter if where_filter else None
        )
        
        print(f"ChromaDB 검색 결과: {len(similar_results['ids'][0])}개")
        
        # 4. 각 후보 아이템 점수 계산
        candidates = []
        
        for i, (chroma_id, metadata, distance) in enumerate(zip(
            similar_results['ids'][0],
            similar_results['metadatas'][0],
            similar_results['distances'][0]
        )):
            # item_id 추출
            item_id = int(chroma_id.replace('item_', ''))
            
            # 자기 자신 제외
            if item_id == query_item_id:
                continue
            
            # PostgreSQL에서 상세 정보 가져오기
            candidate_item = self.get_item_details(item_id)
            if not candidate_item:
                continue
            
            # 색상 점수 계산
            if recommend_type == 'bottom':
                query_color = query_item.get('top_color')
                candidate_color = candidate_item.get('bottom_color')
                query_style = query_item.get('style')
                candidate_style = candidate_item.get('style')
            else:
                query_color = query_item.get('bottom_color')
                candidate_color = candidate_item.get('top_color')
                query_style = query_item.get('style')
                candidate_style = candidate_item.get('style')
            
            color_score = self.calculate_color_score(query_color, candidate_color)
            style_score = self.calculate_style_score(query_style, candidate_style)
            
            # ChromaDB 유사도 점수 (거리를 유사도로 변환)
            similarity_score = 1 - distance
            
            # 최종 점수 계산 (가중 평균)
            final_score = (
                similarity_score * 0.4 +  # 임베딩 유사도
                color_score * 0.35 +      # 색상 조화
                style_score * 0.25        # 스타일 매칭
            )
            
            candidates.append({
                'item_id': item_id,
                'item': candidate_item,
                'final_score': final_score,
                'similarity_score': similarity_score,
                'color_score': color_score,
                'style_score': style_score,
                'query_color': query_color,
                'candidate_color': candidate_color,
            })
        
        # 5. 점수 기준 정렬
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 6. 상위 n개 반환
        top_candidates = candidates[:n_results]
        
        print(f"\n추천 결과: {len(top_candidates)}개")
        for i, cand in enumerate(top_candidates[:5]):
            print(f"\n{i+1}. 아이템 ID: {cand['item_id']}")
            print(f"   최종 점수: {cand['final_score']:.3f}")
            print(f"   - 유사도: {cand['similarity_score']:.3f}")
            print(f"   - 색상 조화: {cand['color_score']:.3f} ({cand['query_color']} + {cand['candidate_color']})")
            print(f"   - 스타일: {cand['style_score']:.3f}")
            
            if recommend_type == 'bottom':
                print(f"   추천: {cand['item']['bottom_category']} ({cand['item']['bottom_color']})")
            else:
                print(f"   추천: {cand['item']['top_category']} ({cand['item']['top_color']})")
        
        return top_candidates
    
    
    def recommend_complete_outfit(self,
                                  user_id: int,
                                  occasion: Optional[str] = None,
                                  style_preference: Optional[str] = None,
                                  n_results: int = 5) -> List[Dict]:
        """
        전체 코디 추천 (상의 + 하의 조합)
        
        Args:
            user_id: 사용자 ID
            occasion: 상황 (캐주얼, 비즈니스, 데이트 등)
            style_preference: 선호 스타일
            n_results: 추천 개수
        """
        
        print(f"\n=== 전체 코디 추천 ===")
        print(f"사용자 ID: {user_id}")
        
        # 사용자의 전체 옷장 가져오기
        with self.db_conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM wardrobe_full_view
                WHERE user_id = %s
                ORDER BY upload_date DESC
            """, (user_id,))
            
            items = [dict(row) for row in cur.fetchall()]
        
        print(f"사용자 옷장: {len(items)}개 아이템")
        
        # 상의와 하의 분리
        tops = [item for item in items if item['has_top']]
        bottoms = [item for item in items if item['has_bottom']]
        
        print(f"  - 상의: {len(tops)}개")
        print(f"  - 하의: {len(bottoms)}개")
        
        # 모든 조합 평가
        outfit_combinations = []
        
        for top in tops:
            for bottom in bottoms:
                # 같은 아이템이면 스킵
                if top['item_id'] == bottom['item_id']:
                    continue
                
                # 색상 조화
                color_score = self.calculate_color_score(
                    top.get('top_color'),
                    bottom.get('bottom_color')
                )
                
                # 스타일 매칭
                style_score = self.calculate_style_score(
                    top.get('style'),
                    bottom.get('style')
                )
                
                # 최종 점수
                final_score = color_score * 0.6 + style_score * 0.4
                
                outfit_combinations.append({
                    'top_item_id': top['item_id'],
                    'bottom_item_id': bottom['item_id'],
                    'top_info': f"{top['top_category']} ({top['top_color']})",
                    'bottom_info': f"{bottom['bottom_category']} ({bottom['bottom_color']})",
                    'score': final_score,
                    'color_score': color_score,
                    'style_score': style_score
                })
        
        # 점수 순 정렬
        outfit_combinations.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n추천 코디 조합 (상위 {n_results}개):")
        for i, outfit in enumerate(outfit_combinations[:n_results]):
            print(f"\n{i+1}. 점수: {outfit['score']:.3f}")
            print(f"   상의: {outfit['top_info']}")
            print(f"   하의: {outfit['bottom_info']}")
            print(f"   색상 조화: {outfit['color_score']:.3f}")
            print(f"   스타일 매칭: {outfit['style_score']:.3f}")
        
        return outfit_combinations[:n_results]
    
    
    def save_recommendation_history(self, user_id: int, query_item_id: int,
                                   recommended_items: List[int],
                                   recommend_type: str):
        """추천 기록 PostgreSQL에 저장"""
        
        with self.db_conn.cursor() as cur:
            cur.execute("""
                INSERT INTO recommendation_history (
                    user_id, query_item_id, recommended_items,
                    filter_category
                ) VALUES (%s, %s, %s, %s)
                RETURNING rec_id
            """, (user_id, query_item_id, psycopg2.extras.Json(recommended_items), recommend_type))
            
            rec_id = cur.fetchone()[0]
            self.db_conn.commit()
            
        print(f"\n추천 기록 저장됨 (ID: {rec_id})")
        return rec_id
    
    
    def _load_image_with_exif(self, image_path: str) -> Image.Image:
        """EXIF 정보를 고려하여 이미지 로드 및 회전 처리"""
        try:
            img = Image.open(image_path)
            # EXIF orientation 자동 처리
            img = ImageOps.exif_transpose(img)
            return img
        except Exception as e:
            print(f"Image load error: {e}")
            return None
    
    
    def visualize_recommendations(self, 
                                   query_item_id: int,
                                   recommendations: List[Dict],
                                   recommend_type: str = 'bottom',
                                   max_display: int = 5):
        """Visualize recommendation results"""
        
        # Get query item info
        query_item = self.get_item_details(query_item_id)
        if not query_item:
            print("Query item not found")
            return
        
        # Number of items to display
        n_display = min(max_display, len(recommendations))
        
        # Create figure (Row 1: query item, Row 2: recommended items)
        fig, axes = plt.subplots(2, n_display + 1, figsize=(4 * (n_display + 1), 8))
        
        if n_display == 0:
            print("No recommendations")
            return
        
        # Row 1: Display query item
        query_img_path = query_item['original_image_path']
        if query_img_path and Path(query_img_path).exists():
            img = self._load_image_with_exif(query_img_path)
            if img:
                axes[0, 0].imshow(img)
                
                title = "Query Item\n"
                if recommend_type == 'bottom':
                    title += f"Top: {query_item.get('top_category', 'N/A')}\n"
                    title += f"Color: {query_item.get('top_color', 'N/A')}"
                else:
                    title += f"Bottom: {query_item.get('bottom_category', 'N/A')}\n"
                    title += f"Color: {query_item.get('bottom_color', 'N/A')}"
                
                axes[0, 0].set_title(title, fontsize=10, weight='bold')
        axes[0, 0].axis('off')
        
        # Hide remaining cells in row 1
        for i in range(1, n_display + 1):
            axes[0, i].axis('off')
        
        # Row 2: Recommended items
        for i, rec in enumerate(recommendations[:n_display]):
            item = rec['item']
            img_path = item['original_image_path']
            
            if img_path and Path(img_path).exists():
                img = self._load_image_with_exif(img_path)
                if img:
                    axes[1, i].imshow(img)
                    
                    # Display recommendation info
                    title = f"Rec #{i+1}\n"
                    title += f"Score: {rec['final_score']:.2f}\n"
                    
                    if recommend_type == 'bottom':
                        title += f"Bottom: {item.get('bottom_category', 'N/A')}\n"
                        title += f"Color: {item.get('bottom_color', 'N/A')}"
                    else:
                        title += f"Top: {item.get('top_category', 'N/A')}\n"
                        title += f"Color: {item.get('top_color', 'N/A')}"
                    
                    axes[1, i].set_title(title, fontsize=9)
            
            axes[1, i].axis('off')
        
        # Hide remaining cells
        if n_display < max_display:
            axes[1, n_display].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    
    def visualize_outfit_combinations(self, outfits: List[Dict], max_display: int = 3):
        """Visualize complete outfit combinations"""
        
        n_display = min(max_display, len(outfits))
        
        if n_display == 0:
            print("No outfit recommendations")
            return
        
        # Create figure (2 images per outfit: top + bottom)
        fig, axes = plt.subplots(n_display, 3, figsize=(12, 4 * n_display))
        
        if n_display == 1:
            axes = axes.reshape(1, -1)
        
        for i, outfit in enumerate(outfits[:n_display]):
            # Top image
            top_item = self.get_item_details(outfit['top_item_id'])
            if top_item and top_item['original_image_path']:
                img_path = top_item['original_image_path']
                if Path(img_path).exists():
                    img = self._load_image_with_exif(img_path)
                    if img:
                        axes[i, 0].imshow(img)
                        axes[i, 0].set_title(f"Top\n{outfit['top_info']}", fontsize=10)
            axes[i, 0].axis('off')
            
            # Plus sign
            axes[i, 1].text(0.5, 0.5, '+', fontsize=60, ha='center', va='center')
            axes[i, 1].axis('off')
            
            # Bottom image
            bottom_item = self.get_item_details(outfit['bottom_item_id'])
            if bottom_item and bottom_item['original_image_path']:
                img_path = bottom_item['original_image_path']
                if Path(img_path).exists():
                    img = self._load_image_with_exif(img_path)
                    if img:
                        axes[i, 2].imshow(img)
                        axes[i, 2].set_title(f"Bottom\n{outfit['bottom_info']}", fontsize=10)
            axes[i, 2].axis('off')
            
            # Outfit info
            fig.text(0.5, 0.95 - (i * (1.0 / n_display)), 
                    f"Outfit #{i+1} - Score: {outfit['score']:.2f} (Color: {outfit['color_score']:.2f}, Style: {outfit['style_score']:.2f})",
                    ha='center', fontsize=12, weight='bold')
        
        plt.tight_layout()
        plt.show()
    
    
    def close(self):
        """연결 종료"""
        if self.db_conn:
            self.db_conn.close()
            print("PostgreSQL 연결 종료")


# 사용 예시
if __name__ == '__main__':
    # 추천 시스템 초기화
    recommender = FashionRecommender(
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
        # 예시 1: 특정 아이템에 어울리는 하의 추천 + 시각화
        print("\n" + "="*60)
        print("예시 1: 상의에 어울리는 하의 추천")
        print("="*60)
        
        recommendations = recommender.recommend_matching_items(
            query_item_id=1,  # 기준 아이템 ID
            recommend_type='bottom',  # 하의 추천
            user_id=1,  # 특정 사용자의 옷장에서만 검색
            n_results=5
        )
        
        # 시각화
        if recommendations:
            recommender.visualize_recommendations(
                query_item_id=1,
                recommendations=recommendations,
                recommend_type='bottom',
                max_display=5
            )
        
        # 예시 2: 전체 코디 조합 추천 + 시각화
        print("\n" + "="*60)
        print("예시 2: 전체 코디 조합 추천")
        print("="*60)
        
        outfits = recommender.recommend_complete_outfit(
            user_id=1,
            n_results=5
        )
        
        # 시각화
        if outfits:
            recommender.visualize_outfit_combinations(
                outfits=outfits,
                max_display=3
            )
        
    finally:
        recommender.close()
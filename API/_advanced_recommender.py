"""
고급 패션 추천 시스템
- 색상 조화, 소재 조합, 핏 조합, 스타일 조합, 계절별 적합성을 모두 고려한 추천
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import colorsys
import math

@dataclass
class FashionItem:
    """패션 아이템 정보"""
    item_id: int
    category: str  # top, bottom, outer, dress
    subcategory: str  # 티셔츠, 청바지, 재킷 등
    color: str
    fit: str  # tight, normal, loose, oversized
    materials: List[str]
    style: str  # casual, formal, sporty, elegant
    season: str  # spring, summer, fall, winter
    is_default: bool = False
    image_path: str = None  # 이미지 파일 경로

@dataclass
class RecommendationScore:
    """추천 점수 세부 정보"""
    total_score: float
    color_harmony: float
    material_combination: float
    fit_combination: float
    style_combination: float
    seasonal_suitability: float
    category_compatibility: float

class AdvancedFashionRecommender:
    """고급 패션 추천 시스템"""
    
    def __init__(self):
        self.color_harmony_weights = {
            'complementary': 0.9,    # 보색 조화
            'analogous': 0.8,        # 유사색 조화
            'triadic': 0.7,          # 삼원색 조화
            'monochromatic': 0.6,    # 단색 조화
            'neutral': 0.5,          # 중성색 조화
            'clashing': 0.2          # 색상 충돌
        }
        
        self.material_combinations = {
            # 상의-하의 소재 조합
            ('cotton', 'denim'): 0.9,
            ('cotton', 'cotton'): 0.8,
            ('silk', 'wool'): 0.7,
            ('knit', 'denim'): 0.8,
            ('leather', 'denim'): 0.6,
            ('wool', 'wool'): 0.9,
            ('linen', 'cotton'): 0.8,
            ('polyester', 'polyester'): 0.5,
            # 아우터 조합
            ('wool', 'cotton'): 0.8,
            ('leather', 'cotton'): 0.7,
            ('denim', 'cotton'): 0.8,
            ('knit', 'cotton'): 0.9,
        }
        
        self.fit_combinations = {
            # 핏 조합 (상의-하의)
            ('tight', 'loose'): 0.9,      # 타이트 상의 + 루즈 하의
            ('loose', 'tight'): 0.9,      # 루즈 상의 + 타이트 하의
            ('normal', 'normal'): 0.8,    # 노멀 + 노멀
            ('tight', 'tight'): 0.6,      # 타이트 + 타이트
            ('loose', 'loose'): 0.5,      # 루즈 + 루즈
            ('oversized', 'tight'): 0.8,  # 오버사이즈 + 타이트
            ('tight', 'oversized'): 0.7,  # 타이트 + 오버사이즈
        }
        
        # 24개 스타일 조합 매트릭스 (한국어 스타일명 지원)
        self.style_combinations = {
            # 기타
            ('기타', '기타'): 1.0, ('기타', '레트로'): 0.6, ('기타', '로맨틱'): 0.5, ('기타', '리조트'): 0.7, ('기타', '매니시'): 0.6, ('기타', '모던'): 0.7, ('기타', '밀리터리'): 0.5, ('기타', '섹시'): 0.6, ('기타', '소피스트케이티드'): 0.6, ('기타', '스트리트'): 0.7, ('기타', '스포티'): 0.6, ('기타', '아방가르드'): 0.8, ('기타', '오리엔탈'): 0.6, ('기타', '웨스턴'): 0.5, ('기타', '젠더리스'): 0.7, ('기타', '컨트리'): 0.6, ('기타', '클래식'): 0.6, ('기타', '키치'): 0.8, ('기타', '톰보이'): 0.6, ('기타', '펑크'): 0.7, ('기타', '페미닌'): 0.5, ('기타', '프레피'): 0.6, ('기타', '히피'): 0.7, ('기타', '힙합'): 0.7,
            # 레트로
            ('레트로', '기타'): 0.6, ('레트로', '레트로'): 1.0, ('레트로', '로맨틱'): 0.8, ('레트로', '리조트'): 0.6, ('레트로', '매니시'): 0.7, ('레트로', '모던'): 0.4, ('레트로', '밀리터리'): 0.5, ('레트로', '섹시'): 0.6, ('레트로', '소피스트케이티드'): 0.7, ('레트로', '스트리트'): 0.5, ('레트로', '스포티'): 0.4, ('레트로', '아방가르드'): 0.6, ('레트로', '오리엔탈'): 0.7, ('레트로', '웨스턴'): 0.6, ('레트로', '젠더리스'): 0.6, ('레트로', '컨트리'): 0.7, ('레트로', '클래식'): 0.9, ('레트로', '키치'): 0.8, ('레트로', '톰보이'): 0.6, ('레트로', '펑크'): 0.5, ('레트로', '페미닌'): 0.8, ('레트로', '프레피'): 0.7, ('레트로', '히피'): 0.8, ('레트로', '힙합'): 0.4,
            # 로맨틱
            ('로맨틱', '기타'): 0.5, ('로맨틱', '레트로'): 0.8, ('로맨틱', '로맨틱'): 1.0, ('로맨틱', '리조트'): 0.8, ('로맨틱', '매니시'): 0.3, ('로맨틱', '모던'): 0.4, ('로맨틱', '밀리터리'): 0.2, ('로맨틱', '섹시'): 0.7, ('로맨틱', '소피스트케이티드'): 0.8, ('로맨틱', '스트리트'): 0.3, ('로맨틱', '스포티'): 0.2, ('로맨틱', '아방가르드'): 0.4, ('로맨틱', '오리엔탈'): 0.8, ('로맨틱', '웨스턴'): 0.3, ('로맨틱', '젠더리스'): 0.4, ('로맨틱', '컨트리'): 0.6, ('로맨틱', '클래식'): 0.9, ('로맨틱', '키치'): 0.6, ('로맨틱', '톰보이'): 0.3, ('로맨틱', '펑크'): 0.2, ('로맨틱', '페미닌'): 1.0, ('로맨틱', '프레피'): 0.6, ('로맨틱', '히피'): 0.5, ('로맨틱', '힙합'): 0.2,
            # 리조트
            ('리조트', '기타'): 0.7, ('리조트', '레트로'): 0.6, ('리조트', '로맨틱'): 0.8, ('리조트', '리조트'): 1.0, ('리조트', '매니시'): 0.5, ('리조트', '모던'): 0.7, ('리조트', '밀리터리'): 0.4, ('리조트', '섹시'): 0.6, ('리조트', '소피스트케이티드'): 0.7, ('리조트', '스트리트'): 0.5, ('리조트', '스포티'): 0.8, ('리조트', '아방가르드'): 0.5, ('리조트', '오리엔탈'): 0.6, ('리조트', '웨스턴'): 0.5, ('리조트', '젠더리스'): 0.6, ('리조트', '컨트리'): 0.7, ('리조트', '클래식'): 0.6, ('리조트', '키치'): 0.5, ('리조트', '톰보이'): 0.5, ('리조트', '펑크'): 0.4, ('리조트', '페미닌'): 0.7, ('리조트', '프레피'): 0.7, ('리조트', '히피'): 0.6, ('리조트', '힙합'): 0.4,
            # 매니시
            ('매니시', '기타'): 0.6, ('매니시', '레트로'): 0.7, ('매니시', '로맨틱'): 0.3, ('매니시', '리조트'): 0.5, ('매니시', '매니시'): 1.0, ('매니시', '모던'): 0.8, ('매니시', '밀리터리'): 0.9, ('매니시', '섹시'): 0.4, ('매니시', '소피스트케이티드'): 0.7, ('매니시', '스트리트'): 0.6, ('매니시', '스포티'): 0.7, ('매니시', '아방가르드'): 0.6, ('매니시', '오리엔탈'): 0.5, ('매니시', '웨스턴'): 0.7, ('매니시', '젠더리스'): 0.9, ('매니시', '컨트리'): 0.6, ('매니시', '클래식'): 0.8, ('매니시', '키치'): 0.4, ('매니시', '톰보이'): 0.9, ('매니시', '펑크'): 0.6, ('매니시', '페미닌'): 0.3, ('매니시', '프레피'): 0.8, ('매니시', '히피'): 0.5, ('매니시', '힙합'): 0.5,
            # 모던
            ('모던', '기타'): 0.7, ('모던', '레트로'): 0.4, ('모던', '로맨틱'): 0.4, ('모던', '리조트'): 0.7, ('모던', '매니시'): 0.8, ('모던', '모던'): 1.0, ('모던', '밀리터리'): 0.6, ('모던', '섹시'): 0.5, ('모던', '소피스트케이티드'): 0.9, ('모던', '스트리트'): 0.7, ('모던', '스포티'): 0.6, ('모던', '아방가르드'): 0.8, ('모던', '오리엔탈'): 0.6, ('모던', '웨스턴'): 0.4, ('모던', '젠더리스'): 0.8, ('모던', '컨트리'): 0.4, ('모던', '클래식'): 0.8, ('모던', '키치'): 0.5, ('모던', '톰보이'): 0.7, ('모던', '펑크'): 0.5, ('모던', '페미닌'): 0.5, ('모던', '프레피'): 0.7, ('모던', '히피'): 0.4, ('모던', '힙합'): 0.6,
            # 밀리터리
            ('밀리터리', '기타'): 0.5, ('밀리터리', '레트로'): 0.5, ('밀리터리', '로맨틱'): 0.2, ('밀리터리', '리조트'): 0.4, ('밀리터리', '매니시'): 0.9, ('밀리터리', '모던'): 0.6, ('밀리터리', '밀리터리'): 1.0, ('밀리터리', '섹시'): 0.3, ('밀리터리', '소피스트케이티드'): 0.5, ('밀리터리', '스트리트'): 0.6, ('밀리터리', '스포티'): 0.7, ('밀리터리', '아방가르드'): 0.5, ('밀리터리', '오리엔탈'): 0.4, ('밀리터리', '웨스턴'): 0.6, ('밀리터리', '젠더리스'): 0.8, ('밀리터리', '컨트리'): 0.5, ('밀리터리', '클래식'): 0.6, ('밀리터리', '키치'): 0.3, ('밀리터리', '톰보이'): 0.8, ('밀리터리', '펑크'): 0.7, ('밀리터리', '페미닌'): 0.2, ('밀리터리', '프레피'): 0.6, ('밀리터리', '히피'): 0.3, ('밀리터리', '힙합'): 0.5,
            # 섹시
            ('섹시', '기타'): 0.6, ('섹시', '레트로'): 0.6, ('섹시', '로맨틱'): 0.7, ('섹시', '리조트'): 0.6, ('섹시', '매니시'): 0.4, ('섹시', '모던'): 0.5, ('섹시', '밀리터리'): 0.3, ('섹시', '섹시'): 1.0, ('섹시', '소피스트케이티드'): 0.6, ('섹시', '스트리트'): 0.5, ('섹시', '스포티'): 0.3, ('섹시', '아방가르드'): 0.6, ('섹시', '오리엔탈'): 0.6, ('섹시', '웨스턴'): 0.4, ('섹시', '젠더리스'): 0.4, ('섹시', '컨트리'): 0.4, ('섹시', '클래식'): 0.5, ('섹시', '키치'): 0.5, ('섹시', '톰보이'): 0.3, ('섹시', '펑크'): 0.6, ('섹시', '페미닌'): 0.9, ('섹시', '프레피'): 0.4, ('섹시', '히피'): 0.4, ('섹시', '힙합'): 0.4,
            # 소피스트케이티드
            ('소피스트케이티드', '기타'): 0.6, ('소피스트케이티드', '레트로'): 0.7, ('소피스트케이티드', '로맨틱'): 0.8, ('소피스트케이티드', '리조트'): 0.7, ('소피스트케이티드', '매니시'): 0.7, ('소피스트케이티드', '모던'): 0.9, ('소피스트케이티드', '밀리터리'): 0.5, ('소피스트케이티드', '섹시'): 0.6, ('소피스트케이티드', '소피스트케이티드'): 1.0, ('소피스트케이티드', '스트리트'): 0.4, ('소피스트케이티드', '스포티'): 0.4, ('소피스트케이티드', '아방가르드'): 0.6, ('소피스트케이티드', '오리엔탈'): 0.8, ('소피스트케이티드', '웨스턴'): 0.5, ('소피스트케이티드', '젠더리스'): 0.6, ('소피스트케이티드', '컨트리'): 0.5, ('소피스트케이티드', '클래식'): 0.9, ('소피스트케이티드', '키치'): 0.4, ('소피스트케이티드', '톰보이'): 0.6, ('소피스트케이티드', '펑크'): 0.3, ('소피스트케이티드', '페미닌'): 0.8, ('소피스트케이티드', '프레피'): 0.8, ('소피스트케이티드', '히피'): 0.4, ('소피스트케이티드', '힙합'): 0.3,
            # 스트리트
            ('스트리트', '기타'): 0.7, ('스트리트', '레트로'): 0.5, ('스트리트', '로맨틱'): 0.3, ('스트리트', '리조트'): 0.5, ('스트리트', '매니시'): 0.6, ('스트리트', '모던'): 0.7, ('스트리트', '밀리터리'): 0.6, ('스트리트', '섹시'): 0.5, ('스트리트', '소피스트케이티드'): 0.4, ('스트리트', '스트리트'): 1.0, ('스트리트', '스포티'): 0.7, ('스트리트', '아방가르드'): 0.7, ('스트리트', '오리엔탈'): 0.5, ('스트리트', '웨스턴'): 0.5, ('스트리트', '젠더리스'): 0.7, ('스트리트', '컨트리'): 0.4, ('스트리트', '클래식'): 0.4, ('스트리트', '키치'): 0.6, ('스트리트', '톰보이'): 0.7, ('스트리트', '펑크'): 0.8, ('스트리트', '페미닌'): 0.3, ('스트리트', '프레피'): 0.5, ('스트리트', '히피'): 0.6, ('스트리트', '힙합'): 0.9,
            # 스포티
            ('스포티', '기타'): 0.6, ('스포티', '레트로'): 0.4, ('스포티', '로맨틱'): 0.2, ('스포티', '리조트'): 0.8, ('스포티', '매니시'): 0.7, ('스포티', '모던'): 0.6, ('스포티', '밀리터리'): 0.7, ('스포티', '섹시'): 0.3, ('스포티', '소피스트케이티드'): 0.4, ('스포티', '스트리트'): 0.7, ('스포티', '스포티'): 1.0, ('스포티', '아방가르드'): 0.4, ('스포티', '오리엔탈'): 0.5, ('스포티', '웨스턴'): 0.6, ('스포티', '젠더리스'): 0.8, ('스포티', '컨트리'): 0.6, ('스포티', '클래식'): 0.4, ('스포티', '키치'): 0.4, ('스포티', '톰보이'): 0.8, ('스포티', '펑크'): 0.5, ('스포티', '페미닌'): 0.2, ('스포티', '프레피'): 0.6, ('스포티', '히피'): 0.5, ('스포티', '힙합'): 0.6,
            # 아방가르드
            ('아방가르드', '기타'): 0.8, ('아방가르드', '레트로'): 0.6, ('아방가르드', '로맨틱'): 0.4, ('아방가르드', '리조트'): 0.5, ('아방가르드', '매니시'): 0.6, ('아방가르드', '모던'): 0.8, ('아방가르드', '밀리터리'): 0.5, ('아방가르드', '섹시'): 0.6, ('아방가르드', '소피스트케이티드'): 0.6, ('아방가르드', '스트리트'): 0.7, ('아방가르드', '스포티'): 0.4, ('아방가르드', '아방가르드'): 1.0, ('아방가르드', '오리엔탈'): 0.6, ('아방가르드', '웨스턴'): 0.4, ('아방가르드', '젠더리스'): 0.7, ('아방가르드', '컨트리'): 0.4, ('아방가르드', '클래식'): 0.4, ('아방가르드', '키치'): 0.8, ('아방가르드', '톰보이'): 0.6, ('아방가르드', '펑크'): 0.7, ('아방가르드', '페미닌'): 0.4, ('아방가르드', '프레피'): 0.4, ('아방가르드', '히피'): 0.6, ('아방가르드', '힙합'): 0.5,
            # 오리엔탈
            ('오리엔탈', '기타'): 0.6, ('오리엔탈', '레트로'): 0.7, ('오리엔탈', '로맨틱'): 0.8, ('오리엔탈', '리조트'): 0.6, ('오리엔탈', '매니시'): 0.5, ('오리엔탈', '모던'): 0.6, ('오리엔탈', '밀리터리'): 0.4, ('오리엔탈', '섹시'): 0.6, ('오리엔탈', '소피스트케이티드'): 0.8, ('오리엔탈', '스트리트'): 0.5, ('오리엔탈', '스포티'): 0.5, ('오리엔탈', '아방가르드'): 0.6, ('오리엔탈', '오리엔탈'): 1.0, ('오리엔탈', '웨스턴'): 0.3, ('오리엔탈', '젠더리스'): 0.5, ('오리엔탈', '컨트리'): 0.5, ('오리엔탈', '클래식'): 0.8, ('오리엔탈', '키치'): 0.5, ('오리엔탈', '톰보이'): 0.4, ('오리엔탈', '펑크'): 0.4, ('오리엔탈', '페미닌'): 0.7, ('오리엔탈', '프레피'): 0.6, ('오리엔탈', '히피'): 0.5, ('오리엔탈', '힙합'): 0.3,
            # 웨스턴
            ('웨스턴', '기타'): 0.5, ('웨스턴', '레트로'): 0.6, ('웨스턴', '로맨틱'): 0.3, ('웨스턴', '리조트'): 0.5, ('웨스턴', '매니시'): 0.7, ('웨스턴', '모던'): 0.4, ('웨스턴', '밀리터리'): 0.6, ('웨스턴', '섹시'): 0.4, ('웨스턴', '소피스트케이티드'): 0.5, ('웨스턴', '스트리트'): 0.5, ('웨스턴', '스포티'): 0.6, ('웨스턴', '아방가르드'): 0.4, ('웨스턴', '오리엔탈'): 0.3, ('웨스턴', '웨스턴'): 1.0, ('웨스턴', '젠더리스'): 0.6, ('웨스턴', '컨트리'): 0.9, ('웨스턴', '클래식'): 0.5, ('웨스턴', '키치'): 0.4, ('웨스턴', '톰보이'): 0.7, ('웨스턴', '펑크'): 0.5, ('웨스턴', '페미닌'): 0.3, ('웨스턴', '프레피'): 0.5, ('웨스턴', '히피'): 0.5, ('웨스턴', '힙합'): 0.4,
            # 젠더리스
            ('젠더리스', '기타'): 0.7, ('젠더리스', '레트로'): 0.6, ('젠더리스', '로맨틱'): 0.4, ('젠더리스', '리조트'): 0.6, ('젠더리스', '매니시'): 0.9, ('젠더리스', '모던'): 0.8, ('젠더리스', '밀리터리'): 0.8, ('젠더리스', '섹시'): 0.4, ('젠더리스', '소피스트케이티드'): 0.6, ('젠더리스', '스트리트'): 0.7, ('젠더리스', '스포티'): 0.8, ('젠더리스', '아방가르드'): 0.7, ('젠더리스', '오리엔탈'): 0.5, ('젠더리스', '웨스턴'): 0.6, ('젠더리스', '젠더리스'): 1.0, ('젠더리스', '컨트리'): 0.5, ('젠더리스', '클래식'): 0.6, ('젠더리스', '키치'): 0.5, ('젠더리스', '톰보이'): 0.9, ('젠더리스', '펑크'): 0.6, ('젠더리스', '페미닌'): 0.4, ('젠더리스', '프레피'): 0.6, ('젠더리스', '히피'): 0.6, ('젠더리스', '힙합'): 0.6,
            # 컨트리
            ('컨트리', '기타'): 0.6, ('컨트리', '레트로'): 0.7, ('컨트리', '로맨틱'): 0.6, ('컨트리', '리조트'): 0.7, ('컨트리', '매니시'): 0.6, ('컨트리', '모던'): 0.4, ('컨트리', '밀리터리'): 0.5, ('컨트리', '섹시'): 0.4, ('컨트리', '소피스트케이티드'): 0.5, ('컨트리', '스트리트'): 0.4, ('컨트리', '스포티'): 0.6, ('컨트리', '아방가르드'): 0.4, ('컨트리', '오리엔탈'): 0.5, ('컨트리', '웨스턴'): 0.9, ('컨트리', '젠더리스'): 0.5, ('컨트리', '컨트리'): 1.0, ('컨트리', '클래식'): 0.6, ('컨트리', '키치'): 0.5, ('컨트리', '톰보이'): 0.6, ('컨트리', '펑크'): 0.4, ('컨트리', '페미닌'): 0.5, ('컨트리', '프레피'): 0.6, ('컨트리', '히피'): 0.7, ('컨트리', '힙합'): 0.3,
            # 클래식
            ('클래식', '기타'): 0.6, ('클래식', '레트로'): 0.9, ('클래식', '로맨틱'): 0.9, ('클래식', '리조트'): 0.6, ('클래식', '매니시'): 0.8, ('클래식', '모던'): 0.8, ('클래식', '밀리터리'): 0.6, ('클래식', '섹시'): 0.5, ('클래식', '소피스트케이티드'): 0.9, ('클래식', '스트리트'): 0.4, ('클래식', '스포티'): 0.4, ('클래식', '아방가르드'): 0.4, ('클래식', '오리엔탈'): 0.8, ('클래식', '웨스턴'): 0.5, ('클래식', '젠더리스'): 0.6, ('클래식', '컨트리'): 0.6, ('클래식', '클래식'): 1.0, ('클래식', '키치'): 0.4, ('클래식', '톰보이'): 0.6, ('클래식', '펑크'): 0.3, ('클래식', '페미닌'): 0.8, ('클래식', '프레피'): 0.9, ('클래식', '히피'): 0.4, ('클래식', '힙합'): 0.3,
            # 키치
            ('키치', '기타'): 0.8, ('키치', '레트로'): 0.8, ('키치', '로맨틱'): 0.6, ('키치', '리조트'): 0.5, ('키치', '매니시'): 0.4, ('키치', '모던'): 0.5, ('키치', '밀리터리'): 0.3, ('키치', '섹시'): 0.5, ('키치', '소피스트케이티드'): 0.4, ('키치', '스트리트'): 0.6, ('키치', '스포티'): 0.4, ('키치', '아방가르드'): 0.8, ('키치', '오리엔탈'): 0.5, ('키치', '웨스턴'): 0.4, ('키치', '젠더리스'): 0.5, ('키치', '컨트리'): 0.5, ('키치', '클래식'): 0.4, ('키치', '키치'): 1.0, ('키치', '톰보이'): 0.4, ('키치', '펑크'): 0.6, ('키치', '페미닌'): 0.6, ('키치', '프레피'): 0.4, ('키치', '히피'): 0.7, ('키치', '힙합'): 0.5,
            # 톰보이
            ('톰보이', '기타'): 0.6, ('톰보이', '레트로'): 0.6, ('톰보이', '로맨틱'): 0.3, ('톰보이', '리조트'): 0.5, ('톰보이', '매니시'): 0.9, ('톰보이', '모던'): 0.7, ('톰보이', '밀리터리'): 0.8, ('톰보이', '섹시'): 0.3, ('톰보이', '소피스트케이티드'): 0.6, ('톰보이', '스트리트'): 0.7, ('톰보이', '스포티'): 0.8, ('톰보이', '아방가르드'): 0.6, ('톰보이', '오리엔탈'): 0.4, ('톰보이', '웨스턴'): 0.7, ('톰보이', '젠더리스'): 0.9, ('톰보이', '컨트리'): 0.6, ('톰보이', '클래식'): 0.6, ('톰보이', '키치'): 0.4, ('톰보이', '톰보이'): 1.0, ('톰보이', '펑크'): 0.6, ('톰보이', '페미닌'): 0.3, ('톰보이', '프레피'): 0.7, ('톰보이', '히피'): 0.5, ('톰보이', '힙합'): 0.5,
            # 펑크
            ('펑크', '기타'): 0.7, ('펑크', '레트로'): 0.5, ('펑크', '로맨틱'): 0.2, ('펑크', '리조트'): 0.4, ('펑크', '매니시'): 0.6, ('펑크', '모던'): 0.5, ('펑크', '밀리터리'): 0.7, ('펑크', '섹시'): 0.6, ('펑크', '소피스트케이티드'): 0.3, ('펑크', '스트리트'): 0.8, ('펑크', '스포티'): 0.5, ('펑크', '아방가르드'): 0.7, ('펑크', '오리엔탈'): 0.4, ('펑크', '웨스턴'): 0.5, ('펑크', '젠더리스'): 0.6, ('펑크', '컨트리'): 0.4, ('펑크', '클래식'): 0.3, ('펑크', '키치'): 0.6, ('펑크', '톰보이'): 0.6, ('펑크', '펑크'): 1.0, ('펑크', '페미닌'): 0.3, ('펑크', '프레피'): 0.4, ('펑크', '히피'): 0.6, ('펑크', '힙합'): 0.7,
            # 페미닌
            ('페미닌', '기타'): 0.5, ('페미닌', '레트로'): 0.8, ('페미닌', '로맨틱'): 1.0, ('페미닌', '리조트'): 0.7, ('페미닌', '매니시'): 0.3, ('페미닌', '모던'): 0.5, ('페미닌', '밀리터리'): 0.2, ('페미닌', '섹시'): 0.9, ('페미닌', '소피스트케이티드'): 0.8, ('페미닌', '스트리트'): 0.3, ('페미닌', '스포티'): 0.2, ('페미닌', '아방가르드'): 0.4, ('페미닌', '오리엔탈'): 0.7, ('페미닌', '웨스턴'): 0.3, ('페미닌', '젠더리스'): 0.4, ('페미닌', '컨트리'): 0.5, ('페미닌', '클래식'): 0.8, ('페미닌', '키치'): 0.6, ('페미닌', '톰보이'): 0.3, ('페미닌', '펑크'): 0.3, ('페미닌', '페미닌'): 1.0, ('페미닌', '프레피'): 0.6, ('페미닌', '히피'): 0.4, ('페미닌', '힙합'): 0.2,
            # 프레피
            ('프레피', '기타'): 0.6, ('프레피', '레트로'): 0.7, ('프레피', '로맨틱'): 0.6, ('프레피', '리조트'): 0.7, ('프레피', '매니시'): 0.8, ('프레피', '모던'): 0.7, ('프레피', '밀리터리'): 0.6, ('프레피', '섹시'): 0.4, ('프레피', '소피스트케이티드'): 0.8, ('프레피', '스트리트'): 0.5, ('프레피', '스포티'): 0.6, ('프레피', '아방가르드'): 0.4, ('프레피', '오리엔탈'): 0.6, ('프레피', '웨스턴'): 0.5, ('프레피', '젠더리스'): 0.6, ('프레피', '컨트리'): 0.6, ('프레피', '클래식'): 0.9, ('프레피', '키치'): 0.4, ('프레피', '톰보이'): 0.7, ('프레피', '펑크'): 0.4, ('프레피', '페미닌'): 0.6, ('프레피', '프레피'): 1.0, ('프레피', '히피'): 0.4, ('프레피', '힙합'): 0.4,
            # 히피
            ('히피', '기타'): 0.7, ('히피', '레트로'): 0.8, ('히피', '로맨틱'): 0.5, ('히피', '리조트'): 0.6, ('히피', '매니시'): 0.5, ('히피', '모던'): 0.4, ('히피', '밀리터리'): 0.3, ('히피', '섹시'): 0.4, ('히피', '소피스트케이티드'): 0.4, ('히피', '스트리트'): 0.6, ('히피', '스포티'): 0.5, ('히피', '아방가르드'): 0.6, ('히피', '오리엔탈'): 0.5, ('히피', '웨스턴'): 0.5, ('히피', '젠더리스'): 0.6, ('히피', '컨트리'): 0.7, ('히피', '클래식'): 0.4, ('히피', '키치'): 0.7, ('히피', '톰보이'): 0.5, ('히피', '펑크'): 0.6, ('히피', '페미닌'): 0.4, ('히피', '프레피'): 0.4, ('히피', '히피'): 1.0, ('히피', '힙합'): 0.5,
            # 힙합
            ('힙합', '기타'): 0.7, ('힙합', '레트로'): 0.4, ('힙합', '로맨틱'): 0.2, ('힙합', '리조트'): 0.4, ('힙합', '매니시'): 0.5, ('힙합', '모던'): 0.6, ('힙합', '밀리터리'): 0.5, ('힙합', '섹시'): 0.4, ('힙합', '소피스트케이티드'): 0.3, ('힙합', '스트리트'): 0.9, ('힙합', '스포티'): 0.6, ('힙합', '아방가르드'): 0.5, ('힙합', '오리엔탈'): 0.3, ('힙합', '웨스턴'): 0.4, ('힙합', '젠더리스'): 0.6, ('힙합', '컨트리'): 0.3, ('힙합', '클래식'): 0.3, ('힙합', '키치'): 0.5, ('힙합', '톰보이'): 0.5, ('힙합', '펑크'): 0.7, ('힙합', '페미닌'): 0.2, ('힙합', '프레피'): 0.4, ('힙합', '히피'): 0.5, ('힙합', '힙합'): 1.0,
        }
        
        self.seasonal_suitability = {
            # 계절별 적합성
            'spring': {
                'materials': {'cotton': 0.9, 'linen': 0.8, 'silk': 0.7, 'wool': 0.3},
                'colors': {'pastel': 0.9, 'bright': 0.8, 'neutral': 0.7, 'dark': 0.4},
                'fits': {'normal': 0.9, 'loose': 0.8, 'tight': 0.7, 'oversized': 0.6}
            },
            'summer': {
                'materials': {'cotton': 0.9, 'linen': 0.9, 'silk': 0.8, 'wool': 0.1},
                'colors': {'bright': 0.9, 'pastel': 0.8, 'neutral': 0.7, 'dark': 0.3},
                'fits': {'loose': 0.9, 'normal': 0.8, 'oversized': 0.7, 'tight': 0.5}
            },
            'fall': {
                'materials': {'wool': 0.9, 'cotton': 0.7, 'leather': 0.8, 'silk': 0.5},
                'colors': {'neutral': 0.9, 'dark': 0.8, 'warm': 0.8, 'bright': 0.4},
                'fits': {'normal': 0.9, 'loose': 0.7, 'tight': 0.8, 'oversized': 0.6}
            },
            'winter': {
                'materials': {'wool': 0.9, 'leather': 0.8, 'knit': 0.9, 'cotton': 0.4},
                'colors': {'dark': 0.9, 'neutral': 0.8, 'warm': 0.7, 'bright': 0.3},
                'fits': {'normal': 0.9, 'tight': 0.8, 'loose': 0.6, 'oversized': 0.7}
            }
        }
        
        self.category_compatibility = {
            # 카테고리 조합 적합성
            ('top', 'bottom'): 0.9,
            ('top', 'outer'): 0.8,
            ('bottom', 'outer'): 0.7,
            ('dress', 'outer'): 0.6,
            ('dress', 'bottom'): 0.5,
            ('outer', 'top'): 0.8,
            ('outer', 'bottom'): 0.7,
        }
        
        # 색상 매핑 (한국어 → 영어)
        self.color_mapping = {
            '블랙': 'black', '화이트': 'white', '네이비': 'navy', '그레이': 'gray',
            '베이지': 'beige', '브라운': 'brown', '레드': 'red', '핑크': 'pink',
            '블루': 'blue', '그린': 'green', '옐로우': 'yellow', '퍼플': 'purple',
            '오렌지': 'orange', '실버': 'silver', '골드': 'gold', 'none': 'neutral'
        }
        
        # 소재 매핑 (한국어 → 영어)
        self.material_mapping = {
            '저지': 'cotton', '우븐': 'cotton', '데님': 'denim', '니트': 'knit',
            '가죽': 'leather', '실크': 'silk', '린넨': 'linen', '울': 'wool',
            '폴리에스터': 'polyester', '레이스': 'lace', '시폰': 'chiffon'
        }
        
        # 핏 매핑 (한국어 → 영어)
        self.fit_mapping = {
            '타이트': 'tight', '노멀': 'normal', '루즈': 'loose', '오버사이즈': 'oversized',
            '와이드': 'loose', '스키니': 'tight', '레귤러': 'normal'
        }

    def hex_to_hsv(self, hex_color: str) -> Tuple[float, float, float]:
        """HEX 색상을 HSV로 변환"""
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        
        return colorsys.rgb_to_hsv(r, g, b)

    def get_color_harmony_score(self, color1: str, color2: str) -> float:
        """두 색상 간의 조화 점수 계산"""
        # 색상 매핑
        color1_en = self.color_mapping.get(color1.lower(), 'neutral')
        color2_en = self.color_mapping.get(color2.lower(), 'neutral')
        
        # 중성색 처리
        if color1_en == 'neutral' or color2_en == 'neutral':
            return self.color_harmony_weights['neutral']
        
        # 기본 색상 조화 규칙
        harmony_rules = {
            # 보색 조화
            ('red', 'green'): 'complementary',
            ('blue', 'orange'): 'complementary',
            ('yellow', 'purple'): 'complementary',
            
            # 유사색 조화
            ('red', 'orange'): 'analogous',
            ('orange', 'yellow'): 'analogous',
            ('yellow', 'green'): 'analogous',
            ('green', 'blue'): 'analogous',
            ('blue', 'purple'): 'analogous',
            ('purple', 'red'): 'analogous',
            
            # 단색 조화
            ('black', 'white'): 'monochromatic',
            ('black', 'gray'): 'monochromatic',
            ('white', 'gray'): 'monochromatic',
            ('navy', 'blue'): 'monochromatic',
            ('brown', 'beige'): 'monochromatic',
        }
        
        # 조화 유형 찾기
        harmony_type = harmony_rules.get((color1_en, color2_en)) or \
                      harmony_rules.get((color2_en, color1_en))
        
        if harmony_type:
            return self.color_harmony_weights[harmony_type]
        
        # 기본 점수 (색상이 다르면 중간 점수)
        return 0.5

    def get_material_combination_score(self, materials1: List[str], materials2: List[str]) -> float:
        """소재 조합 점수 계산"""
        if not materials1 or not materials2:
            return 0.5
        
        # 소재 매핑
        mats1_en = [self.material_mapping.get(mat.lower(), mat.lower()) for mat in materials1]
        mats2_en = [self.material_mapping.get(mat.lower(), mat.lower()) for mat in materials2]
        
        max_score = 0.0
        for mat1 in mats1_en:
            for mat2 in mats2_en:
                # 직접 조합 찾기
                score = self.material_combinations.get((mat1, mat2), 0.0)
                if score == 0.0:
                    # 역순 조합 찾기
                    score = self.material_combinations.get((mat2, mat1), 0.0)
                
                max_score = max(max_score, score)
        
        return max_score if max_score > 0 else 0.5

    def get_fit_combination_score(self, fit1: str, fit2: str) -> float:
        """핏 조합 점수 계산"""
        fit1_en = self.fit_mapping.get(fit1.lower(), fit1.lower())
        fit2_en = self.fit_mapping.get(fit2.lower(), fit2.lower())
        
        return self.fit_combinations.get((fit1_en, fit2_en), 0.5)

    def get_style_combination_score(self, style1: str, style2: str) -> float:
        """스타일 조합 점수 계산"""
        return self.style_combinations.get((style1.lower(), style2.lower()), 0.5)

    def get_seasonal_suitability_score(self, item: FashionItem, season: str) -> float:
        """계절별 적합성 점수 계산"""
        if season not in self.seasonal_suitability:
            return 0.5
        
        season_data = self.seasonal_suitability[season]
        
        # 소재 점수
        material_score = 0.0
        if item.materials:
            for material in item.materials:
                mat_en = self.material_mapping.get(material.lower(), material.lower())
                material_score += season_data['materials'].get(mat_en, 0.5)
            material_score /= len(item.materials)
        else:
            material_score = 0.5
        
        # 색상 점수
        color_en = self.color_mapping.get(item.color.lower(), 'neutral')
        color_type = self._get_color_type(color_en)
        color_score = season_data['colors'].get(color_type, 0.5)
        
        # 핏 점수
        fit_en = self.fit_mapping.get(item.fit.lower(), item.fit.lower())
        fit_score = season_data['fits'].get(fit_en, 0.5)
        
        # 가중 평균
        return (material_score * 0.4 + color_score * 0.3 + fit_score * 0.3)

    def _get_color_type(self, color: str) -> str:
        """색상을 타입으로 분류"""
        bright_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink']
        pastel_colors = ['light blue', 'light pink', 'light green', 'light yellow']
        dark_colors = ['black', 'navy', 'dark blue', 'dark green', 'dark red']
        warm_colors = ['red', 'orange', 'yellow', 'brown', 'beige']
        neutral_colors = ['white', 'black', 'gray', 'beige', 'brown']
        
        if color in bright_colors:
            return 'bright'
        elif color in pastel_colors:
            return 'pastel'
        elif color in dark_colors:
            return 'dark'
        elif color in warm_colors:
            return 'warm'
        elif color in neutral_colors:
            return 'neutral'
        else:
            return 'neutral'

    def get_category_compatibility_score(self, category1: str, category2: str) -> float:
        """카테고리 조합 적합성 점수"""
        return self.category_compatibility.get((category1, category2), 0.5)

    def calculate_recommendation_score(self, 
                                     base_item: FashionItem, 
                                     candidate_item: FashionItem,
                                     season: str = 'spring') -> RecommendationScore:
        """종합 추천 점수 계산"""
        
        # 1. 색상 조화 점수
        color_harmony = self.get_color_harmony_score(base_item.color, candidate_item.color)
        
        # 2. 소재 조합 점수
        material_combination = self.get_material_combination_score(
            base_item.materials, candidate_item.materials
        )
        
        # 3. 핏 조합 점수
        fit_combination = self.get_fit_combination_score(base_item.fit, candidate_item.fit)
        
        # 4. 스타일 조합 점수
        style_combination = self.get_style_combination_score(base_item.style, candidate_item.style)
        
        # 5. 계절별 적합성 점수
        seasonal_suitability = self.get_seasonal_suitability_score(candidate_item, season)
        
        # 6. 카테고리 조합 적합성 점수
        category_compatibility = self.get_category_compatibility_score(
            base_item.category, candidate_item.category
        )
        
        # 가중 평균으로 총점 계산
        weights = {
            'color_harmony': 0.25,
            'material_combination': 0.20,
            'fit_combination': 0.20,
            'style_combination': 0.15,
            'seasonal_suitability': 0.10,
            'category_compatibility': 0.10
        }
        
        total_score = (
            color_harmony * weights['color_harmony'] +
            material_combination * weights['material_combination'] +
            fit_combination * weights['fit_combination'] +
            style_combination * weights['style_combination'] +
            seasonal_suitability * weights['seasonal_suitability'] +
            category_compatibility * weights['category_compatibility']
        )
        
        return RecommendationScore(
            total_score=total_score,
            color_harmony=color_harmony,
            material_combination=material_combination,
            fit_combination=fit_combination,
            style_combination=style_combination,
            seasonal_suitability=seasonal_suitability,
            category_compatibility=category_compatibility
        )

    def recommend_items(self, 
                       base_item: FashionItem, 
                       candidate_items: List[FashionItem],
                       season: str = 'spring',
                       n_results: int = 5) -> List[Tuple[FashionItem, RecommendationScore]]:
        """아이템 추천"""
        
        scored_items = []
        for candidate in candidate_items:
            if candidate.item_id == base_item.item_id:
                continue  # 자기 자신 제외
            
            score = self.calculate_recommendation_score(base_item, candidate, season)
            scored_items.append((candidate, score))
        
        # 점수 순으로 정렬
        scored_items.sort(key=lambda x: x[1].total_score, reverse=True)
        
        return scored_items[:n_results]

    def get_recommendation_explanation(self, base_item: FashionItem, 
                                     candidate_item: FashionItem, 
                                     score: RecommendationScore) -> str:
        """추천 이유 설명 생성"""
        explanations = []
        
        if score.color_harmony > 0.7:
            explanations.append(f"🎨 색상 조화가 좋습니다 ({score.color_harmony:.2f})")
        elif score.color_harmony < 0.4:
            explanations.append(f"🎨 색상 조화를 개선할 수 있습니다 ({score.color_harmony:.2f})")
        
        if score.material_combination > 0.7:
            explanations.append(f"🧵 소재 조합이 적합합니다 ({score.material_combination:.2f})")
        
        if score.fit_combination > 0.7:
            explanations.append(f"📏 핏 조합이 잘 맞습니다 ({score.fit_combination:.2f})")
        
        if score.style_combination > 0.7:
            explanations.append(f"✨ 스타일이 조화롭습니다 ({score.style_combination:.2f})")
        
        if score.seasonal_suitability > 0.7:
            explanations.append(f"🌤️ 계절에 적합합니다 ({score.seasonal_suitability:.2f})")
        
        if not explanations:
            explanations.append("💡 기본적인 조합입니다")
        
        return " | ".join(explanations)

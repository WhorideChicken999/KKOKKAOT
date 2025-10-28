"""
추천 시스템 API
- 유사 아이템 추천
- 코디 추천 (상의-하의 매칭)
- 고급 추천 (색상/스타일 기반)
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from pathlib import Path

router = APIRouter(prefix="/api/recommendations", tags=["추천"])

# 전역 변수 (메인에서 주입)
pipeline = None
advanced_recommender = None


@router.get("/default/{user_id}")
async def get_default_recommendations(user_id: int):
    """사용자 스타일 기반 기본 추천"""
    print(f"\n{'='*60}")
    print(f"🎯 기본 아이템 추천 조회 (user_id: {user_id})")
    print(f"{'='*60}")
    
    if not pipeline:
        return []
    
    try:
        # 일단 빈 배열 반환 (프론트엔드 에러 방지)
        return []
        
    except Exception as e:
        print(f"❌ 추천 조회 오류: {e}")
        return []


@router.get("/advanced/{item_id}")
async def get_advanced_recommendations(
    item_id: int,
    user_id: Optional[int] = None,
    n_results: int = 10
):
    """고급 추천 시스템 (색상/스타일 기반)"""
    print(f"\n{'='*60}")
    print(f"🎯 고급 추천 시스템 (item_id: {item_id}, user_id: {user_id})")
    print(f"{'='*60}")
    
    if not advanced_recommender or not pipeline:
        raise HTTPException(status_code=503, detail="추천 시스템이 초기화되지 않았습니다")
    
    try:
        from _advanced_recommender import FashionItem
        
        # 기준 아이템 정보 조회 (상세)
        with pipeline.db.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    w.item_id, w.gender, w.style, w.is_default,
                    w.has_top, w.has_bottom, w.has_outer, w.has_dress,
                    t.category as top_cat, t.color as top_color, t.fit as top_fit, t.material as top_mat,
                    b.category as bottom_cat, b.color as bottom_color, b.fit as bottom_fit, b.material as bottom_mat,
                    o.category as outer_cat, o.color as outer_color, o.fit as outer_fit, o.material as outer_mat,
                    d.category as dress_cat, d.color as dress_color, d.material as dress_mat
                FROM wardrobe_items w
                LEFT JOIN top_attributes_new t ON w.item_id = t.item_id
                LEFT JOIN bottom_attributes_new b ON w.item_id = b.item_id
                LEFT JOIN outer_attributes_new o ON w.item_id = o.item_id
                LEFT JOIN dress_attributes_new d ON w.item_id = d.item_id
                WHERE w.item_id = %s
            """, (item_id,))
            
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="기준 아이템을 찾을 수 없습니다")
            
            # 카테고리 결정
            if row[6]:  # has_outer
                category, subcategory, color, fit, material = "outer", row[16] or "아우터", row[17] or "검정", row[18] or "normal", row[19] or "cotton"
            elif row[7]:  # has_dress
                category, subcategory, color, fit, material = "dress", row[20] or "원피스", row[21] or "흰색", "normal", row[22] or "cotton"
            elif row[4]:  # has_top
                category, subcategory, color, fit, material = "top", row[8] or "티셔츠", row[9] or "흰색", row[10] or "normal", row[11] or "cotton"
            else:  # has_bottom
                category, subcategory, color, fit, material = "bottom", row[12] or "청바지", row[13] or "파랑", row[14] or "normal", row[15] or "denim"
            
            base_item_obj = FashionItem(
                item_id=row[0], category=category, subcategory=subcategory, color=color, fit=fit,
                materials=[material] if material else [], style=row[2] or "캐주얼", season="spring", is_default=row[3]
            )
            
            # 후보 아이템 조회
            cur.execute("""
                SELECT 
                    w.item_id, w.gender, w.style, w.is_default,
                    w.has_top, w.has_bottom, w.has_outer, w.has_dress,
                    t.category, t.color, t.fit, t.material,
                    b.category, b.color, b.fit, b.material,
                    o.category, o.color, o.fit, o.material,
                    d.category, d.color, d.material
                FROM wardrobe_items w
                LEFT JOIN top_attributes_new t ON w.item_id = t.item_id
                LEFT JOIN bottom_attributes_new b ON w.item_id = b.item_id
                LEFT JOIN outer_attributes_new o ON w.item_id = o.item_id
                LEFT JOIN dress_attributes_new d ON w.item_id = d.item_id
                WHERE (w.user_id = %s OR (w.user_id = 0 AND w.is_default = TRUE)) AND w.item_id != %s
                LIMIT 100
            """, (user_id or 1, item_id))
            
            candidate_rows = cur.fetchall()
        
        # 후보 아이템을 FashionItem 객체로 변환
        candidate_items = []
        for row in candidate_rows:
            if row[6]:  # has_outer
                cat, subcat, col, ft, mat = "outer", row[16] or "아우터", row[17] or "검정", row[18] or "normal", row[19] or "cotton"
            elif row[7]:  # has_dress
                cat, subcat, col, ft, mat = "dress", row[20] or "원피스", row[21] or "흰색", "normal", row[22] or "cotton"
            elif row[4]:  # has_top
                cat, subcat, col, ft, mat = "top", row[8] or "티셔츠", row[9] or "흰색", row[10] or "normal", row[11] or "cotton"
            else:  # has_bottom
                cat, subcat, col, ft, mat = "bottom", row[12] or "청바지", row[13] or "파랑", row[14] or "normal", row[15] or "denim"
            
            candidate_items.append(FashionItem(
                item_id=row[0], category=cat, subcategory=subcat, color=col, fit=ft,
                materials=[mat] if mat else [], style=row[2] or "캐주얼", season="spring", is_default=row[3]
            ))
        
        # 추천 실행
        recommendations_with_scores = advanced_recommender.recommend_items(base_item_obj, candidate_items, "spring", n_results)
        
        # 결과를 프론트엔드 형식으로 변환
        recommendations = []
        for fashion_item, score in recommendations_with_scores:
            img_path = f"/api/processed-images/user_0/full/item_{fashion_item.item_id}_full.jpg" if fashion_item.is_default else f"/api/processed-images/user_{user_id or 1}/full/item_{fashion_item.item_id}_full.jpg"
            
            recommendations.append({
                "item_id": fashion_item.item_id, "id": fashion_item.item_id, "category": fashion_item.category,
                "subcategory": fashion_item.subcategory, "color": fashion_item.color, "style": fashion_item.style,
                "is_default": fashion_item.is_default, "image_path": img_path,
                "score": {"total": score.total_score, "color_harmony": score.color_harmony, "material": score.material_combination,
                          "fit": score.fit_combination, "style": score.style_combination, "seasonal": score.seasonal_suitability},
                "explanation": advanced_recommender.get_recommendation_explanation(base_item_obj, fashion_item, score)
            })
        
        print(f"  ✅ {len(recommendations)}개 추천 완료")
        print(f"{'='*60}\n")
        
        return {"success": True, "recommendations": recommendations}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 고급 추천 오류: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

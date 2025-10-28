-- ============================================
-- 모든 사용자 데이터 완전 삭제 SQL 스크립트
-- ============================================

-- 1. 삭제 전 현재 상태 확인
SELECT 'Users' as table_name, COUNT(*) as count FROM public.users
UNION ALL
SELECT 'Wardrobe Items', COUNT(*) FROM public.wardrobe_items
UNION ALL
SELECT 'Top Attributes', COUNT(*) FROM public.top_attributes_new
UNION ALL
SELECT 'Bottom Attributes', COUNT(*) FROM public.bottom_attributes_new
UNION ALL
SELECT 'Outer Attributes', COUNT(*) FROM public.outer_attributes_new
UNION ALL
SELECT 'Dress Attributes', COUNT(*) FROM public.dress_attributes_new
UNION ALL
SELECT 'User Recommendations', COUNT(*) FROM public.user_recommendations
UNION ALL
SELECT 'Recommendation History', COUNT(*) FROM public.recommendation_history;

-- 2. 모든 속성 데이터 삭제
DELETE FROM public.top_attributes_new;
DELETE FROM public.bottom_attributes_new;
DELETE FROM public.outer_attributes_new;
DELETE FROM public.dress_attributes_new;

-- 3. 모든 옷장 아이템 삭제
DELETE FROM public.wardrobe_items;

-- 4. 모든 추천 데이터 삭제
DELETE FROM public.user_recommendations;
DELETE FROM public.recommendation_history;

-- 5. 모든 사용자 계정 삭제 (user_id 0과 1 제외)
DELETE FROM public.users WHERE user_id > 1;

-- 6. 삭제 후 상태 확인
SELECT 'Users' as table_name, COUNT(*) as count FROM public.users
UNION ALL
SELECT 'Wardrobe Items', COUNT(*) FROM public.wardrobe_items
UNION ALL
SELECT 'Top Attributes', COUNT(*) FROM public.top_attributes_new
UNION ALL
SELECT 'Bottom Attributes', COUNT(*) FROM public.bottom_attributes_new
UNION ALL
SELECT 'Outer Attributes', COUNT(*) FROM public.outer_attributes_new
UNION ALL
SELECT 'Dress Attributes', COUNT(*) FROM public.dress_attributes_new
UNION ALL
SELECT 'User Recommendations', COUNT(*) FROM public.user_recommendations
UNION ALL
SELECT 'Recommendation History', COUNT(*) FROM public.recommendation_history;

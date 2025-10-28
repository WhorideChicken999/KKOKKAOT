-- ============================================
-- 사용자 데이터 완전 삭제 SQL 스크립트
-- ============================================

-- 1. 삭제할 사용자 ID 확인 (user_id 0과 1 제외)
SELECT user_id, username FROM public.users WHERE user_id > 1;

-- 2. 해당 사용자들의 아이템 속성 데이터 삭제
-- top_attributes_new 테이블에서 삭제
DELETE FROM public.top_attributes_new
WHERE item_id IN (SELECT item_id FROM public.wardrobe_items WHERE user_id > 1);

-- bottom_attributes_new 테이블에서 삭제
DELETE FROM public.bottom_attributes_new
WHERE item_id IN (SELECT item_id FROM public.wardrobe_items WHERE user_id > 1);

-- outer_attributes_new 테이블에서 삭제
DELETE FROM public.outer_attributes_new
WHERE item_id IN (SELECT item_id FROM public.wardrobe_items WHERE user_id > 1);

-- dress_attributes_new 테이블에서 삭제
DELETE FROM public.dress_attributes_new
WHERE item_id IN (SELECT item_id FROM public.wardrobe_items WHERE user_id > 1);

-- 3. 해당 사용자들의 wardrobe_items 삭제
DELETE FROM public.wardrobe_items
WHERE user_id > 1;

-- 4. user_recommendations 테이블에서 해당 사용자들의 추천 데이터 삭제
DELETE FROM public.user_recommendations
WHERE user_id > 1;

-- 5. recommendation_history 테이블에서 해당 사용자들의 추천 히스토리 삭제
DELETE FROM public.recommendation_history
WHERE user_id > 1;

-- 6. 마지막으로, user_id 0과 1을 제외한 사용자 목록 삭제
DELETE FROM public.users
WHERE user_id > 1;

-- 7. 삭제 결과 확인
SELECT 'Users' as table_name, COUNT(*) as remaining_count FROM public.users
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

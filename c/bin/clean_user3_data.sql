-- 사용자 3의 불완전한 데이터 정리
-- 이미지 파일이 없는 아이템들을 삭제

-- 1. 사용자 3의 아이템 중 이미지 파일이 없는 것들 확인
SELECT 
    w.item_id,
    w.original_image_path,
    w.upload_date,
    w.has_top,
    w.has_bottom,
    w.has_outer,
    w.has_dress
FROM wardrobe_items w
WHERE w.user_id = 3
ORDER BY w.upload_date DESC;

-- 2. 사용자 3의 모든 데이터 삭제 (이미지 파일이 없으므로)
DELETE FROM top_attributes_new
WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE user_id = 3);

DELETE FROM bottom_attributes_new
WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE user_id = 3);

DELETE FROM outer_attributes_new
WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE user_id = 3);

DELETE FROM dress_attributes_new
WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE user_id = 3);

DELETE FROM wardrobe_items
WHERE user_id = 3;

-- 3. 사용자 3 계정도 삭제 (선택사항)
-- DELETE FROM users WHERE user_id = 3;

-- 4. 정리 결과 확인
SELECT 'Users' as table_name, COUNT(*) as count FROM users WHERE user_id = 3
UNION ALL
SELECT 'Wardrobe Items', COUNT(*) FROM wardrobe_items WHERE user_id = 3
UNION ALL
SELECT 'Top Attributes', COUNT(*) FROM top_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE user_id = 3)
UNION ALL
SELECT 'Bottom Attributes', COUNT(*) FROM bottom_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE user_id = 3)
UNION ALL
SELECT 'Outer Attributes', COUNT(*) FROM outer_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE user_id = 3)
UNION ALL
SELECT 'Dress Attributes', COUNT(*) FROM dress_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE user_id = 3);

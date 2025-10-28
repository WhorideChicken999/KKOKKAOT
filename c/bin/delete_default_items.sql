-- 모든 기본 아이템 완전 삭제
BEGIN;

-- 1. 기본 아이템 속성 테이블들 먼저 삭제
DELETE FROM top_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE is_default = TRUE);
DELETE FROM bottom_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE is_default = TRUE);
DELETE FROM outer_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE is_default = TRUE);
DELETE FROM dress_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE is_default = TRUE);

-- 2. 기본 아이템 메인 테이블 삭제
DELETE FROM wardrobe_items WHERE is_default = TRUE;

-- 3. 삭제된 아이템 수 확인
SELECT COUNT(*) as deleted_count FROM wardrobe_items WHERE is_default = TRUE;

COMMIT;

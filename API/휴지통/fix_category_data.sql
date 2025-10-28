-- 잘못 분류된 카테고리 데이터 수정
-- 아우터로 잘못 분류된 하의 아이템들을 수정

-- 1. 현재 데이터 상태 확인
SELECT 
    w.item_id,
    w.has_top,
    w.has_bottom,
    w.has_outer,
    w.has_dress,
    t.category as top_category,
    b.category as bottom_category
FROM wardrobe_items w
LEFT JOIN top_attributes t ON w.item_id = t.item_id
LEFT JOIN bottom_attributes b ON w.item_id = b.item_id
WHERE w.has_outer = TRUE OR w.has_bottom = TRUE
ORDER BY w.item_id;

-- 2. 아우터로 잘못 분류된 하의 아이템 수정
-- (실제로는 하의인데 아우터로 분류된 경우)
-- YOLO 클래스 매핑 오류로 인해 하의가 아우터로 분류된 경우
UPDATE wardrobe_items 
SET 
    has_outer = FALSE,
    has_bottom = TRUE
WHERE 
    has_outer = TRUE 
    AND has_bottom = FALSE
    AND has_top = FALSE
    AND has_dress = FALSE;

-- 3. 상의와 하의가 모두 있는 경우 처리
-- (실제로는 상의+하의 세트인데 아우터로만 분류된 경우)
UPDATE wardrobe_items 
SET 
    has_outer = FALSE,
    has_top = TRUE,
    has_bottom = TRUE
WHERE 
    has_outer = TRUE 
    AND has_bottom = FALSE
    AND has_top = FALSE
    AND has_dress = FALSE
    AND item_id IN (
        SELECT item_id FROM top_attributes WHERE item_id IS NOT NULL
    );

-- 4. 수정 후 데이터 확인
SELECT 
    w.item_id,
    w.has_top,
    w.has_bottom,
    w.has_outer,
    w.has_dress,
    t.category as top_category,
    b.category as bottom_category
FROM wardrobe_items w
LEFT JOIN top_attributes t ON w.item_id = t.item_id
LEFT JOIN bottom_attributes b ON w.item_id = b.item_id
WHERE w.has_outer = TRUE OR w.has_bottom = TRUE
ORDER BY w.item_id;

-- 5. 카테고리별 아이템 개수 확인
SELECT 
    '전체' as category,
    COUNT(*) as count
FROM wardrobe_items
UNION ALL
SELECT 
    '상의' as category,
    COUNT(*) as count
FROM wardrobe_items
WHERE has_top = TRUE
UNION ALL
SELECT 
    '하의' as category,
    COUNT(*) as count
FROM wardrobe_items
WHERE has_bottom = TRUE
UNION ALL
SELECT 
    '아우터' as category,
    COUNT(*) as count
FROM wardrobe_items
WHERE has_outer = TRUE
UNION ALL
SELECT 
    '드레스' as category,
    COUNT(*) as count
FROM wardrobe_items
WHERE has_dress = TRUE;

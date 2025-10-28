-- 4개 카테고리 지원을 위한 데이터베이스 업데이트
-- has_outer, has_dress 컬럼 추가

-- wardrobe_items 테이블에 컬럼 추가
ALTER TABLE wardrobe_items 
ADD COLUMN IF NOT EXISTS has_outer BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS has_dress BOOLEAN DEFAULT FALSE;

-- 기존 데이터 업데이트 (has_top이 true인 경우 has_outer로 분류)
-- 이 부분은 실제 데이터에 따라 조정이 필요할 수 있습니다
UPDATE wardrobe_items 
SET has_outer = TRUE 
WHERE has_top = TRUE AND has_bottom = FALSE;

-- 인덱스 추가
CREATE INDEX IF NOT EXISTS idx_wardrobe_outer ON wardrobe_items(has_outer);
CREATE INDEX IF NOT EXISTS idx_wardrobe_dress ON wardrobe_items(has_dress);

-- 뷰 업데이트
DROP VIEW IF EXISTS wardrobe_full_view;
CREATE VIEW wardrobe_full_view AS
SELECT 
    w.item_id,
    w.user_id,
    u.username,
    w.original_image_path,
    w.upload_date,
    w.has_top,
    w.has_bottom,
    w.has_outer,
    w.has_dress,
    w.gender,
    w.style,
    w.chroma_embedding_id,
    
    -- 상의 정보
    t.category AS top_category,
    t.color AS top_color,
    t.fit AS top_fit,
    t.materials AS top_materials,
    t.prints AS top_prints,
    
    -- 하의 정보
    b.category AS bottom_category,
    b.color AS bottom_color,
    b.fit AS bottom_fit,
    b.materials AS bottom_materials,
    b.prints AS bottom_prints
    
FROM wardrobe_items w
JOIN users u ON w.user_id = u.user_id
LEFT JOIN top_attributes t ON w.item_id = t.item_id
LEFT JOIN bottom_attributes b ON w.item_id = b.item_id;

-- 코멘트 추가
COMMENT ON COLUMN wardrobe_items.has_outer IS '아우터 아이템 여부 (YOLO Detection 결과)';
COMMENT ON COLUMN wardrobe_items.has_dress IS '드레스 아이템 여부 (YOLO Detection 결과)';

-- 업데이트 완료 메시지
SELECT 'Database updated successfully for 4-category support (top, bottom, outer, dress)' as message;

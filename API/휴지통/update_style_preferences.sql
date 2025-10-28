-- 사용자 스타일 선호도 컬럼 추가
-- 24개 스타일: 기타, 레트로, 로맨틱, 리조트, 매니시, 모던, 밀리터리, 섹시, 소피스트케이티드, 
-- 스트리트, 스포티, 아방가르드, 오리엔탈, 웨스턴, 젠더리스, 컨트리, 클래식, 키치, 
-- 톰보이, 펑크, 페미닌, 프레피, 히피, 히프합

-- 1. users 테이블에 style_preferences 컬럼 추가
ALTER TABLE users ADD COLUMN IF NOT EXISTS style_preferences JSONB DEFAULT '[]'::jsonb;

-- 2. 인덱스 추가 (스타일 선호도 검색 최적화)
CREATE INDEX IF NOT EXISTS idx_users_style_preferences ON users USING GIN (style_preferences);

-- 3. 기존 사용자들의 기본 스타일 선호도 설정 (빈 배열)
UPDATE users SET style_preferences = '[]'::jsonb WHERE style_preferences IS NULL;

-- 4. wardrobe_items 테이블에 스타일 정보 추가 (옷의 스타일 분류용)
ALTER TABLE wardrobe_items ADD COLUMN IF NOT EXISTS detected_style VARCHAR(50);
ALTER TABLE wardrobe_items ADD COLUMN IF NOT EXISTS style_confidence FLOAT DEFAULT 0.0;

-- 5. 스타일 관련 인덱스 추가
CREATE INDEX IF NOT EXISTS idx_wardrobe_detected_style ON wardrobe_items (detected_style);
CREATE INDEX IF NOT EXISTS idx_wardrobe_style_confidence ON wardrobe_items (style_confidence);

-- 6. 스타일 선호도 검색을 위한 함수 생성
CREATE OR REPLACE FUNCTION get_users_by_style_preference(preferred_style TEXT)
RETURNS TABLE(user_id INTEGER, username VARCHAR, style_preferences JSONB) AS $$
BEGIN
    RETURN QUERY
    SELECT u.user_id, u.username, u.style_preferences
    FROM users u
    WHERE u.style_preferences::text ILIKE '%' || preferred_style || '%';
END;
$$ LANGUAGE plpgsql;

-- 7. 스타일 매칭 점수 계산 함수
CREATE OR REPLACE FUNCTION calculate_style_match_score(user_preferences JSONB, item_style TEXT)
RETURNS FLOAT AS $$
DECLARE
    style_array TEXT[];
    match_count INTEGER := 0;
    total_styles INTEGER;
BEGIN
    -- JSONB 배열을 텍스트 배열로 변환
    SELECT ARRAY(SELECT jsonb_array_elements_text(user_preferences)) INTO style_array;
    
    -- 총 스타일 개수
    total_styles := array_length(style_array, 1);
    
    -- 매칭되는 스타일 개수 계산
    IF total_styles > 0 THEN
        SELECT COUNT(*) INTO match_count
        FROM unnest(style_array) AS style
        WHERE style = item_style;
        
        -- 매칭 점수 반환 (0.0 ~ 1.0)
        RETURN CASE 
            WHEN match_count > 0 THEN 1.0
            ELSE 0.0
        END;
    END IF;
    
    RETURN 0.0;
END;
$$ LANGUAGE plpgsql;

-- 8. 스타일 기반 추천을 위한 뷰 생성
CREATE OR REPLACE VIEW style_recommendations_view AS
SELECT 
    w.item_id,
    w.user_id,
    w.detected_style,
    w.style_confidence,
    u.username,
    u.style_preferences,
    calculate_style_match_score(u.style_preferences, w.detected_style) as style_match_score
FROM wardrobe_items w
JOIN users u ON w.user_id = u.user_id
WHERE w.detected_style IS NOT NULL;

-- 9. 샘플 데이터 삽입 (테스트용)
-- INSERT INTO users (username, email, password_hash, style_preferences) 
-- VALUES ('test_user', 'test@example.com', 'hashed_password', '["스트리트", "스포티", "모던"]'::jsonb);

COMMENT ON COLUMN users.style_preferences IS 'User preferred fashion styles (JSON array)';
COMMENT ON COLUMN wardrobe_items.detected_style IS 'AI detected clothing style';
COMMENT ON COLUMN wardrobe_items.style_confidence IS 'Style detection confidence (0.0 ~ 1.0)';

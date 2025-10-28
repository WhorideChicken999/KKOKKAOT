-- ========================================
-- PostgreSQL 데이터베이스 업데이트 스크립트
-- pgAdmin에서 직접 실행하세요
-- ========================================

-- 1. wardrobe_items 테이블에 새로운 컬럼 추가
ALTER TABLE wardrobe_items 
ADD COLUMN IF NOT EXISTS style VARCHAR(100),
ADD COLUMN IF NOT EXISTS style_confidence FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS chroma_id VARCHAR(100),
ADD COLUMN IF NOT EXISTS embedding_vector TEXT,
ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;

-- 2. 상의 속성 테이블 (새로운 구조)
CREATE TABLE IF NOT EXISTS top_attributes_new (
    id SERIAL PRIMARY KEY,
    item_id INTEGER REFERENCES wardrobe_items(item_id) ON DELETE CASCADE,
    category VARCHAR(100),
    color VARCHAR(100),
    fit VARCHAR(100),
    material VARCHAR(100),
    print_pattern VARCHAR(100),
    style VARCHAR(100),
    sleeve_length VARCHAR(100),
    category_confidence FLOAT DEFAULT 0.0,
    color_confidence FLOAT DEFAULT 0.0,
    fit_confidence FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. 하의 속성 테이블 (새로운 구조)
CREATE TABLE IF NOT EXISTS bottom_attributes_new (
    id SERIAL PRIMARY KEY,
    item_id INTEGER REFERENCES wardrobe_items(item_id) ON DELETE CASCADE,
    category VARCHAR(100),
    color VARCHAR(100),
    fit VARCHAR(100),
    material VARCHAR(100),
    print_pattern VARCHAR(100),
    style VARCHAR(100),
    length VARCHAR(100),
    category_confidence FLOAT DEFAULT 0.0,
    color_confidence FLOAT DEFAULT 0.0,
    fit_confidence FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. 아우터 속성 테이블 (새로운 구조)
CREATE TABLE IF NOT EXISTS outer_attributes_new (
    id SERIAL PRIMARY KEY,
    item_id INTEGER REFERENCES wardrobe_items(item_id) ON DELETE CASCADE,
    category VARCHAR(100),
    color VARCHAR(100),
    fit VARCHAR(100),
    material VARCHAR(100),
    print_pattern VARCHAR(100),
    style VARCHAR(100),
    sleeve_length VARCHAR(100),
    category_confidence FLOAT DEFAULT 0.0,
    color_confidence FLOAT DEFAULT 0.0,
    fit_confidence FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. 원피스 속성 테이블 (새로운 구조)
CREATE TABLE IF NOT EXISTS dress_attributes_new (
    id SERIAL PRIMARY KEY,
    item_id INTEGER REFERENCES wardrobe_items(item_id) ON DELETE CASCADE,
    category VARCHAR(100),
    color VARCHAR(100),
    material VARCHAR(100),
    print_pattern VARCHAR(100),
    style VARCHAR(100),
    category_confidence FLOAT DEFAULT 0.0,
    color_confidence FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. 날씨 정보 테이블
CREATE TABLE IF NOT EXISTS weather_info (
    id SERIAL PRIMARY KEY,
    location VARCHAR(100) NOT NULL,
    temperature FLOAT,
    humidity FLOAT,
    weather_condition VARCHAR(100),
    wind_speed FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. 추천 기록 테이블
CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    recommendation_type VARCHAR(50) NOT NULL,
    recommended_items TEXT,
    query_params TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 8. 기존 데이터 마이그레이션 (상의)
INSERT INTO top_attributes_new (item_id, category, color, fit, material, print_pattern, style, sleeve_length, category_confidence, color_confidence, fit_confidence)
SELECT 
    item_id,
    category,
    color,
    fit,
    COALESCE(materials->>0, 'Unknown') as material,
    COALESCE(prints->>0, 'Unknown') as print_pattern,
    'Unknown' as style,
    sleeve_length,
    category_confidence,
    color_confidence,
    fit_confidence
FROM top_attributes
WHERE NOT EXISTS (SELECT 1 FROM top_attributes_new WHERE top_attributes_new.item_id = top_attributes.item_id);

-- 9. 기존 데이터 마이그레이션 (하의)
INSERT INTO bottom_attributes_new (item_id, category, color, fit, material, print_pattern, style, length, category_confidence, color_confidence, fit_confidence)
SELECT 
    item_id,
    category,
    color,
    fit,
    COALESCE(materials->>0, 'Unknown') as material,
    COALESCE(prints->>0, 'Unknown') as print_pattern,
    'Unknown' as style,
    length,
    category_confidence,
    color_confidence,
    fit_confidence
FROM bottom_attributes
WHERE NOT EXISTS (SELECT 1 FROM bottom_attributes_new WHERE bottom_attributes_new.item_id = bottom_attributes.item_id);

-- 10. 샘플 날씨 데이터 추가
INSERT INTO weather_info (location, temperature, humidity, weather_condition, wind_speed)
VALUES ('Seoul', 22.5, 65.0, '맑음', 3.2)
ON CONFLICT DO NOTHING;

-- 11. 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_wardrobe_items_style ON wardrobe_items(style);
CREATE INDEX IF NOT EXISTS idx_wardrobe_items_chroma_id ON wardrobe_items(chroma_id);
CREATE INDEX IF NOT EXISTS idx_wardrobe_items_is_active ON wardrobe_items(is_active);

CREATE INDEX IF NOT EXISTS idx_top_attributes_new_item_id ON top_attributes_new(item_id);
CREATE INDEX IF NOT EXISTS idx_top_attributes_new_category ON top_attributes_new(category);
CREATE INDEX IF NOT EXISTS idx_top_attributes_new_color ON top_attributes_new(color);

CREATE INDEX IF NOT EXISTS idx_bottom_attributes_new_item_id ON bottom_attributes_new(item_id);
CREATE INDEX IF NOT EXISTS idx_bottom_attributes_new_category ON bottom_attributes_new(category);
CREATE INDEX IF NOT EXISTS idx_bottom_attributes_new_color ON bottom_attributes_new(color);

CREATE INDEX IF NOT EXISTS idx_outer_attributes_new_item_id ON outer_attributes_new(item_id);
CREATE INDEX IF NOT EXISTS idx_outer_attributes_new_category ON outer_attributes_new(category);
CREATE INDEX IF NOT EXISTS idx_outer_attributes_new_color ON outer_attributes_new(color);

CREATE INDEX IF NOT EXISTS idx_dress_attributes_new_item_id ON dress_attributes_new(item_id);
CREATE INDEX IF NOT EXISTS idx_dress_attributes_new_category ON dress_attributes_new(category);
CREATE INDEX IF NOT EXISTS idx_dress_attributes_new_color ON dress_attributes_new(color);

CREATE INDEX IF NOT EXISTS idx_recommendations_user_id ON recommendations(user_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_type ON recommendations(recommendation_type);

-- 12. 업데이트 완료 메시지
DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE '✅ PostgreSQL 데이터베이스 업데이트 완료!';
    RAISE NOTICE '========================================';
    RAISE NOTICE '새로운 테이블이 생성되었습니다:';
    RAISE NOTICE '  - top_attributes_new (상의 속성)';
    RAISE NOTICE '  - bottom_attributes_new (하의 속성)';
    RAISE NOTICE '  - outer_attributes_new (아우터 속성)';
    RAISE NOTICE '  - dress_attributes_new (원피스 속성)';
    RAISE NOTICE '  - weather_info (날씨 정보)';
    RAISE NOTICE '  - recommendations (추천 기록)';
    RAISE NOTICE '';
    RAISE NOTICE 'wardrobe_items 테이블에 새로운 컬럼이 추가되었습니다:';
    RAISE NOTICE '  - style, style_confidence, chroma_id, embedding_vector, is_active';
    RAISE NOTICE '';
    RAISE NOTICE '기존 데이터가 새로운 구조로 마이그레이션되었습니다.';
    RAISE NOTICE '이제 백엔드 서버를 실행할 수 있습니다!';
    RAISE NOTICE '========================================';
END $$;

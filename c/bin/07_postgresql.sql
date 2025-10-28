-- 사용자 테이블
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 가상옷장 아이템 테이블
CREATE TABLE wardrobe_items (
    item_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    
    -- 이미지 정보
    original_image_path TEXT NOT NULL,
    processed_image_path TEXT,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- YOLO 분리 정보
    has_top BOOLEAN DEFAULT FALSE,
    has_bottom BOOLEAN DEFAULT FALSE,
    top_image_path TEXT,
    bottom_image_path TEXT,
    waist_y INTEGER,
    
    -- ChromaDB 정보
    chroma_embedding_id VARCHAR(100),
    
    -- 메타데이터
    gender VARCHAR(20),
    style VARCHAR(50),
    
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- 상의 속성 테이블
CREATE TABLE top_attributes (
    top_id SERIAL PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES wardrobe_items(item_id) ON DELETE CASCADE,
    
    category VARCHAR(50),
    color VARCHAR(50),
    sub_color VARCHAR(50),
    fit VARCHAR(50),
    length VARCHAR(50),
    sleeve_length VARCHAR(50),
    neckline VARCHAR(50),
    collar VARCHAR(50),
    
    -- 다중 속성 (JSON 배열)
    materials JSONB,
    prints JSONB,
    details JSONB,
    
    -- 예측 확신도
    category_confidence FLOAT,
    color_confidence FLOAT,
    fit_confidence FLOAT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 하의 속성 테이블
CREATE TABLE bottom_attributes (
    bottom_id SERIAL PRIMARY KEY,
    item_id INTEGER NOT NULL REFERENCES wardrobe_items(item_id) ON DELETE CASCADE,
    
    category VARCHAR(50),
    color VARCHAR(50),
    sub_color VARCHAR(50),
    fit VARCHAR(50),
    length VARCHAR(50),
    
    -- 다중 속성 (JSON 배열)
    materials JSONB,
    prints JSONB,
    details JSONB,
    
    -- 예측 확신도
    category_confidence FLOAT,
    color_confidence FLOAT,
    fit_confidence FLOAT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 추천 기록 테이블
CREATE TABLE recommendation_history (
    rec_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    query_item_id INTEGER REFERENCES wardrobe_items(item_id) ON DELETE SET NULL,
    
    -- 추천 결과 (JSON 배열)
    recommended_items JSONB NOT NULL,
    
    -- 필터 조건
    filter_gender VARCHAR(20),
    filter_style VARCHAR(50),
    filter_category VARCHAR(50),
    
    search_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 인덱스 생성
CREATE INDEX idx_wardrobe_user ON wardrobe_items(user_id);
CREATE INDEX idx_wardrobe_gender ON wardrobe_items(gender);
CREATE INDEX idx_wardrobe_style ON wardrobe_items(style);
CREATE INDEX idx_wardrobe_upload ON wardrobe_items(upload_date DESC);

CREATE INDEX idx_top_category ON top_attributes(category);
CREATE INDEX idx_top_color ON top_attributes(color);
CREATE INDEX idx_top_item ON top_attributes(item_id);

CREATE INDEX idx_bottom_category ON bottom_attributes(category);
CREATE INDEX idx_bottom_color ON bottom_attributes(color);
CREATE INDEX idx_bottom_item ON bottom_attributes(item_id);

CREATE INDEX idx_rec_user ON recommendation_history(user_id);
CREATE INDEX idx_rec_date ON recommendation_history(search_date DESC);

-- 업데이트 트리거 (updated_at 자동 갱신)
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 뷰: 전체 아이템 정보 (JOIN)
CREATE VIEW wardrobe_full_view AS
SELECT 
    w.item_id,
    w.user_id,
    u.username,
    w.original_image_path,
    w.upload_date,
    w.has_top,
    w.has_bottom,
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

-- 샘플 사용자 생성 (테스트용)
INSERT INTO users (username, email) VALUES 
    ('test_user', 'test@example.com'),
    ('demo_user', 'demo@example.com');

-- 코멘트 추가
COMMENT ON TABLE wardrobe_items IS '사용자별 가상옷장 아이템 메인 테이블';
COMMENT ON TABLE top_attributes IS '상의 상세 속성 정보';
COMMENT ON TABLE bottom_attributes IS '하의 상세 속성 정보';
COMMENT ON TABLE recommendation_history IS '추천 검색 이력';
COMMENT ON COLUMN wardrobe_items.chroma_embedding_id IS 'ChromaDB에 저장된 임베딩 ID';
COMMENT ON COLUMN top_attributes.materials IS 'JSON 배열 형식: ["우븐", "레이스"]';
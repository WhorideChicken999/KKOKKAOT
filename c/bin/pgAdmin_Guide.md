# pgAdmin에서 데이터베이스 업데이트하기

## 🚀 단계별 실행 방법

### 1단계: pgAdmin 열기
- pgAdmin 4를 실행합니다
- 서버에 연결합니다 (보통 localhost:5432)

### 2단계: 데이터베이스 선택
- 왼쪽 트리에서 `kkokkaot_closet` 데이터베이스를 찾습니다
- 데이터베이스를 클릭하여 선택합니다

### 3단계: SQL 쿼리 도구 열기
- 상단 메뉴에서 **Tools** → **Query Tool** 클릭
- 또는 데이터베이스를 우클릭 → **Query Tool** 선택

### 4단계: SQL 스크립트 실행
1. `database_update.sql` 파일을 열어서 전체 내용을 복사합니다
2. pgAdmin의 Query Tool에 붙여넣습니다
3. **F5** 키를 누르거나 **Execute** 버튼을 클릭합니다

### 5단계: 결과 확인
- 하단의 **Messages** 탭에서 실행 결과를 확인합니다
- 성공 메시지가 나타나면 업데이트가 완료된 것입니다

## 📋 실행할 SQL 스크립트 내용

### 추가되는 테이블
- `top_attributes_new` - 상의 속성 (7개 속성)
- `bottom_attributes_new` - 하의 속성 (7개 속성)
- `outer_attributes_new` - 아우터 속성 (7개 속성)
- `dress_attributes_new` - 원피스 속성 (5개 속성)
- `weather_info` - 날씨 정보
- `recommendations` - 추천 기록

### 업데이트되는 테이블
- `wardrobe_items`에 새로운 컬럼 추가:
  - `style` - 스타일
  - `style_confidence` - 스타일 신뢰도
  - `chroma_id` - ChromaDB ID
  - `embedding_vector` - 임베딩 벡터
  - `is_active` - 활성 상태

## ✅ 실행 후 확인

### 테이블 목록 확인
```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name LIKE '%_new'
ORDER BY table_name;
```

### 새로운 컬럼 확인
```sql
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'wardrobe_items' 
AND column_name IN ('style', 'style_confidence', 'chroma_id', 'embedding_vector', 'is_active')
ORDER BY column_name;
```

## ⚠️ 주의사항

1. **백업 권장**: 실행 전에 데이터베이스를 백업하는 것을 권장합니다
2. **기존 데이터 보존**: 기존 데이터는 그대로 유지됩니다
3. **새로운 테이블**: `*_new` 접미사로 새로운 테이블이 생성됩니다
4. **마이그레이션**: 기존 데이터가 새로운 구조로 자동 변환됩니다

## 🔧 문제 해결

### 권한 오류
- postgres 사용자로 로그인했는지 확인
- 데이터베이스 소유자 권한이 있는지 확인

### 연결 오류
- PostgreSQL 서비스가 실행 중인지 확인
- 포트 5432가 사용 가능한지 확인

### 실행 오류
- SQL 스크립트를 한 번에 실행하지 말고 단계별로 실행
- 오류 메시지를 확인하고 해당 부분만 수정하여 재실행

## 🎯 완료 후

업데이트가 완료되면:
1. **백엔드 서버 실행**: `python backend_server.py`
2. **API 테스트**: http://127.0.0.1:4000/docs
3. **데이터 확인**: 새로운 테이블에 데이터가 정상적으로 들어갔는지 확인

# 🔧 백엔드 리팩토링 가이드

## 📊 변경 사항

### Before (기존)
```
API/
└── backend_server.py (3,647줄) ❌ 너무 복잡!
```

### After (개선)
```
API/
├── backend_server_new.py      (180줄) ✅ 간단!
├── config/
│   └── settings.py            (설정 통합)
├── models/
│   └── schemas.py             (데이터 구조)
└── routers/                   (API 분리)
    ├── auth.py                (로그인/회원가입)
    ├── wardrobe.py            (옷장 관리)
    ├── recommendations.py     (추천 시스템)
    ├── chat.py                (LLM 챗봇)
    ├── images.py              (이미지 제공)
    ├── admin.py               (관리자)
    └── weather.py             (날씨)
```

## 📝 다음 단계 (중요!)

### 1단계: auth.py는 완성됨 ✅
- `routers/auth.py`는 이미 완성되어 있습니다.
- 회원가입, 로그인 API가 동작합니다.

### 2단계: 나머지 라우터 파일 완성 필요 ⚠️

각 라우터 파일에는 **TODO 주석**이 있습니다.
원본 `backend_server.py`에서 해당 코드를 **복사**해서 붙여넣어야 합니다.

#### 복사 방법:
1. `backend_server.py` 열기
2. TODO 주석에 표시된 줄 번호 찾기
3. 해당 함수 복사
4. 라우터 파일에 붙여넣기
5. `@app.post` → `@router.post` 변경
6. `@app.get` → `@router.get` 변경

#### 예시: wardrobe.py 완성하기
```python
# backend_server.py 줄 292-414 복사
@app.post("/api/upload-wardrobe")
async def upload_wardrobe_item(...):
    # 코드...

# ↓ 변경 후 wardrobe.py에 붙여넣기 ↓

@router.post("/upload")  # /api/wardrobe 접두사가 자동으로 붙음
async def upload_wardrobe_item(...):
    # 코드...
```

### 3단계: 테스트

#### 옵션 A: 기존 서버로 계속 사용
```bash
python backend_server.py  # 기존 (3647줄)
```

#### 옵션 B: 새 서버로 전환 (auth만 작동)
```bash
python backend_server_new.py  # 새 버전 (180줄)
```

⚠️ **주의**: 새 서버는 auth.py만 완성되어 있으므로 로그인/회원가입만 작동합니다!

## 📂 파일별 작업 현황

| 파일 | 상태 | 라인수 | 작업 필요 |
|------|------|--------|----------|
| `backend_server_new.py` | ✅ 완료 | 180줄 | 없음 |
| `config/settings.py` | ✅ 완료 | 58줄 | 없음 |
| `models/schemas.py` | ✅ 완료 | 68줄 | 없음 |
| `routers/auth.py` | ✅ 완료 | 200줄 | 없음 |
| `routers/wardrobe.py` | ⚠️ TODO | 94줄 | 6개 API 이전 |
| `routers/recommendations.py` | ⚠️ TODO | 80줄 | 9개 API 이전 |
| `routers/chat.py` | ⚠️ TODO | 60줄 | 3개 API 이전 |
| `routers/images.py` | ⚠️ TODO | 50줄 | 6개 API 이전 |
| `routers/admin.py` | ⚠️ TODO | 47줄 | 3개 API 이전 |
| `routers/weather.py` | ⚠️ TODO | 27줄 | 1개 API 이전 |

## 🎯 권장 작업 순서

1. **wardrobe.py** - 가장 많이 사용되는 옷장 관리 API
2. **recommendations.py** - 핵심 추천 기능
3. **chat.py** - LLM 챗봇
4. **images.py** - 이미지 제공 (간단)
5. **weather.py** - 날씨 (간단)
6. **admin.py** - 관리자 (덜 중요)

## 💡 팁

### 코드 복사 시 주의사항
1. **전역 변수 사용**: `pipeline`, `llm_recommender` 등은 라우터 파일 상단에 선언됨
2. **경로 변경**: 라우터의 `prefix`가 자동으로 붙으므로 경로에서 제거
   - `@app.post("/api/wardrobe/upload")` 
   - → `@router.post("/upload")`
3. **import 추가**: 필요한 모듈은 라우터 파일 상단에 import

### 천천히 진행하세요
- 한 번에 모든 파일을 완성할 필요 없습니다
- 하나씩 완성하면서 테스트하세요
- 기존 `backend_server.py`는 백업으로 남겨두세요

## 🔄 전환 계획

### 단계별 전환
1. **현재**: `backend_server.py` 사용 (안정적)
2. **작업 중**: 라우터 파일들 하나씩 완성
3. **테스트**: 각 라우터 완성 후 `backend_server_new.py`로 테스트
4. **전환**: 모든 라우터 완성 후 `backend_server_new.py`를 메인으로 사용
5. **정리**: 기존 `backend_server.py`를 `backend_server_old.py`로 백업

## 📞 문제 발생 시

- 기존 서버로 되돌리기: `python backend_server.py`
- 에러 확인: 터미널 로그 확인
- import 오류: 경로 및 모듈명 확인

---

## ✨ 리팩토링 완료 후 장점

✅ **유지보수 쉬움** - 기능별로 파일 분리  
✅ **디버깅 쉬움** - 문제 발생 위치 빠르게 파악  
✅ **코드 가독성** - 각 파일 100-200줄로 관리  
✅ **협업 용이** - 여러 사람이 동시 작업 가능  
✅ **확장 용이** - 새 기능 추가가 간단  

**작업 화이팅! 🚀**


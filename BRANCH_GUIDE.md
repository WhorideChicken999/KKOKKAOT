# 🌿 꼬까옷 Git 브랜치 사용 가이드

## 📋 목차
1. [브랜치 구조](#브랜치-구조)
2. [브랜치별 사용법](#브랜치별-사용법)
3. [커밋 메시지 규칙](#커밋-메시지-규칙)
4. [작업 플로우](#작업-플로우)
5. [자주 사용하는 명령어](#자주-사용하는-명령어)

---

## 🌳 브랜치 구조

```
main (프로덕션)
├── develop (개발 통합)
│   ├── feature/llm (LLM 개발)
│   ├── feature/sam2 (SAM2 개발)
│   ├── feature/dnn (DNN 개발)
│   ├── feature/yolo (YOLO 개발)
│   └── bugfix/* (버그 수정)
└── release/* (릴리즈)
```

---

## 📚 브랜치별 사용법

### 1️⃣ `main` 브랜치
**용도:** 실제 배포되는 안정적인 코드만 포함

**규칙:**
- ❌ 직접 push 금지
- ✅ `release` 브랜치에서만 merge
- ✅ 태그로 버전 관리 (v1.0.0, v1.1.0 등)

**사용법:**
```bash
# main은 건드리지 않고 조회만
git checkout main
git pull origin main
```

---

### 2️⃣ `develop` 브랜치
**용도:** 개발 중인 기능들을 통합하는 브랜치

**규칙:**
- ❌ 직접 작업 금지
- ✅ feature 브랜치에서만 merge
- ✅ 항상 최신 상태 유지

**사용법:**
```bash
# develop 최신 상태로 업데이트
git checkout develop
git pull origin develop

# feature 브랜치 merge (GitHub PR 권장)
git merge feature/llm
git push origin develop
```

---

### 3️⃣ `feature/llm` 브랜치
**용도:** LLM 관련 기능 개발 (GPT, Claude API 등)

**작업 예시:**
- 프롬프트 엔지니어링
- LLM 응답 처리
- 스타일 추천 텍스트 생성

**사용법:**
```bash
# 1. 브랜치 이동 및 최신화
git checkout feature/llm
git pull origin feature/llm

# 2. 작업 후 커밋
git add .
git commit -m "feat(llm): GPT-4 프롬프트 개선"

# 3. GitHub에 push
git push origin feature/llm

# 4. GitHub에서 PR 생성 (feature/llm → develop)
```

---

### 4️⃣ `feature/sam2` 브랜치
**용도:** SAM2 (Segment Anything Model 2) 관련 개발

**작업 예시:**
- 이미지 세그멘테이션
- 배경 제거
- 옷 영역 자동 추출

**사용법:**
```bash
git checkout feature/sam2
git pull origin feature/sam2

# 작업...
git add .
git commit -m "feat(sam2): 배경 제거 정확도 향상"
git push origin feature/sam2
```

---

### 5️⃣ `feature/dnn` 브랜치
**용도:** DNN (Deep Neural Network) 모델 개발

**작업 예시:**
- 스타일 분류 모델
- 색상 매칭 알고리즘
- 패션 속성 예측

**사용법:**
```bash
git checkout feature/dnn
git pull origin feature/dnn

# 작업...
git add .
git commit -m "feat(dnn): 스타일 분류 모델 정확도 개선"
git push origin feature/dnn
```

---

### 6️⃣ `feature/yolo` 브랜치
**용도:** YOLO 객체 인식 개발

**작업 예시:**
- 옷 탐지 모델 학습
- YOLO Pose 적용
- 바운딩 박스 최적화

**사용법:**
```bash
git checkout feature/yolo
git pull origin feature/yolo

# 작업...
git add .
git commit -m "feat(yolo): 옷 탐지 정확도 95%로 향상"
git push origin feature/yolo
```

---

### 7️⃣ `bugfix/*` 브랜치
**용도:** 버그 수정 전용

**네이밍:**
- `bugfix/issue-123` (이슈 번호)
- `bugfix/login-error` (버그 설명)
- `bugfix/image-upload-fail`

**사용법:**
```bash
# develop에서 버그픽스 브랜치 생성
git checkout develop
git checkout -b bugfix/login-error

# 버그 수정 후
git add .
git commit -m "fix: 로그인 시 세션 만료 오류 수정"
git push origin bugfix/login-error

# PR 생성 (bugfix/login-error → develop)
```

---

### 8️⃣ `release/*` 브랜치
**용도:** 릴리즈 준비 및 최종 테스트

**네이밍:**
- `release/v1.0.0`
- `release/v1.1.0`

**사용법:**
```bash
# develop에서 릴리즈 브랜치 생성
git checkout develop
git checkout -b release/v1.0.0

# 버전 번호 업데이트, 최종 테스트
git add .
git commit -m "chore: v1.0.0 릴리즈 준비"
git push origin release/v1.0.0

# main에 merge 후 태그 생성
git checkout main
git merge release/v1.0.0
git tag -a v1.0.0 -m "Version 1.0.0 Release"
git push origin main --tags

# develop에도 merge (버전 동기화)
git checkout develop
git merge release/v1.0.0
git push origin develop
```

---

## 📝 커밋 메시지 규칙

### 기본 형식
```
<type>(<scope>): <subject>

<body> (선택)

<footer> (선택)
```

### Type (필수)
| Type | 설명 | 예시 |
|------|------|------|
| `feat` | 새로운 기능 추가 | `feat(llm): GPT-4 통합` |
| `fix` | 버그 수정 | `fix: 이미지 업로드 오류 수정` |
| `docs` | 문서 수정 | `docs: README 업데이트` |
| `style` | 코드 포맷팅 (기능 변경 없음) | `style: 들여쓰기 수정` |
| `refactor` | 코드 리팩토링 | `refactor: API 호출 로직 개선` |
| `test` | 테스트 코드 추가 | `test: YOLO 모델 단위 테스트` |
| `chore` | 빌드/설정 변경 | `chore: 패키지 버전 업데이트` |
| `perf` | 성능 개선 | `perf: 이미지 로딩 속도 개선` |

### Scope (선택)
프로젝트 영역을 명시합니다.

**예시:**
- `(llm)`: LLM 관련
- `(sam2)`: SAM2 관련
- `(yolo)`: YOLO 관련
- `(dnn)`: DNN 관련
- `(ui)`: 사용자 인터페이스
- `(api)`: 백엔드 API
- `(db)`: 데이터베이스

### Subject (필수)
- 50자 이내
- 명령문 형태 ("추가했다" ❌, "추가" ✅)
- 마침표 없음
- 한글 또는 영어

### 예시

✅ **좋은 커밋 메시지:**
```bash
feat(yolo): 상의/하의 자동 분리 기능 추가

YOLO Pose를 활용하여 전신 사진에서 상의와 하의를 
자동으로 분리하는 기능을 구현했습니다.

Closes #42
```

```bash
fix(api): 이미지 업로드 시 500 에러 수정
```

```bash
refactor(dnn): 스타일 예측 모델 코드 정리
```

```bash
docs: 브랜치 사용 가이드 추가
```

❌ **나쁜 커밋 메시지:**
```bash
update
수정함
fix bug
asdfasdf
ㅇㅇ
```

---

## 🔄 작업 플로우

### 일반적인 개발 플로우

```bash
# 1. 최신 develop 받아오기
git checkout develop
git pull origin develop

# 2. 작업할 feature 브랜치로 이동
git checkout feature/llm
git pull origin feature/llm

# 3. develop 최신 변경사항 가져오기 (충돌 방지)
git merge develop

# 4. 작업 진행
# 코드 작성...

# 5. 변경사항 확인
git status
git diff

# 6. 커밋
git add .
git commit -m "feat(llm): 스타일 추천 프롬프트 개선"

# 7. GitHub에 push
git push origin feature/llm

# 8. GitHub에서 Pull Request 생성
# feature/llm → develop

# 9. 코드 리뷰 후 merge
```

### 긴급 버그 수정 플로우

```bash
# 1. develop에서 bugfix 브랜치 생성
git checkout develop
git checkout -b bugfix/urgent-login-error

# 2. 버그 수정
# 코드 수정...

# 3. 커밋 및 push
git add .
git commit -m "fix: 로그인 세션 만료 오류 긴급 수정"
git push origin bugfix/urgent-login-error

# 4. PR 생성 및 즉시 merge
# bugfix/urgent-login-error → develop

# 5. 브랜치 삭제 (merge 완료 후)
git branch -d bugfix/urgent-login-error
git push origin --delete bugfix/urgent-login-error
```

---

## 💻 자주 사용하는 명령어

### 브랜치 관련

```bash
# 현재 브랜치 확인
git branch

# 모든 브랜치 보기 (로컬 + 원격)
git branch -a

# 브랜치 이동
git checkout feature/llm

# 새 브랜치 생성 및 이동
git checkout -b feature/new-feature

# 브랜치 삭제 (로컬)
git branch -d feature/old-feature

# 브랜치 삭제 (원격)
git push origin --delete feature/old-feature
```

### 커밋 관련

```bash
# 변경사항 확인
git status

# 변경 내용 상세 보기
git diff

# 모든 파일 스테이징
git add .

# 특정 파일만 스테이징
git add file.py

# 커밋
git commit -m "feat: 새 기능"

# 마지막 커밋 메시지 수정
git commit --amend

# 커밋 로그 보기
git log --oneline
```

### 동기화 관련

```bash
# 원격 변경사항 가져오기 (merge 안 함)
git fetch origin

# 원격 변경사항 가져오기 + merge
git pull origin develop

# 로컬 변경사항 올리기
git push origin feature/llm

# 강제 push (주의!)
git push -f origin feature/llm
```

### 병합 관련

```bash
# develop 브랜치를 현재 브랜치에 병합
git merge develop

# 충돌 발생 시 현재 상태 확인
git status

# 병합 취소
git merge --abort

# 충돌 해결 후
git add .
git commit
```

### 되돌리기 관련

```bash
# 마지막 커밋 취소 (변경사항 유지)
git reset --soft HEAD~1

# 마지막 커밋 취소 (변경사항 버림)
git reset --hard HEAD~1

# 특정 파일 변경사항 취소
git checkout -- file.py

# 모든 변경사항 취소 (주의!)
git reset --hard HEAD
```

---

## ⚠️ 주의사항

### 1. main 브랜치는 절대 직접 수정 금지
```bash
# ❌ 이렇게 하지 마세요
git checkout main
git add .
git commit -m "수정"
```

### 2. develop에서 직접 작업 금지
```bash
# ❌ 이렇게 하지 마세요
git checkout develop
# 작업...

# ✅ 이렇게 하세요
git checkout feature/llm
# 작업...
```

### 3. push 전에 항상 pull
```bash
# ✅ 충돌 방지
git pull origin feature/llm
git push origin feature/llm
```

### 4. 큰 파일은 .gitignore 추가
```bash
# 모델 파일, 데이터셋 등
*.pth
*.h5
*.pkl
data/
uploads/
```

### 5. 민감 정보 커밋 금지
```bash
# .env 파일 예시
API_KEY=your-secret-key
DB_PASSWORD=password123

# .gitignore에 추가
.env
.env.local
config.json
```

---

## 📞 도움이 필요할 때

### 자주 발생하는 문제

**1. 충돌(Conflict) 발생**
```bash
# 충돌 파일 확인
git status

# 파일 열어서 수동 해결
# <<<<<<< HEAD
# 내 코드
# =======
# 다른 사람 코드
# >>>>>>> feature/llm

# 해결 후
git add .
git commit
```

**2. 잘못된 브랜치에서 작업**
```bash
# 변경사항 임시 저장
git stash

# 올바른 브랜치로 이동
git checkout feature/llm

# 변경사항 복원
git stash pop
```

**3. 커밋 메시지 오타**
```bash
# 마지막 커밋 메시지 수정
git commit --amend

# push 했다면 강제 push (주의!)
git push -f origin feature/llm
```

---

## 📚 추가 학습 자료

- [Git 공식 문서](https://git-scm.com/doc)
- [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

**마지막 업데이트:** 2025-10-14  
**작성자:** 꼬까옷 팀

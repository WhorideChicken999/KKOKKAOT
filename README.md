# 꼬까옷 (kkokkaot) 👔

AI 기반 가상 옷장 및 패션 추천 서비스

## 📋 프로젝트 개요

꼬까옷은 사용자의 옷장을 디지털화하고, AI(LLM/YOLO/SAM2/DNN)를 활용하여 스타일 추천 및 코디 제안을 제공하는 가상 옷장 애플리케이션입니다.

[대용량 데이터 다운로드 (Google Drive 폴더)](https://drive.google.com/drive/folders/1NunAF1a3_fgePWbFsZ_4mUStBeRh_SvF?usp=sharing) :contentReference[oaicite:0]{index=0}

## 📚 문서 - 사전 숙지 필수

- **[브랜치 사용 가이드](./BRANCH_GUIDE.md)** - Git 브랜치 전략 및 커밋 규칙
- **[필수 라이브러리](./kkokkaot_req.md)** - requirements.txt

## 🎯 주요 기능

- **가상 옷장 관리**: 사용자의 의류를 촬영하여 디지털 옷장으로 관리
- **AI 기반 옷 인식**: YOLO 모델을 활용한 자동 의류 탐지 및 분류
- **스타일 추천**: LLM을 활용한 사용자의 옷장 데이터를 기반으로 한 개인화 추천
- **포즈 분석**: YOLO Pose를 활용한 착용 이미지 분석
- **데이터베이스 연동**: PostgreSQL 기반 의류 정보 저장 및 관리

## 🗂️ 프로젝트 구조

```
kkokkaot/
├── c/                           # 핵심 처리 모듈
│   ├── 00_data_sampling.py     # 데이터 샘플링
│   ├── 01_check_gender_wear.py # 성별/착용 확인
│   ├── 02_chroma.py            # Chroma DB 연동
│   ├── 03_embeded_test.py      # 임베딩 테스트
│   ├── 04_yolopose.py          # YOLO Pose 모델
│   ├── 06_yolo_cut.py          # 이미지 자동 크롭
│   ├── 07_postgresql.sql       # DB 스키마
│   ├── 08_connect.py           # DB 연결
│   ├── 09_add_data.py          # 데이터 추가
│   ├── 10_recommendation.py    # 추천 알고리즘
│   └── _main_pipeline.py       # 메인 파이프라인
│
├── project/                    # React Native 프로젝트
│   └── (Expo 앱 파일들)
│
├── ppt/                        # 진행상황 공유 ppt
├── .expo/                      # Expo 설정
├── backend_server.py           # FastAPI/Flask 백엔드 서버
├── package.json                # Node.js 의존성
└── kkokkaot_req.txt            # Python 의존성
```

## 🧪 기술 스택

### Frontend
- **React Native** + **Expo**: 크로스 플랫폼 모바일 앱
- **JavaScript/TypeScript**: 앱 로직 구현

### Backend
- **Python**: 핵심 AI 처리 로직
- **FastAPI/Flask**: REST API 서버
- **PostgreSQL**: 사용자 및 의류 데이터 저장

### AI/ML
- **YOLO v11**: 의류 객체 탐지
- **YOLO Pose**: 착용 포즈 분석
- **PyTorch**: 딥러닝 모델 프레임워크
- **LLM**: 추후 추가 예정
- **Chroma DB**: 벡터 데이터베이스 (임베딩 저장)

### 추가 기술
- **RDKit**: 화학 구조 처리 (옷감 분석용)
- **OpenCV**: 이미지 전처리
- **ngrok**: 로컬 서버 터널링

## 🚀 시작하기

### 필요 조건

- Python 3.8+
- Node.js 16+
- PostgreSQL 12+
- Expo CLI

### 설치 방법

**1. Python 의존성 설치**
```bash
pip install -r kkokkaot_req.txt
```

**2. Node.js 의존성 설치**
```bash
npm install
# 또는
yarn install
```

**3. 데이터베이스 설정**
```bash
# PostgreSQL 설치 후
psql -U postgres -f c/07_postgresql.sql
```

**4. 환경 변수 설정**
```bash
# .env 파일 생성
DB_HOST=localhost
DB_PORT=5432
DB_NAME=kkokkaot
DB_USER=your_username
DB_PASSWORD=your_password
```

### 실행 방법

**백엔드 서버 실행**
```bash
python backend_server.py
```

**프론트엔드 앱 실행**
```bash
cd project
expo start
```

## 📊 주요 모듈 설명

### 1. 데이터 수집 (00_data_sampling.py)
- 사용자 옷 이미지 수집 및 샘플링
- 데이터 전처리 및 정규화

### 2. 의류 인식 (04_yolopose.py, 06_yolo_cut.py)
- YOLO 모델을 활용한 의류 탐지
- 자동 크롭 및 배경 제거
- 포즈 분석을 통한 착용 상태 확인

### 3. 추천 시스템 (10_recommendation.py)
- 사용자 취향 분석
- 유사 의류 검색
- 코디 조합 추천

### 4. 데이터베이스 (08_connect.py, 09_add_data.py)
- PostgreSQL 연동
- 의류 정보 CRUD
- 사용자 데이터 관리

### 5. 메인 파이프라인 (_main_pipeline.py)
- 전체 프로세스 통합
- 이미지 입력 → 분석 → 저장 → 추천

## 🔍 AI 모델 정보

### YOLO v11
- **용도**: 의류 객체 탐지
- **입력**: 사용자 촬영 이미지
- **출력**: 의류 종류, 위치, 신뢰도

### YOLO Pose
- **용도**: 착용 이미지 분석
- **입력**: 사람이 포함된 이미지
- **출력**: 신체 키포인트, 의류 착용 상태

### Fashion Models (fashion_*.pth)
- **fashion_top_model.pth**: 상의 분류
- **fashion_bottom_model.pth**: 하의 분류
- **fashion_attr_resnet50_epochs200_best.pth**: 의류 속성 분석

> ⚠️ **주의**: 모델 파일(.pth)은 Git에 포함되지 않습니다. 별도로 다운로드 받아야 합니다.
> [모델 다운로드 링크] (Google Drive 등)

## 📱 앱 기능

1. **옷장 등록**
   - 카메라로 의류 촬영
   - 자동 배경 제거 및 분류
   - 옷장에 저장

2. **스타일 추천**
   - AI 기반 코디 제안
   - 날씨/일정 고려한 추천
   - 유사 스타일 검색

3. **옷장 관리**
   - 카테고리별 정리
   - 착용 빈도 통계
   - 의류 태그 관리

## 📈 개발 타임라인

- **10월 5일**: 프로젝트 초기 설정
- **10월 10~13일**: 핵심 AI 모듈 개발
- **진행 중**: 앱 UI/UX 개선 및 기능 추가

## 🤝 협업 가이드

### Git 작업 흐름
```bash
# 최신 코드 받기
git pull origin main

# 새 브랜치 생성
git checkout -b feature/your-feature

# 작업 후 커밋
git add .
git commit -m "Add: your feature"

# 푸시
git push origin feature/your-feature
```

### 주의사항
- 대용량 파일(.pth, 이미지)은 Git에 올리지 않기
- .gitignore 확인 후 커밋
- 민감 정보(.env)는 절대 커밋 금지

## 🐛 문제 해결

### 모델 파일 없음
```bash
# 모델 파일을 models/ 폴더에 다운로드
# 또는 팀원에게 파일 요청
```

### 데이터베이스 연결 오류
```bash
# PostgreSQL 실행 확인
# .env 파일 설정 확인
```

### Expo 앱 실행 오류
```bash
# node_modules 재설치
rm -rf node_modules
npm install
```

## 👨‍💻 팀원

- **이문용** - 1석사
- **김재현** - 2석사
- **김현진** - 아이1
- **어지유** - 아이2
- **박재익** - 3석사(취업 후 중도 하차)

---

⭐ Made with 💙 by 삼석사와아이들

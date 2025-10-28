"""
꼬까옷 백엔드 서버 (리팩토링 버전)
- 모듈화된 구조
- 유지보수 용이
"""
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from pathlib import Path
import sys

# 설정 및 모델 import
from config import settings
sys.path.append(str(Path(__file__).parent))
from pipeline.main import FashionPipeline
from _llm_recommender import LLMRecommender
from _advanced_recommender import AdvancedFashionRecommender

# 라우터 import
from routers import auth, wardrobe, recommendations, chat, images, admin, weather


# ============================================
# 전역 AI 시스템
# ============================================
pipeline = None
llm_recommender = None
advanced_recommender = None


# ============================================
# 서버 생명주기 관리
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행"""
    global pipeline, llm_recommender, advanced_recommender
    
    # === Startup ===
    print("\n" + "="*60)
    print("🚀 꼬까옷 서버 초기화 중...")
    print("="*60 + "\n")
    
    # 1. AI 파이프라인 초기화
    print("🤖 AI 파이프라인 초기화 중...")
    try:
        pipeline = FashionPipeline(
            gender_model_path=settings.MODEL_PATHS['gender_model'],
            style_model_path=settings.MODEL_PATHS['style_model'],
            yolo_model_path=settings.MODEL_PATHS['yolo_detection'],
            top_model_path=settings.MODEL_PATHS['top_model'],
            bottom_model_path=settings.MODEL_PATHS['bottom_model'],
            outer_model_path=settings.MODEL_PATHS['outer_model'],
            dress_model_path=settings.MODEL_PATHS['dress_model'],
            db_config=settings.DB_CONFIG
        )
        print("✅ AI 파이프라인 초기화 완료!\n")
        
        # 라우터에 pipeline 주입
        auth.pipeline = pipeline
        wardrobe.pipeline = pipeline
        recommendations.pipeline = pipeline
        admin.pipeline = pipeline
        chat.pipeline = pipeline
        
    except Exception as e:
        print(f"⚠️ AI 파이프라인 초기화 실패: {e}")
        print("⚠️ 일부 기능이 제한됩니다.\n")
        pipeline = None
    
    # 2. LLM 추천 시스템 초기화
    print("💬 LLM 추천 시스템 초기화 중...")
    try:
        llm_recommender = LLMRecommender(db_config=settings.DB_CONFIG)
        print("✅ LLM 추천 시스템 초기화 완료!\n")
        chat.llm_recommender = llm_recommender
    except Exception as e:
        print(f"⚠️ LLM 추천 시스템 초기화 실패: {e}")
        print("⚠️ 대화형 추천 기능이 제한됩니다.\n")
        llm_recommender = None
    
    # 3. 고급 추천 시스템 초기화
    print("🎨 고급 추천 시스템 초기화 중...")
    try:
        advanced_recommender = AdvancedFashionRecommender()
        print("✅ 고급 추천 시스템 초기화 완료!\n")
        recommendations.advanced_recommender = advanced_recommender
    except Exception as e:
        print(f"⚠️ 고급 추천 시스템 초기화 실패: {e}")
        print("⚠️ 고급 추천 기능이 제한됩니다.\n")
        advanced_recommender = None
    
    print("="*60)
    print("✅ 서버 초기화 완료!")
    print(f"📡 서버 주소: http://{settings.SERVER_HOST}:{settings.SERVER_PORT}")
    print("="*60 + "\n")
    
    yield  # 서버 실행
    
    # === Shutdown ===
    print("\n" + "="*60)
    print("🛑 서버 종료 중...")
    print("="*60 + "\n")
    
    if pipeline:
        pipeline.close()
        print("✅ AI 파이프라인 종료")
    
    if llm_recommender:
        llm_recommender.close()
        print("✅ LLM 추천 시스템 종료")
    
    print("\n✅ 서버 종료 완료\n")


# ============================================
# FastAPI 앱 생성
# ============================================
app = FastAPI(
    title="꼬까옷 API",
    description="AI 기반 가상 옷장 및 패션 추천 서비스",
    version="2.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# 라우터 등록
# ============================================
app.include_router(auth.router)
app.include_router(wardrobe.router)
app.include_router(recommendations.router)
app.include_router(chat.router)
app.include_router(images.router)
app.include_router(admin.router)
app.include_router(weather.router)


# ============================================
# 기존 경로 호환성 엔드포인트
# ============================================
@app.post("/api/upload-wardrobe")
async def upload_wardrobe_legacy(
    user_id: int = Form(...),
    image: UploadFile = File(...)
):
    """기존 경로 호환 (프론트엔드는 'image'라는 이름으로 파일 전송)"""
    print(f"[DEBUG] upload_wardrobe_legacy 호출: user_id={user_id}, file={image.filename if image else None}")
    try:
        result = await wardrobe.upload_wardrobe_item(user_id, image)
        print(f"[DEBUG] 결과: {result}")
        return result
    except Exception as e:
        print(f"[ERROR] 업로드 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": str(e),
            "error_type": "server_error"
        }


# ============================================
# 루트 엔드포인트
# ============================================
@app.get("/")
def root():
    """서버 상태 확인"""
    return {
        "service": "꼬까옷 API",
        "version": "2.0.0",
        "status": "running",
        "ai_pipeline": "active" if pipeline else "inactive",
        "llm_recommender": "active" if llm_recommender else "inactive",
        "advanced_recommender": "active" if advanced_recommender else "inactive"
    }


# ============================================
# 서버 실행
# ============================================
if __name__ == "__main__":
    uvicorn.run(
        "backend_server_new:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=False
    )


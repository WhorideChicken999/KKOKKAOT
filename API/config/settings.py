"""
설정 파일
모든 경로, DB 설정, API 키 등을 여기서 관리합니다.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

# 환경변수 로드
load_dotenv()

# 기본 경로
BASE_DIR = Path(__file__).parent.parent
API_DIR = BASE_DIR

# 데이터베이스 설정
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'kkokkaot_closet'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '000000')
}

# AI 모델 경로
MODEL_PATHS = {
    'gender_model': str(API_DIR / "pre_trained_weights" / "gender_best_model.pth"),
    'style_model': str(API_DIR / "pre_trained_weights" / "k_fashion_final_model_1019.pth"),
    'yolo_detection': str(API_DIR / "pre_trained_weights" / "yolo_best.pt"),
    'yolo_pose': str(API_DIR / "pre_trained_weights" / "yolo11n-pose.pt"),
    'top_model': str(API_DIR / "pre_trained_weights" / "top_best_model.pth"),
    'bottom_model': str(API_DIR / "pre_trained_weights" / "bottom_best_model.pth"),
    'outer_model': str(API_DIR / "pre_trained_weights" / "outer_best_model.pth"),
    'dress_model': str(API_DIR / "pre_trained_weights" / "dress_best_model.pth"),
    'schema': str(API_DIR / "kfashion_attributes_schema.csv"),
    'chroma_db': str(API_DIR / "chroma_db")
}

# 이미지 저장 경로
IMAGE_PATHS = {
    'uploaded': API_DIR / "uploaded_images",
    'processed': API_DIR / "processed_images",
    'default_items': API_DIR / "default_items",
    'represent': API_DIR / "represent_image"
}

# 디렉토리 생성
for path in IMAGE_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# 날씨 API
WEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')
WEATHER_API_URL = "http://api.openweathermap.org/data/2.5/weather"

# 서버 설정
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 4000
DEBUG = False


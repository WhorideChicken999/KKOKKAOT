"""
이미지 제공 API
- 업로드된 이미지 제공
- 처리된 이미지 제공
- 대표 이미지 제공
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import os
import urllib.parse
import random
from config import settings

router = APIRouter(prefix="/api", tags=["이미지"])

# 경로 설정
UPLOAD_DIR = settings.IMAGE_PATHS['uploaded']
PROCESSED_DIR = settings.IMAGE_PATHS['processed']
DEFAULT_ITEMS_DIR = settings.IMAGE_PATHS['default_items']
REPRESENT_DIR = settings.IMAGE_PATHS['represent']


@router.get("/images/{filename}")
def get_image(filename: str):
    """업로드된 이미지 또는 기본 아이템 이미지 파일 제공"""
    
    # 1. uploaded_images 폴더에서 찾기
    file_path = UPLOAD_DIR / filename
    if os.path.exists(str(file_path)):
        return FileResponse(str(file_path))
    
    # 2. default_items 폴더에서 찾기
    default_path = DEFAULT_ITEMS_DIR / filename
    if os.path.exists(str(default_path)):
        return FileResponse(str(default_path))
    
    # 3. 둘 다 없으면 404
    print(f"❌ 이미지 파일을 찾을 수 없습니다: {filename}")
    print(f"   - uploaded_images: {file_path}")
    print(f"   - default_items: {default_path}")
    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/represent-images/{filename}")
def get_represent_image(filename: str):
    """스타일 대표 이미지 파일 제공"""
    
    # URL 디코딩 처리
    decoded_filename = urllib.parse.unquote(filename)
    print(f"🔍 요청된 파일명: {filename}")
    print(f"🔍 디코딩된 파일명: {decoded_filename}")
    
    # represent_image 폴더에서 찾기
    represent_path = REPRESENT_DIR / decoded_filename
    print(f"🔍 검색 경로: {represent_path}")
    
    if os.path.exists(str(represent_path)):
        print(f"✅ 파일 발견: {represent_path}")
        return FileResponse(str(represent_path))
    
    # 없으면 404
    print(f"❌ 스타일 대표 이미지를 찾을 수 없습니다: {decoded_filename}")
    print(f"   - represent_image: {represent_path}")
    raise HTTPException(status_code=404, detail="Represent image not found")


@router.get("/processed-images/{category}/{filename}")
def get_processed_image_by_category(category: str, filename: str):
    """카테고리별 분리된 이미지 파일 제공 (full/top/bottom/outer) - 기본 아이템용"""
    
    # 허용된 카테고리 체크
    allowed_categories = ['full', 'top', 'bottom', 'outer', 'dress']
    if category not in allowed_categories:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    # 1순위: API/processed_images에서 찾기 (사용자 아이템)
    api_file_path = PROCESSED_DIR / category / filename
    if os.path.exists(str(api_file_path)):
        return FileResponse(str(api_file_path))
    
    # 2순위: processed_default_images에서 찾기 (기본 아이템)
    default_file_path = Path("D:/kkokkaot/processed_default_images") / category / filename
    if os.path.exists(str(default_file_path)):
        print(f"✅ processed_default_images에서 이미지 사용: {default_file_path}")
        return FileResponse(str(default_file_path))
    
    # 3순위: 기본 아이템 이미지에서 찾기
    default_file_path = DEFAULT_ITEMS_DIR / filename
    if os.path.exists(str(default_file_path)):
        print(f"✅ 기본 아이템 이미지 사용: {default_file_path}")
        return FileResponse(str(default_file_path))
    
    print(f"❌ 이미지 파일을 찾을 수 없습니다: {api_file_path}")
    print(f"❌ 기본 아이템 이미지도 찾을 수 없습니다: {default_file_path}")
    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/processed-images/user_{user_id}/{category}/{filename}")
def get_user_processed_image(user_id: int, category: str, filename: str):
    """사용자별 분리된 이미지 파일 제공"""
    
    # 허용된 카테고리 체크
    allowed_categories = ['full', 'top', 'bottom', 'outer', 'dress']
    if category not in allowed_categories:
        raise HTTPException(status_code=400, detail="Invalid category")
    
    # 사용자별 이미지 경로
    user_file_path = PROCESSED_DIR / f"user_{user_id}" / category / filename
    
    if os.path.exists(str(user_file_path)):
        print(f"✅ 사용자 {user_id} 이미지 사용: {user_file_path}")
        return FileResponse(str(user_file_path))
    
    print(f"⚠️ 사용자 {user_id} 이미지를 찾을 수 없습니다: {user_file_path}")
    
    # 1순위: 기본 아이템 이미지에서 찾기
    if DEFAULT_ITEMS_DIR.exists():
        # 기본 아이템 폴더에서 랜덤 이미지 선택
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(DEFAULT_ITEMS_DIR.glob(f"*{ext}"))
        
        if image_files:
            random_image = random.choice(image_files)
            print(f"✅ 기본 아이템 이미지로 대체: {random_image}")
            return FileResponse(str(random_image))
    
    # 2순위: 플레이스홀더 이미지
    placeholder_path = DEFAULT_ITEMS_DIR / "placeholder.jpg"
    if os.path.exists(str(placeholder_path)):
        print(f"✅ 플레이스홀더 이미지 사용: {placeholder_path}")
        return FileResponse(str(placeholder_path))
    
    # 3순위: 404
    print(f"⚠️ 모든 이미지 소스 실패")
    raise HTTPException(status_code=404, detail="User image not found")


@router.get("/processed-images/{filename}")
def get_processed_image(filename: str):
    """분리된 이미지 제공 (레거시)"""
    
    # full, top, bottom, outer 순서로 검색
    categories = ['full', 'top', 'bottom', 'outer', 'dress']
    
    # 1순위: API/processed_images에서 찾기
    for category in categories:
        api_file_path = PROCESSED_DIR / category / filename
        if os.path.exists(str(api_file_path)):
            return FileResponse(str(api_file_path))
    
    # 2순위: processed_default_images에서 찾기
    for category in categories:
        default_file_path = Path("D:/kkokkaot/processed_default_images") / category / filename
        if os.path.exists(str(default_file_path)):
            print(f"✅ processed_default_images에서 이미지 사용: {default_file_path}")
            return FileResponse(str(default_file_path))
    
    # 3순위: default_items에서 찾기
    default_file_path = DEFAULT_ITEMS_DIR / filename
    if os.path.exists(str(default_file_path)):
        print(f"✅ 기본 아이템 이미지 사용: {default_file_path}")
        return FileResponse(str(default_file_path))
    
    print(f"❌ 이미지 파일을 찾을 수 없습니다: {filename}")
    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/default-images/{filename}")
def get_default_image(filename: str):
    """기본 아이템 이미지 제공"""
    
    file_path = DEFAULT_ITEMS_DIR / filename
    
    if os.path.exists(str(file_path)):
        return FileResponse(str(file_path))
    else:
        print(f"❌ 기본 아이템 이미지를 찾을 수 없습니다: {file_path}")
        raise HTTPException(status_code=404, detail="Default image not found")


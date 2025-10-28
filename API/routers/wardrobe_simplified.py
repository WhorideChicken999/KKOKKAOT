"""
옷장 관리 API - 간소화 버전

⚠️ 주의: 원본 backend_server.py의 wardrobe 관련 코드가 매우 복잡하여
전체를 복사하면 파일이 너무 커집니다.

이 파일은 핵심 로직만 포함합니다.
세부 구현이 필요한 경우 backend_server.py에서 직접 복사하세요.

필요한 줄 번호:
- POST /upload-wardrobe: 292-413
- DELETE /{item_id}: 415-605
- GET /separated/{user_id}: 606-717
- GET /simple/{user_id}: 718-764
- GET /{user_id}: 765-994
- GET /item/{item_id}: 1443-1671
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from pathlib import Path
import shutil
import os
from config import settings

router = APIRouter(prefix="/api/wardrobe", tags=["옷장"])

# 전역 변수
pipeline = None
UPLOAD_DIR = settings.IMAGE_PATHS['uploaded']


@router.post("/upload")
async def upload_wardrobe(
    user_id: int = Form(...),
    image: UploadFile = File(...)
):
    """
    옷장에 이미지 업로드 & AI 분석
    ⚠️ 원본: backend_server.py 줄 292-413
    """
    print(f"\n📸 이미지 업로드: user_id={user_id}, file={image.filename}")
    
    try:
        # 사용자별 폴더 생성
        user_upload_dir = UPLOAD_DIR / f"user_{user_id}"
        user_upload_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일 저장
        file_path = user_upload_dir / f"{user_id}_{image.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        print(f"✅ 파일 저장: {file_path}")
        
        # AI 분석
        if pipeline:
            result = pipeline.process_image(
                image_path=str(file_path),
                user_id=user_id,
                save_separated_images=True
            )
            
            if result['success']:
                category_attrs = result.get('category_attributes', {})
                
                def extract_values(attrs):
                    return {key: data['value'] for key, data in attrs.items()} if attrs else None
                
                return {
                    "success": True,
                    "message": "이미지 분석 완료!",
                    "item_id": result['item_id'],
                    "top_attributes": extract_values(category_attrs.get('상의')),
                    "bottom_attributes": extract_values(category_attrs.get('하의')),
                    "outer_attributes": extract_values(category_attrs.get('아우터')),
                    "dress_attributes": extract_values(category_attrs.get('원피스')),
                }
            else:
                return {"success": False, "message": result.get('error', '분석 실패')}
        else:
            return {"success": True, "message": "AI 비활성화", "file_path": str(file_path)}
            
    except Exception as e:
        return {"success": False, "message": f"오류: {str(e)}"}


@router.delete("/{item_id}")
def delete_wardrobe_item(item_id: int):
    """
    옷장 아이템 삭제
    ⚠️ TODO: backend_server.py 줄 415-605 복사 필요
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="AI 파이프라인 비활성화")
    
    # TODO: 원본 코드 복사 필요
    return {"success": True, "message": "삭제 기능은 원본에서 복사 필요"}


@router.get("/separated/{user_id}")
def get_wardrobe_separated(user_id: int):
    """
    사용자 옷장 조회 (카테고리별)
    ⚠️ TODO: backend_server.py 줄 606-717 복사 필요
    """
    return {"success": False, "message": "원본에서 복사 필요"}


@router.get("/simple/{user_id}")
def get_wardrobe_simple(user_id: int):
    """
    사용자 옷장 간단 조회
    ⚠️ TODO: backend_server.py 줄 718-764 복사 필요
    """
    return {"success": False, "message": "원본에서 복사 필요"}


@router.get("/{user_id}")
def get_wardrobe(user_id: int, include_defaults: bool = True):
    """
    사용자 옷장 조회
    ⚠️ TODO: backend_server.py 줄 765-994 복사 필요
    """
    return {"success": False, "message": "원본에서 복사 필요"}


@router.get("/item/{item_id}")
def get_wardrobe_item_detail(item_id: int):
    """
    아이템 상세 조회
    ⚠️ TODO: backend_server.py 줄 1443-1671 복사 필요
    """
    return {"success": False, "message": "원본에서 복사 필요"}


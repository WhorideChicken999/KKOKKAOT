"""
옷장 관리 API
- 이미지 업로드 및 AI 분석
- 옷장 아이템 조회/삭제
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
from typing import Optional
import json

router = APIRouter(prefix="/api/wardrobe", tags=["옷장"])

# 전역 변수 (메인에서 주입)
pipeline = None


@router.post("/upload")
async def upload_wardrobe_item(
    user_id: int = Form(...),
    file: UploadFile = File(...)
):
    """
    옷장 아이템 업로드 및 AI 분석
    """
    print(f"\n{'='*60}")
    print(f"📤 옷장 업로드 요청 (user_id: {user_id})")
    print(f"{'='*60}")
    
    if not pipeline:
        return {
            "success": False,
            "message": "서버 초기화 실패",
            "error_type": "server_error"
        }
    
    try:
        # 1. 파일 저장
        upload_dir = Path("./uploaded_images") / f"user_{user_id}"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"  ✅ 파일 저장 완료: {file_path}")
        
        # 2. AI 파이프라인 처리
        result = pipeline.process(
            image_path=str(file_path),
            user_id=user_id,
            save_to_db=True
        )
        
        if result['success']:
            print(f"  ✅ AI 분석 완료 (item_id: {result.get('item_id')})")
            print(f"{'='*60}\n")
            return result
        else:
            print(f"  ❌ AI 분석 실패: {result.get('message')}")
            print(f"{'='*60}\n")
            return result
            
    except Exception as e:
        print(f"❌ 업로드 오류: {e}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "message": f"업로드 중 오류 발생: {str(e)}",
            "error_type": "upload_error"
        }


@router.delete("/{item_id}")
async def delete_wardrobe_item(item_id: int):
    """옷장 아이템 삭제"""
    print(f"\n{'='*60}")
    print(f"🗑️ 아이템 삭제 요청 (item_id: {item_id})")
    print(f"{'='*60}")
    
    if not pipeline:
        return {"success": False, "message": "서버 초기화 실패"}
    
    try:
        with pipeline.db.conn.cursor() as cur:
            # 1. 속성 테이블들 삭제
            cur.execute("DELETE FROM top_attributes_new WHERE item_id = %s", (item_id,))
            cur.execute("DELETE FROM bottom_attributes_new WHERE item_id = %s", (item_id,))
            cur.execute("DELETE FROM outer_attributes_new WHERE item_id = %s", (item_id,))
            cur.execute("DELETE FROM dress_attributes_new WHERE item_id = %s", (item_id,))
            
            # 2. 메인 테이블 삭제
            cur.execute("DELETE FROM wardrobe_items WHERE item_id = %s", (item_id,))
            
            pipeline.db.conn.commit()
        
        print(f"  ✅ 아이템 삭제 완료")
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "message": "아이템이 삭제되었습니다."
        }
        
    except Exception as e:
        pipeline.db.conn.rollback()
        print(f"❌ 삭제 오류: {e}")
        print(f"{'='*60}\n")
        
        return {
            "success": False,
            "message": f"삭제 중 오류 발생: {str(e)}"
        }


@router.get("/separated/{user_id}")
async def get_wardrobe_separated(user_id: int):
    """사용자 옷장 조회 (사용자 아이템 / 기본 아이템 분리)"""
    print(f"\n{'='*60}")
    print(f"👔 옷장 조회 (분리) - user_id: {user_id}")
    print(f"{'='*60}")
    
    if not pipeline:
        return {
            "success": False,
            "message": "서버 초기화 실패",
            "user_items": [],
            "default_items": []
        }
    
    try:
        with pipeline.db.conn.cursor() as cur:
            # 사용자 아이템
            cur.execute("""
                SELECT 
                    item_id, original_image_path, gender, style,
                    has_top, has_bottom, has_outer, has_dress, is_default
                FROM wardrobe_items
                WHERE user_id = %s AND is_default = FALSE
                ORDER BY item_id DESC
            """, (user_id,))
            user_rows = cur.fetchall()
            
            # 기본 아이템
            cur.execute("""
                SELECT 
                    item_id, original_image_path, gender, style,
                    has_top, has_bottom, has_outer, has_dress, is_default
                FROM wardrobe_items
                WHERE user_id = 0 AND is_default = TRUE
                ORDER BY item_id
                LIMIT 20
            """)
            default_rows = cur.fetchall()
        
        user_items = []
        for row in user_rows:
            item_id = row[0]
            has_top = row[4]
            has_bottom = row[5]
            has_outer = row[6]
            has_dress = row[7]
            
            # 전체 이미지 사용
            image_path = f"/api/processed-images/user_{user_id}/full/item_{item_id}_full.jpg"
            
            # 각 부분별 이미지 경로도 추가
            top_image = f"/api/processed-images/user_{user_id}/top/item_{item_id}_top.jpg" if has_top else None
            bottom_image = f"/api/processed-images/user_{user_id}/bottom/item_{item_id}_bottom.jpg" if has_bottom else None
            outer_image = f"/api/processed-images/user_{user_id}/outer/item_{item_id}_outer.jpg" if has_outer else None
            dress_image = f"/api/processed-images/user_{user_id}/dress/item_{item_id}_dress.jpg" if has_dress else None
            
            user_items.append({
                "item_id": item_id,
                "id": item_id,  # 프론트엔드 호환 (임시)
                "image_path": image_path,  # 전체 이미지
                "top_image": top_image,
                "bottom_image": bottom_image,
                "outer_image": outer_image,
                "dress_image": dress_image,
                "gender": row[2],
                "style": row[3],
                "has_top": has_top,
                "has_bottom": has_bottom,
                "has_outer": has_outer,
                "has_dress": has_dress,
                "is_default": row[8]
            })
        
        default_items = []
        for row in default_rows:
            item_id = row[0]
            has_top = row[4]
            has_bottom = row[5]
            has_outer = row[6]
            has_dress = row[7]
            
            # 전체 이미지 사용 (user_0)
            image_path = f"/api/processed-images/user_0/full/item_{item_id}_full.jpg"
            
            # 각 부분별 이미지 경로도 추가 (user_0)
            top_image = f"/api/processed-images/user_0/top/item_{item_id}_top.jpg" if has_top else None
            bottom_image = f"/api/processed-images/user_0/bottom/item_{item_id}_bottom.jpg" if has_bottom else None
            outer_image = f"/api/processed-images/user_0/outer/item_{item_id}_outer.jpg" if has_outer else None
            dress_image = f"/api/processed-images/user_0/dress/item_{item_id}_dress.jpg" if has_dress else None
            
            default_items.append({
                "item_id": item_id,
                "id": item_id,  # 프론트엔드 호환 (임시)
                "image_path": image_path,  # 전체 이미지
                "top_image": top_image,
                "bottom_image": bottom_image,
                "outer_image": outer_image,
                "dress_image": dress_image,
                "gender": row[2],
                "style": row[3],
                "has_top": has_top,
                "has_bottom": has_bottom,
                "has_outer": has_outer,
                "has_dress": has_dress,
                "is_default": row[8]
            })
        
        print(f"  ✅ 사용자 아이템: {len(user_items)}개")
        print(f"  ✅ 기본 아이템: {len(default_items)}개")
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "message": "조회 성공",
            "user_items": user_items,
            "default_items": default_items
        }
        
    except Exception as e:
        print(f"❌ 조회 오류: {e}")
        print(f"{'='*60}\n")
        
        return {
            "success": False,
            "message": f"조회 중 오류 발생: {str(e)}",
            "user_items": [],
            "default_items": []
        }


@router.get("/simple/{user_id}")
async def get_wardrobe_simple(user_id: int):
    """사용자 옷장 간단 조회 (기본 정보만)"""
    print(f"\n👔 간단 옷장 조회 - user_id: {user_id}")
    
    if not pipeline:
        return {"success": False, "message": "서버 초기화 실패", "items": []}
    
    try:
        with pipeline.db.conn.cursor() as cur:
            cur.execute("""
                SELECT item_id, original_image_path, created_at, style, gender
                FROM wardrobe_items
                WHERE user_id = %s OR (user_id = 0 AND is_default = TRUE)
                ORDER BY is_default, created_at DESC
            """, (user_id,))
            rows = cur.fetchall()
        
        items = []
        for row in rows:
            item_id = row[0]
            
            # is_default 확인을 위해 추가 쿼리 필요 (또는 쿼리 수정)
            # 일단 user_id로 판단
            with pipeline.db.conn.cursor() as cur2:
                cur2.execute("SELECT is_default FROM wardrobe_items WHERE item_id = %s", (item_id,))
                is_default_row = cur2.fetchone()
                is_default = is_default_row[0] if is_default_row else False
            
            # 이미지 경로
            if is_default:
                image_path = f"/api/processed-images/user_0/full/item_{item_id}_full.jpg"
            else:
                image_path = f"/api/processed-images/user_{user_id}/full/item_{item_id}_full.jpg"
            
            items.append({
                "item_id": item_id,
                "id": item_id,  # 프론트엔드 호환 (임시)
                "image_path": image_path,
                "created_at": str(row[2]) if row[2] else None,
                "style": row[3],
                "gender": row[4]
            })
        
        return {"success": True, "items": items}
        
    except Exception as e:
        return {"success": False, "message": str(e), "items": []}


@router.get("/{user_id}")
async def get_wardrobe(
    user_id: int,
    include_defaults: bool = True
):
    """사용자 옷장 전체 조회"""
    print(f"\n{'='*60}")
    print(f"👔 옷장 전체 조회 - user_id: {user_id}")
    print(f"{'='*60}")
    
    if not pipeline:
        return {"success": False, "message": "서버 초기화 실패", "items": []}
    
    try:
        with pipeline.db.conn.cursor() as cur:
            if include_defaults:
                query = """
                    SELECT item_id, original_image_path, created_at, gender, style,
                           has_top, has_bottom, has_outer, has_dress, is_default
                    FROM wardrobe_items
                    WHERE user_id = %s OR (user_id = 0 AND is_default = TRUE)
                    ORDER BY is_default, created_at DESC
                """
                cur.execute(query, (user_id,))
            else:
                query = """
                    SELECT item_id, original_image_path, created_at, gender, style,
                           has_top, has_bottom, has_outer, has_dress, is_default
                    FROM wardrobe_items
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                """
                cur.execute(query, (user_id,))
            
            rows = cur.fetchall()
        
        items = []
        for row in rows:
            item_id = row[0]
            is_default = row[9]
            has_top = row[5]
            has_bottom = row[6]
            has_outer = row[7]
            has_dress = row[8]
            
            # 전체 이미지 경로
            if is_default:
                image_path = f"/api/processed-images/user_0/full/item_{item_id}_full.jpg"
            else:
                image_path = f"/api/processed-images/user_{user_id}/full/item_{item_id}_full.jpg"
            
            # 카테고리별 crop 이미지 경로
            if is_default:
                top_image = f"/api/processed-images/user_0/top/item_{item_id}_top.jpg" if has_top else None
                bottom_image = f"/api/processed-images/user_0/bottom/item_{item_id}_bottom.jpg" if has_bottom else None
                outer_image = f"/api/processed-images/user_0/outer/item_{item_id}_outer.jpg" if has_outer else None
                dress_image = f"/api/processed-images/user_0/dress/item_{item_id}_dress.jpg" if has_dress else None
            else:
                top_image = f"/api/processed-images/user_{user_id}/top/item_{item_id}_top.jpg" if has_top else None
                bottom_image = f"/api/processed-images/user_{user_id}/bottom/item_{item_id}_bottom.jpg" if has_bottom else None
                outer_image = f"/api/processed-images/user_{user_id}/outer/item_{item_id}_outer.jpg" if has_outer else None
                dress_image = f"/api/processed-images/user_{user_id}/dress/item_{item_id}_dress.jpg" if has_dress else None
            
            items.append({
                "item_id": item_id,
                "id": item_id,  # 프론트엔드 호환 (임시)
                "image_path": image_path,
                "top_image": top_image,
                "bottom_image": bottom_image,
                "outer_image": outer_image,
                "dress_image": dress_image,
                "created_at": str(row[2]) if row[2] else None,
                "gender": row[3],
                "style": row[4],
                "has_top": has_top,
                "has_bottom": has_bottom,
                "has_outer": has_outer,
                "has_dress": has_dress,
                "is_default": is_default
            })
        
        print(f"  ✅ 총 {len(items)}개 아이템 조회")
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "message": "조회 성공",
            "items": items
        }
        
    except Exception as e:
        print(f"❌ 조회 오류: {e}")
        print(f"{'='*60}\n")
        
        return {
            "success": False,
            "message": f"조회 중 오류 발생: {str(e)}",
            "items": []
        }


@router.get("/item/{item_id}")
async def get_wardrobe_item_detail(item_id: int):
    """옷장 아이템 상세 조회"""
    print(f"\n👔 아이템 상세 조회 - item_id: {item_id}")
    
    if not pipeline:
        return {"success": False, "message": "서버 초기화 실패"}
    
    try:
        with pipeline.db.conn.cursor() as cur:
            # 메인 정보
            cur.execute("""
                SELECT item_id, original_image_path, created_at, gender, style,
                       has_top, has_bottom, has_outer, has_dress, is_default
                FROM wardrobe_items
                WHERE item_id = %s
            """, (item_id,))
            item_row = cur.fetchone()
            
            if not item_row:
                return {"success": False, "message": "아이템을 찾을 수 없습니다."}
            
            # 속성 조회
            attributes = {}
            
            if item_row[5]:  # has_top
                cur.execute("SELECT * FROM top_attributes_new WHERE item_id = %s", (item_id,))
                top = cur.fetchone()
                if top:
                    attributes['top'] = {
                        "category": top[1],
                        "color": top[2],
                        "fit": top[3],
                        "material": top[4],
                        "print_pattern": top[5],
                        "style": top[6],
                        "sleeve_length": top[7],
                        "gender": top[8]
                    }
            
            if item_row[6]:  # has_bottom
                cur.execute("SELECT * FROM bottom_attributes_new WHERE item_id = %s", (item_id,))
                bottom = cur.fetchone()
                if bottom:
                    attributes['bottom'] = {
                        "category": bottom[1],
                        "color": bottom[2],
                        "fit": bottom[3],
                        "material": bottom[4],
                        "print_pattern": bottom[5],
                        "style": bottom[6],
                        "length": bottom[7],
                        "gender": bottom[8]
                    }
            
            if item_row[7]:  # has_outer
                cur.execute("SELECT * FROM outer_attributes_new WHERE item_id = %s", (item_id,))
                outer = cur.fetchone()
                if outer:
                    attributes['outer'] = {
                        "category": outer[1],
                        "color": outer[2],
                        "fit": outer[3],
                        "material": outer[4],
                        "print_pattern": outer[5],
                        "style": outer[6],
                        "sleeve_length": outer[7],
                        "gender": outer[8]
                    }
            
            if item_row[8]:  # has_dress
                cur.execute("SELECT * FROM dress_attributes_new WHERE item_id = %s", (item_id,))
                dress = cur.fetchone()
                if dress:
                    attributes['dress'] = {
                        "category": dress[1],
                        "color": dress[2],
                        "material": dress[3],
                        "print_pattern": dress[4],
                        "style": dress[5],
                        "gender": dress[6]
                    }
        
        # user_id 조회
        with pipeline.db.conn.cursor() as cur:
            cur.execute("SELECT user_id FROM wardrobe_items WHERE item_id = %s", (item_id,))
            user_id_row = cur.fetchone()
            user_id = user_id_row[0] if user_id_row else 0
        
        # 이미지 경로 생성
        is_default = item_row[9]
        has_top = item_row[5]
        has_bottom = item_row[6]
        has_outer = item_row[7]
        has_dress = item_row[8]
        
        # 이미지 경로 생성 (user_0: 기본 아이템, user_{user_id}: 사용자 아이템)
        if is_default:
            base_user_folder = "user_0"
        else:
            base_user_folder = f"user_{user_id}"
        
        # 전체 이미지 및 부분별 이미지 경로
        image_path = f"/api/processed-images/{base_user_folder}/full/item_{item_id}_full.jpg"
        top_image = f"/api/processed-images/{base_user_folder}/top/item_{item_id}_top.jpg" if has_top else None
        bottom_image = f"/api/processed-images/{base_user_folder}/bottom/item_{item_id}_bottom.jpg" if has_bottom else None
        outer_image = f"/api/processed-images/{base_user_folder}/outer/item_{item_id}_outer.jpg" if has_outer else None
        dress_image = f"/api/processed-images/{base_user_folder}/dress/item_{item_id}_dress.jpg" if has_dress else None
        
        return {
            "success": True,
            "item": {
                "item_id": item_row[0],
                "id": item_row[0],  # 프론트엔드 호환 (임시)
                "image_path": image_path,  # 전체 이미지
                "top_image": top_image,
                "bottom_image": bottom_image,
                "outer_image": outer_image,
                "dress_image": dress_image,
                "created_at": str(item_row[2]) if item_row[2] else None,
                "gender": item_row[3],
                "style": item_row[4],
                "has_top": item_row[5],
                "has_bottom": item_row[6],
                "has_outer": item_row[7],
                "has_dress": item_row[8],
                "is_default": is_default,
                "attributes": attributes
            }
        }
        
    except Exception as e:
        print(f"❌ 상세 조회 오류: {e}")
        return {"success": False, "message": str(e)}

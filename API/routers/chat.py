"""
LLM 챗봇 API
- 대화형 추천
- 이미지 기반 대화
- 채팅 세션 관리
"""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from pathlib import Path
import shutil

router = APIRouter(prefix="/api/chat", tags=["챗봇"])

# 전역 변수 (메인에서 주입)
llm_recommender = None
pipeline = None


@router.post("/upload")
async def chat_upload_image(
    user_id: int = Form(...),
    image: UploadFile = File(...)
):
    """LLM 채팅에서 이미지 업로드 & AI 분석"""
    print(f"\n{'='*60}")
    print(f"📸 LLM 채팅 이미지 업로드")
    print(f"  - user_id: {user_id}")
    print(f"  - filename: {image.filename}")
    print(f"{'='*60}")
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="AI 파이프라인이 초기화되지 않았습니다")
    
    try:
        # 이미지 저장
        from config.settings import IMAGE_PATHS
        user_upload_dir = IMAGE_PATHS['uploaded'] / f"user_{user_id}"
        user_upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = user_upload_dir / f"{user_id}_{image.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        print(f"✅ 파일 저장: {file_path}")
        
        # AI 분석 (YOLO + 속성 예측)
        result = pipeline.process(
            image_path=str(file_path),
            user_id=user_id,
            save_to_db=True
        )
        
        if not result.get('success'):
            return {
                "success": False,
                "message": f"분석 실패: {result.get('error', '알 수 없는 오류')}"
            }
        
        item_id = result.get('item_id')
        print(f"✅ AI 분석 완료 - item_id: {item_id}")
        
        # 저장된 아이템 정보 가져오기
        with pipeline.db.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    w.item_id,
                    w.has_top,
                    w.has_bottom,
                    w.has_outer,
                    w.has_dress,
                    w.gender,
                    t.category as top_category,
                    t.color as top_color,
                    b.category as bottom_category,
                    b.color as bottom_color,
                    o.category as outer_category,
                    o.color as outer_color,
                    d.category as dress_category,
                    d.color as dress_color
                FROM wardrobe_items w
                LEFT JOIN top_attributes_new t ON w.item_id = t.item_id
                LEFT JOIN bottom_attributes_new b ON w.item_id = b.item_id
                LEFT JOIN outer_attributes_new o ON w.item_id = o.item_id
                LEFT JOIN dress_attributes_new d ON w.item_id = d.item_id
                WHERE w.item_id = %s
            """, (item_id,))
            
            row = cur.fetchone()
            
            if not row:
                return {
                    "success": False,
                    "message": "아이템 정보를 찾을 수 없습니다."
                }
            
            # 아이템 이름 생성
            has_top = row[1]
            has_bottom = row[2]
            has_outer = row[3]
            has_dress = row[4]
            gender = row[5]
            
            if has_dress:
                item_name = f"{row[12] or ''} {row[13] or ''}".strip() or "원피스"
                category = "dress"
            elif has_outer:
                item_name = f"{row[10] or ''} {row[11] or ''}".strip() or "아우터"
                category = "outer"
            elif has_top:
                item_name = f"{row[6] or ''} {row[7] or ''}".strip() or "상의"
                category = "top"
            elif has_bottom:
                item_name = f"{row[8] or ''} {row[9] or ''}".strip() or "하의"
                category = "bottom"
            else:
                item_name = "의류 아이템"
                category = "unknown"
        
        print(f"✅ 업로드 완료: {item_name} (ID: {item_id})")
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "message": f"✅ {item_name}을(를) 성공적으로 추가했습니다!",
            "item_id": item_id,
            "item_name": item_name,
            "category": category,
            "gender": gender,
            "image_path": f"/api/processed-images/user_{user_id}/full/item_{item_id}_full.jpg"
        }
        
    except Exception as e:
        print(f"❌ 업로드 오류: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "message": f"업로드 중 오류가 발생했습니다: {str(e)}"
        }


@router.post("/recommend")
async def chat_recommend(
    user_id: int = Form(...),
    message: str = Form(...),
    selected_items: Optional[str] = Form(None)  # 👈 프론트엔드와 일치시킴
):
    """대화형 추천 (LLM)"""
    print(f"\n{'='*60}")
    print(f"💬 LLM 채팅 요청 (user_id: {user_id})")
    print(f"📝 메시지: {message}")
    print(f"👕 선택된 아이템 (raw): {selected_items}")
    print(f"{'='*60}")
    
    if not llm_recommender:
        raise HTTPException(status_code=503, detail="LLM 추천 시스템이 초기화되지 않았습니다")
    
    try:
        # 선택된 아이템 ID 파싱
        item_ids = []
        if selected_items:
            try:
                import json
                item_ids = json.loads(selected_items)
                print(f"✅ 파싱된 선택 아이템 IDs: {item_ids}")
            except Exception as e:
                print(f"❌ 선택 아이템 파싱 실패: {e}")
                item_ids = []
        else:
            print(f"⚠️ 선택된 아이템 없음 (selected_items가 None)")
        
        # LLM 추천 시스템 호출
        result = llm_recommender.chat(
            user_id=user_id,
            user_message=message,
            selected_item_ids=item_ids if item_ids else None
        )
        
        print(f"  ✅ LLM 응답 생성 완료")
        
        # 추천 아이템 ID를 상세 정보로 변환
        recommendation_ids = result.get('recommendations', [])
        detailed_recommendations = []
        
        if recommendation_ids and pipeline:
            with pipeline.db.conn.cursor() as cur:
                for item_id in recommendation_ids:
                    cur.execute("""
                        SELECT 
                            w.item_id,
                            w.has_top,
                            w.has_bottom,
                            w.has_outer,
                            w.has_dress,
                            w.is_default,
                            w.gender,
                            w.style,
                            t.category as top_category,
                            t.color as top_color,
                            b.category as bottom_category,
                            b.color as bottom_color,
                            o.category as outer_category,
                            o.color as outer_color,
                            d.category as dress_category,
                            d.color as dress_color
                        FROM wardrobe_items w
                        LEFT JOIN top_attributes_new t ON w.item_id = t.item_id
                        LEFT JOIN bottom_attributes_new b ON w.item_id = b.item_id
                        LEFT JOIN outer_attributes_new o ON w.item_id = o.item_id
                        LEFT JOIN dress_attributes_new d ON w.item_id = d.item_id
                        WHERE w.item_id = %s
                    """, (item_id,))
                    
                    row = cur.fetchone()
                    if row:
                        has_top = row[1]
                        has_bottom = row[2]
                        has_outer = row[3]
                        has_dress = row[4]
                        is_default = row[5]
                        gender = row[6]
                        style = row[7]
                        
                        # 전체 이미지 경로
                        if is_default:
                            # 기본 아이템: user_0 폴더에서 찾기
                            image_path = f"/api/processed-images/user_0/full/item_{item_id}_full.jpg"
                            print(f"  📸 기본 아이템 {item_id} 이미지 경로: {image_path}")
                        else:
                            # 사용자 아이템: user_{user_id} 폴더에서 찾기
                            image_path = f"/api/processed-images/user_{user_id}/full/item_{item_id}_full.jpg"
                            print(f"  📸 사용자 아이템 {item_id} 이미지 경로: {image_path}")
                        
                        detailed_recommendations.append({
                            "item_id": item_id,
                            "id": item_id,
                            "image_path": image_path,
                            "has_top": has_top,
                            "has_bottom": has_bottom,
                            "has_outer": has_outer,
                            "has_dress": has_dress,
                            "is_default": is_default,
                            "gender": gender,
                            "style": style
                        })
        
        print(f"  ✅ 상세 추천 아이템: {len(detailed_recommendations)}개")
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "response": result.get('response', ''),
            "recommendations": detailed_recommendations,
            "context": result.get('context', {}),
            "need_more_info": result.get('need_more_info', False)
        }
        
    except Exception as e:
        print(f"❌ LLM 채팅 오류: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "response": "죄송합니다. 현재 AI 서비스에 일시적인 문제가 발생했습니다.",
            "recommendations": []
        }


@router.post("/reset")
async def reset_chat_history(user_id: int = Form(...)):
    """채팅 기록 초기화"""
    print(f"\n🔄 채팅 기록 초기화 (user_id: {user_id})")
    
    if not llm_recommender:
        raise HTTPException(status_code=503, detail="LLM 추천 시스템이 초기화되지 않았습니다")
    
    try:
        llm_recommender.reset_conversation(user_id)
        
        return {
            "success": True,
            "message": "채팅 기록이 초기화되었습니다."
        }
        
    except Exception as e:
        print(f"❌ 채팅 초기화 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

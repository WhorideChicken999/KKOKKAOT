"""
관리자 API
- 기본 아이템 관리
- 데이터베이스 유지보수
"""
from fastapi import APIRouter, HTTPException
from pathlib import Path

router = APIRouter(prefix="/api", tags=["관리자"])

# 전역 변수 (메인에서 주입)
pipeline = None


def process_default_items_internal():
    """기본 아이템들을 AI 파이프라인으로 처리"""
    print("\n🎯 기본 아이템 AI 분석 시작...")
    
    # 기본 아이템 이미지 폴더 경로
    default_items_dir = Path("./default_items")
    
    if not default_items_dir.exists():
        print("❌ default_items 폴더가 없습니다.")
        print("💡 default_items 폴더를 생성하고 이미지를 넣어주세요.")
        return 0
    
    # 기본 아이템 이미지 파일들 찾기
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(default_items_dir.glob(ext))
    
    if not image_files:
        print("❌ 기본 아이템 이미지가 없습니다.")
        print("💡 default_items 폴더에 이미지 파일을 넣어주세요.")
        return 0
    
    print(f"📁 {len(image_files)}개의 기본 아이템 이미지 발견")
    
    # 기존 기본 아이템 데이터 완전 삭제
    try:
        with pipeline.db.conn.cursor() as cur:
            print("🗑️ 기존 기본 아이템 데이터 완전 삭제 중...")
            
            # 1. 기본 아이템 속성 테이블들 먼저 삭제
            cur.execute("DELETE FROM top_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE is_default = TRUE)")
            cur.execute("DELETE FROM bottom_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE is_default = TRUE)")
            cur.execute("DELETE FROM outer_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE is_default = TRUE)")
            cur.execute("DELETE FROM dress_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE is_default = TRUE)")
            
            # 2. 기본 아이템 메인 테이블 삭제
            cur.execute("DELETE FROM wardrobe_items WHERE is_default = TRUE")
            
            # 3. ChromaDB에서도 삭제
            try:
                # 기본 아이템들의 ChromaDB ID 패턴: item_XXX (user_id=0)
                cur.execute("SELECT chroma_embedding_id FROM wardrobe_items WHERE user_id = 0")
                chroma_ids = cur.fetchall()
                for (chroma_id,) in chroma_ids:
                    if chroma_id:
                        try:
                            pipeline.chroma_collection.delete(ids=[chroma_id])
                        except:
                            pass
            except:
                pass
            
            pipeline.db.conn.commit()
            print("✅ 기존 기본 아이템 데이터 완전 삭제 완료")
            
    except Exception as e:
        print(f"⚠️ 기존 데이터 삭제 중 오류: {e}")
        pipeline.db.conn.rollback()
    
    # 각 이미지에 대해 AI 분석 수행
    processed_count = 0
    for image_file in image_files:
        try:
            print(f"\n📸 기본 아이템 분석: {image_file.name}")
            
            # AI 파이프라인으로 분석
            result = pipeline.process_image(
                str(image_file), 
                user_id=0,  # 기본 아이템은 user_id=0
                save_separated_images=True
            )
            
            if result['success']:
                # 기본 아이템으로 마킹
                with pipeline.db.conn.cursor() as cur:
                    cur.execute("""
                        UPDATE wardrobe_items 
                        SET is_default = TRUE 
                        WHERE item_id = %s
                    """, (result['item_id'],))
                    pipeline.db.conn.commit()
                
                processed_count += 1
                print(f"✅ 기본 아이템 분석 완료: {image_file.name} (ID: {result['item_id']})")
            else:
                print(f"❌ 기본 아이템 분석 실패: {image_file.name} - {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ 기본 아이템 처리 중 오류: {image_file.name} - {e}")
            continue
    
    print(f"\n🎉 기본 아이템 AI 분석 완료: {processed_count}/{len(image_files)}개 성공")
    return processed_count


@router.delete("/default-items")
def delete_all_default_items():
    """모든 기본 아이템 삭제 API"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="AI 파이프라인이 비활성화되어 있습니다.")
        
        with pipeline.db.conn.cursor() as cur:
            # 삭제할 아이템 수 확인
            cur.execute("SELECT COUNT(*) FROM wardrobe_items WHERE is_default = TRUE")
            count = cur.fetchone()[0]
            
            if count == 0:
                return {
                    "success": True,
                    "message": "삭제할 기본 아이템이 없습니다.",
                    "deleted_count": 0
                }
            
            # 기본 아이템 속성 테이블들 먼저 삭제
            cur.execute("DELETE FROM top_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE is_default = TRUE)")
            cur.execute("DELETE FROM bottom_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE is_default = TRUE)")
            cur.execute("DELETE FROM outer_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE is_default = TRUE)")
            cur.execute("DELETE FROM dress_attributes_new WHERE item_id IN (SELECT item_id FROM wardrobe_items WHERE is_default = TRUE)")
            
            # 기본 아이템 메인 테이블 삭제
            cur.execute("DELETE FROM wardrobe_items WHERE is_default = TRUE")
            
            pipeline.db.conn.commit()
            
        return {
            "success": True,
            "message": f"기본 아이템 {count}개 삭제 완료",
            "deleted_count": count
        }
    except Exception as e:
        print(f"❌ 기본 아이템 삭제 API 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"기본 아이템 삭제 오류: {str(e)}"
        )


@router.post("/process-default-items")
def process_default_items_api():
    """기본 아이템 AI 분석 API (수동 실행)"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="AI 파이프라인이 비활성화되어 있습니다.")
        
        processed_count = process_default_items_internal()
        
        return {
            "success": True,
            "message": f"기본 아이템 AI 분석 완료: {processed_count}개 처리됨",
            "processed_count": processed_count
        }
    except Exception as e:
        print(f"❌ 기본 아이템 처리 API 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"기본 아이템 처리 오류: {str(e)}"
        )


@router.post("/fix-default-items-images")
def fix_default_items_images():
    """기본 아이템 이미지 경로 수정 API"""
    try:
        if not pipeline:
            raise HTTPException(status_code=503, detail="AI 파이프라인이 비활성화되어 있습니다.")
        
        with pipeline.db.conn.cursor() as cur:
            # 기본 아이템들의 이미지 경로 업데이트
            cur.execute("""
                UPDATE wardrobe_items 
                SET 
                    saved_full_image = 'processed_images/user_0/full/item_' || item_id || '_full.jpg',
                    saved_top_image = CASE WHEN has_top THEN 'processed_images/user_0/top/item_' || item_id || '_top.jpg' ELSE NULL END,
                    saved_bottom_image = CASE WHEN has_bottom THEN 'processed_images/user_0/bottom/item_' || item_id || '_bottom.jpg' ELSE NULL END,
                    saved_outer_image = CASE WHEN has_outer THEN 'processed_images/user_0/outer/item_' || item_id || '_outer.jpg' ELSE NULL END,
                    saved_dress_image = CASE WHEN has_dress THEN 'processed_images/user_0/dress/item_' || item_id || '_dress.jpg' ELSE NULL END
                WHERE is_default = TRUE
            """)
            
            updated_count = cur.rowcount
            pipeline.db.conn.commit()
            
        return {
            "success": True,
            "message": f"기본 아이템 이미지 경로 {updated_count}개 수정 완료",
            "updated_count": updated_count
        }
    except Exception as e:
        print(f"❌ 이미지 경로 수정 API 오류: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"이미지 경로 수정 오류: {str(e)}"
        )


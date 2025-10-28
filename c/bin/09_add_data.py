import random
from pathlib import Path
from tqdm import tqdm
import time
from _main_pipeline import FashionPipeline

# 설정
BASE_PATH = Path('D:/D_Study/kkokkaot/k_fashion_data/원천데이터')
TARGET_IMAGES = 300
USER_ID = 1

# 23개 스타일 폴더
STYLE_FOLDERS = [
    '레트로', '로맨틱', '리조트', '매니시', '모던', '밀리터리', 
    '섹시', '소피스트케이티드', '스트리트', '스포티', '아방가르드', 
    '오리엔탈', '웨스턴', '젠더리스', '컨트리', '클래식', '키치', 
    '톰보이', '펑크', '페미닌', '프레피', '히피', '힙합'
]


def collect_images_from_styles(base_path: Path, 
                                target_count: int = 300) -> list:
    """각 스타일 폴더에서 랜덤 이미지 수집"""
    
    print(f"Collecting images from {base_path}")
    print(f"Target: {target_count} images")
    
    all_images = []
    per_style = target_count // len(STYLE_FOLDERS)
    extra = target_count % len(STYLE_FOLDERS)
    
    print(f"\nPer style: ~{per_style} images")
    
    for i, style in enumerate(STYLE_FOLDERS):
        style_path = base_path / style
        
        if not style_path.exists():
            print(f"  [SKIP] {style} - folder not found")
            continue
        
        # 해당 스타일의 모든 jpg 파일
        jpg_files = list(style_path.glob('*.jpg'))
        
        if not jpg_files:
            print(f"  [SKIP] {style} - no images")
            continue
        
        # 이번 스타일에서 가져올 개수
        count = per_style + (1 if i < extra else 0)
        count = min(count, len(jpg_files))
        
        # 랜덤 샘플링
        selected = random.sample(jpg_files, count)
        all_images.extend(selected)
        
        print(f"  [OK] {style}: {count}/{len(jpg_files)} images")
    
    print(f"\nTotal collected: {len(all_images)} images")
    return all_images


def batch_process_images(image_paths: list, user_id: int = 1):
    """배치 이미지 처리"""
    
    print(f"\n{'='*60}")
    print(f"Batch Processing Started")
    print(f"Total images: {len(image_paths)}")
    print(f"User ID: {user_id}")
    print(f"{'='*60}\n")
    
    # 파이프라인 초기화
    pipeline = FashionPipeline(
        yolo_pose_path="D:/D_Study/kkokkaot/yolo11n-pose.pt",
        attribute_model_path="D:/D_Study/kkokkaot/fashion_attribute_model.pth",
        chroma_path="./chroma_db",
        db_config={
            'host': 'localhost',
            'port': 5432,
            'database': 'kkokkaot_closet',
            'user': 'postgres',
            'password': '000000'
        }
    )
    
    # 결과 추적
    success_count = 0
    fail_count = 0
    fail_list = []
    
    start_time = time.time()
    
    try:
        for img_path in tqdm(image_paths, desc="Processing"):
            try:
                result = pipeline.process_image(
                    image_path=str(img_path),
                    user_id=user_id,
                    save_separated_images=False  # 속도를 위해 False
                )
                
                if result['success']:
                    success_count += 1
                else:
                    fail_count += 1
                    fail_list.append((str(img_path), result.get('error', 'Unknown')))
                
            except Exception as e:
                fail_count += 1
                fail_list.append((str(img_path), str(e)))
                continue
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"Batch Processing Completed")
        print(f"{'='*60}")
        print(f"Success: {success_count}/{len(image_paths)}")
        print(f"Failed: {fail_count}/{len(image_paths)}")
        print(f"Time: {elapsed_time:.1f}s ({elapsed_time/len(image_paths):.2f}s per image)")
        
        if fail_list:
            print(f"\n=== Failed Images ===")
            for path, error in fail_list[:10]:  # 처음 10개만
                print(f"  {Path(path).name}: {error[:50]}")
            
            if len(fail_list) > 10:
                print(f"  ... and {len(fail_list)-10} more")
        
        # 최종 통계
        print(f"\n=== Final Statistics ===")
        with pipeline.db_conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM wardrobe_items WHERE user_id = %s", (user_id,))
            total_items = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM wardrobe_items WHERE user_id = %s AND has_top = true", (user_id,))
            top_items = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM wardrobe_items WHERE user_id = %s AND has_bottom = true", (user_id,))
            bottom_items = cur.fetchone()[0]
            
            print(f"Total items in DB: {total_items}")
            print(f"  - With top: {top_items}")
            print(f"  - With bottom: {bottom_items}")
        
    finally:
        pipeline.close()
    
    return success_count, fail_count, fail_list


def main():
    """메인 실행 함수"""
    
    print("\n" + "="*60)
    print("Batch Image Processing for Fashion Wardrobe")
    print("="*60 + "\n")
    
    # 1. 이미지 수집
    image_paths = collect_images_from_styles(
        base_path=BASE_PATH,
        target_count=TARGET_IMAGES
    )
    
    if not image_paths:
        print("\nNo images found. Check the base path.")
        return
    
    # 2. 진행 확인
    print(f"\nReady to process {len(image_paths)} images.")
    response = input("Continue? (y/n): ")
    
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # 3. 배치 처리
    success, fail, fail_list = batch_process_images(image_paths, USER_ID)
    
    # 4. 실패 목록 저장
    if fail_list:
        with open('failed_images.txt', 'w', encoding='utf-8') as f:
            for path, error in fail_list:
                f.write(f"{path}\t{error}\n")
        print(f"\nFailed images saved to: failed_images.txt")
    
    print("\nAll done!")


if __name__ == '__main__':
    # 재현성을 위한 시드
    random.seed(42)
    
    # 실행
    main()
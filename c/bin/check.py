from pathlib import Path
from tqdm import tqdm

def remove_unlabeled_images_fast(
    image_dir: str = "converted_data/all_images",
    cnn_dir: str = "converted_data/cnn"
):
    """라벨링 없는 이미지 빠르게 삭제 (파일명 기반)"""
    
    print("📋 라벨링 파일 목록 생성 중...")
    
    # CNN JSON 파일명에서 image_id 추출 (파일 열지 않음!)
    # cnn_22614.json -> 22614.jpg
    labeled_ids = set()
    for json_file in Path(cnn_dir).glob("cnn_*.json"):
        # "cnn_22614.json" -> "22614"
        image_id = json_file.stem.replace("cnn_", "")
        labeled_ids.add(f"{image_id}.jpg")
    
    print(f"✅ 라벨링 파일: {len(labeled_ids):,}개\n")
    
    # 이미지 파일 확인 및 삭제
    image_files = list(Path(image_dir).glob("*.jpg"))
    print(f"📁 전체 이미지: {len(image_files):,}개\n")
    
    removed = 0
    kept = 0
    
    for img_file in tqdm(image_files, desc="삭제 중"):
        if img_file.name not in labeled_ids:
            img_file.unlink()  # 삭제
            removed += 1
        else:
            kept += 1
    
    print(f"\n{'='*60}")
    print(f"✅ 유지: {kept:,}개")
    print(f"🗑️ 삭제: {removed:,}개")
    print(f"{'='*60}")

# 실행
remove_unlabeled_images_fast()
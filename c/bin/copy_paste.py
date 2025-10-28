import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def copy_all_images_to_one_folder(
    source_dir: str = "D:/D_Study/kkokkaot/k_fashion_data/원천데이터",
    dest_dir: str = "D:/D_Study/kkokkaot/k_fashion_data/all_images",
    max_workers: int = 8  # 병렬 처리 스레드 수
):
    """
    모든 하위 폴더의 이미지를 한 폴더로 복사 (고속 처리)
    
    Args:
        source_dir: 원천데이터 폴더 경로
        dest_dir: 이미지를 모을 폴더 경로
        max_workers: 병렬 처리할 스레드 수 (높을수록 빠름)
    """
    
    # 대상 폴더 생성
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # 모든 이미지 파일 찾기
    source_path = Path(source_dir)
    image_files = []
    
    print("📁 이미지 파일 검색 중...")
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(source_path.rglob(ext))
    
    print(f"✅ 총 {len(image_files):,}개 이미지 파일 발견\n")
    
    # 복사 함수
    def copy_file(src_file):
        try:
            dest_file = dest_path / src_file.name
            
            # 파일명 중복 시 처리
            if dest_file.exists():
                # 이미 같은 파일이면 스킵
                if dest_file.stat().st_size == src_file.stat().st_size:
                    return f"⏭️  스킵: {src_file.name}"
                
                # 다른 파일이면 번호 붙이기
                counter = 1
                while dest_file.exists():
                    stem = src_file.stem
                    suffix = src_file.suffix
                    dest_file = dest_path / f"{stem}_{counter}{suffix}"
                    counter += 1
            
            shutil.copy2(src_file, dest_file)
            return f"✅ 복사: {src_file.name}"
        
        except Exception as e:
            return f"❌ 실패: {src_file.name} - {e}"
    
    # 병렬 복사 실행
    print(f"🚀 복사 시작 (병렬 처리: {max_workers} 스레드)\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(copy_file, image_files),
            total=len(image_files),
            desc="복사 중",
            unit="파일"
        ))
    
    # 결과 요약
    success = sum(1 for r in results if r.startswith("✅"))
    skip = sum(1 for r in results if r.startswith("⏭️"))
    fail = sum(1 for r in results if r.startswith("❌"))
    
    print(f"\n{'='*60}")
    print(f"✅ 복사 완료: {success:,}개")
    print(f"⏭️  스킵: {skip:,}개")
    print(f"❌ 실패: {fail:,}개")
    print(f"📂 저장 위치: {dest_dir}")
    print(f"{'='*60}\n")


# ========================================
# 실행
# ========================================
if __name__ == "__main__":
    copy_all_images_to_one_folder(
        source_dir="D:/D_Study/kkokkaot/k_fashion_data/원천데이터",
        dest_dir="D:/D_Study/kkokkaot/converted_data/all_images",
        max_workers=16  # 💡 컴퓨터 성능에 따라 조절 (4~16)
    )
# ```

# **실행하면:**
# ```
# 📁 이미지 파일 검색 중...
# ✅ 총 50,000개 이미지 파일 발견

# 🚀 복사 시작 (병렬 처리: 16 스레드)

# 복사 중: 100%|██████████| 50000/50000 [00:45<00:00, 1111.23파일/s]

# ============================================================
# ✅ 복사 완료: 49,995개
# ⏭️  스킵: 5개
# ❌ 실패: 0개
# 📂 저장 위치: D:/D_Study/kkokkaot/k_fashion_data/all_images
# ============================================================
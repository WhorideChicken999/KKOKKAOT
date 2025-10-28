import os
import time
from pathlib import Path

def process_default_items_in_batches(batch_size=1000):
    """기본 아이템을 배치별로 처리"""
    
    default_items_dir = Path("./default_items")
    if not default_items_dir.exists():
        print("❌ default_items 폴더가 없습니다.")
        return
    
    # 모든 이미지 파일 찾기
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(default_items_dir.glob(ext))
    
    total_files = len(image_files)
    print(f"📁 총 {total_files}개의 이미지 파일 발견")
    
    if total_files == 0:
        print("❌ 처리할 이미지가 없습니다.")
        return
    
    # 배치별로 처리
    batches = [image_files[i:i+batch_size] for i in range(0, total_files, batch_size)]
    
    print(f"🔄 {len(batches)}개 배치로 나누어 처리합니다.")
    print(f"   - 배치 크기: {batch_size}개")
    print(f"   - 예상 시간: {len(batches) * 30}분 (배치당 30분 가정)")
    
    for i, batch in enumerate(batches, 1):
        print(f"\n{'='*60}")
        print(f"🔄 배치 {i}/{len(batches)} 처리 중... ({len(batch)}개 파일)")
        print(f"{'='*60}")
        
        # 배치 폴더 생성
        batch_dir = Path(f"./default_items_batch_{i}")
        batch_dir.mkdir(exist_ok=True)
        
        # 파일들을 배치 폴더로 이동
        for img_file in batch:
            new_path = batch_dir / img_file.name
            img_file.rename(new_path)
        
        print(f"✅ 배치 {i} 준비 완료: {batch_dir}")
        print(f"💡 이제 다음 명령어를 실행하세요:")
        print(f"   POST http://127.0.0.1:4000/api/process-default-items")
        print(f"   (배치 {i} 완료 후 Enter 키를 누르세요)")
        
        input("배치 처리가 완료되면 Enter를 누르세요...")
        
        # 원래 위치로 파일들 복원
        for img_file in batch_dir.glob("*"):
            original_path = default_items_dir / img_file.name
            img_file.rename(original_path)
        
        batch_dir.rmdir()  # 빈 배치 폴더 삭제
        print(f"✅ 배치 {i} 완료!")

if __name__ == "__main__":
    print("🎯 기본 아이템 배치 처리 시작")
    print("⚠️ 10만개는 매우 큰 규모입니다!")
    print("💡 1000개씩 나누어 처리하는 것을 권장합니다.")
    
    batch_size = int(input("배치 크기를 입력하세요 (기본값: 1000): ") or "1000")
    process_default_items_in_batches(batch_size)

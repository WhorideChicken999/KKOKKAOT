import os
import random
import shutil
from pathlib import Path

def select_random_images(source_dir, target_dir, count=150):
    """소스 디렉토리에서 랜덤하게 이미지를 선택하여 타겟 디렉토리로 복사"""
    
    # 경로 설정
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 타겟 디렉토리 생성
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 소스 디렉토리에서 모든 이미지 파일 찾기
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    all_images = []
    
    for ext in image_extensions:
        all_images.extend(source_path.glob(f"*{ext}"))
    
    print(f"소스 디렉토리에서 {len(all_images)}개의 이미지 파일 발견")
    
    if len(all_images) < count:
        print(f"요청한 개수({count})보다 적은 이미지({len(all_images)})가 있습니다.")
        print(f"사용 가능한 모든 이미지({len(all_images)})를 복사합니다.")
        count = len(all_images)
    
    # 랜덤하게 선택
    selected_images = random.sample(all_images, count)
    
    print(f"{count}개의 이미지를 랜덤하게 선택했습니다.")
    
    # 복사 시작
    copied_count = 0
    for i, image_path in enumerate(selected_images, 1):
        try:
            # 타겟 경로 설정
            target_file = target_path / image_path.name
            
            # 파일명 중복 처리
            counter = 1
            original_name = target_file.stem
            original_ext = target_file.suffix
            
            while target_file.exists():
                target_file = target_path / f"{original_name}_{counter}{original_ext}"
                counter += 1
            
            # 파일 복사
            shutil.copy2(image_path, target_file)
            copied_count += 1
            
            if i % 10 == 0 or i == count:
                print(f"진행률: {i}/{count} ({i/count*100:.1f}%) - {image_path.name}")
                
        except Exception as e:
            print(f"복사 실패: {image_path.name} - {e}")
    
    print(f"\n완료! {copied_count}개의 이미지가 복사되었습니다.")
    print(f"복사된 위치: {target_path}")
    
    return copied_count

if __name__ == "__main__":
    # 설정
    source_directory = r"D:\kkokkaot\converted_data\all_images"
    target_directory = r"D:\kkokkaot\default_items"
    image_count = 150
    
    print("랜덤 이미지 선택 시작")
    print(f"소스: {source_directory}")
    print(f"타겟: {target_directory}")
    print(f"개수: {image_count}개")
    print("-" * 50)
    
    # 실행
    copied = select_random_images(source_directory, target_directory, image_count)
    
    print(f"\n작업 완료! {copied}개의 이미지가 default_items 폴더에 저장되었습니다.")
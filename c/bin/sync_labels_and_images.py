#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN 라벨과 크롭 이미지 동기화 스크립트
크롭된 이미지가 없는 CNN 라벨 파일을 제거하여 수를 맞춥니다.
"""

import os
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class LabelImageSynchronizer:
    """라벨-이미지 동기화 클래스"""
    
    def __init__(self, data_dir="D:/converted_data/prepared_data"):
        self.data_dir = Path(data_dir)
        
        self.cnn_labels_dir = self.data_dir / "cnn_labels"
        self.cropped_images_dir = self.data_dir / "cropped_images"
        
        print(f"📁 데이터 디렉토리: {self.data_dir}")
        print(f"📁 CNN 라벨 디렉토리: {self.cnn_labels_dir}")
        print(f"📁 크롭 이미지 디렉토리: {self.cropped_images_dir}")
    
    def collect_cropped_image_ids(self, category):
        """크롭된 이미지 ID 수집"""
        cropped_images_dir = self.cropped_images_dir / category
        cropped_image_ids = set()
        
        if cropped_images_dir.exists():
            for img_file in cropped_images_dir.glob("*.jpg"):
                # 파일명 형식: imageID_category_sequence.jpg
                filename = img_file.stem
                parts = filename.split('_')
                if len(parts) >= 2:
                    image_id = parts[0]
                    cropped_image_ids.add(image_id)
        
        return cropped_image_ids
    
    def sync_category(self, category):
        """특정 카테고리 동기화"""
        print(f"\n🔄 {category} 동기화 중...")
        
        # 크롭된 이미지 ID 수집
        cropped_image_ids = self.collect_cropped_image_ids(category)
        print(f"  크롭된 이미지: {len(cropped_image_ids)}개 고유 ID")
        
        # CNN 라벨 파일 확인
        cnn_labels_dir = self.cnn_labels_dir / category
        
        if not cnn_labels_dir.exists():
            print(f"  ⚠️ CNN 라벨 디렉토리가 없습니다: {cnn_labels_dir}")
            return 0, 0, 0
        
        label_files = list(cnn_labels_dir.glob("*.json"))
        print(f"  CNN 라벨 파일: {len(label_files)}개")
        
        removed_count = 0
        kept_count = 0
        error_count = 0
        
        for label_file in tqdm(label_files, desc=f"  {category} 처리 중"):
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                image_id = str(data.get('image_id'))
                
                # 크롭된 이미지가 없으면 라벨 파일 삭제
                if image_id not in cropped_image_ids:
                    label_file.unlink()
                    removed_count += 1
                else:
                    kept_count += 1
                    
            except Exception as e:
                print(f"\n  ⚠️ 라벨 파일 처리 오류 {label_file}: {e}")
                error_count += 1
                continue
        
        print(f"  ✅ 완료: 유지 {kept_count}개, 제거 {removed_count}개, 오류 {error_count}개")
        
        return kept_count, removed_count, error_count
    
    def verify_sync(self, category):
        """동기화 검증"""
        # CNN 라벨 개수
        cnn_labels_dir = self.cnn_labels_dir / category
        cnn_count = len(list(cnn_labels_dir.glob("*.json"))) if cnn_labels_dir.exists() else 0
        
        # 크롭 이미지 개수 (고유 ID 기준)
        cropped_image_ids = self.collect_cropped_image_ids(category)
        img_count = len(cropped_image_ids)
        
        return cnn_count, img_count
    
    def run_sync(self):
        """전체 동기화 실행"""
        print("🚀 CNN 라벨과 크롭 이미지 동기화 시작!")
        print("=" * 60)
        
        categories = ['상의', '하의', '아우터', '원피스']
        
        # 동기화 전 상태
        print("\n📊 동기화 전 상태:")
        print("-" * 40)
        before_stats = {}
        for category in categories:
            cnn_count, img_count = self.verify_sync(category)
            before_stats[category] = {'cnn': cnn_count, 'img': img_count}
            diff = cnn_count - img_count
            print(f"{category:8s}: CNN 라벨 {cnn_count:5d}개, 크롭 이미지 {img_count:5d}개 (차이: {diff:+5d})")
        
        # 동기화 실행
        total_kept = 0
        total_removed = 0
        total_errors = 0
        
        for category in categories:
            kept, removed, errors = self.sync_category(category)
            total_kept += kept
            total_removed += removed
            total_errors += errors
        
        # 동기화 후 상태
        print("\n📊 동기화 후 상태:")
        print("-" * 40)
        after_stats = {}
        all_matched = True
        
        for category in categories:
            cnn_count, img_count = self.verify_sync(category)
            after_stats[category] = {'cnn': cnn_count, 'img': img_count}
            match_status = "✅" if cnn_count == img_count else "❌"
            
            if cnn_count != img_count:
                all_matched = False
            
            before_cnn = before_stats[category]['cnn']
            removed = before_cnn - cnn_count
            
            print(f"{category:8s}: CNN 라벨 {cnn_count:5d}개, 크롭 이미지 {img_count:5d}개 {match_status} (제거: {removed}개)")
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("📊 동기화 완료!")
        print("=" * 60)
        print(f"총 유지된 라벨: {total_kept:,}개")
        print(f"총 제거된 라벨: {total_removed:,}개")
        print(f"총 오류: {total_errors}개")
        
        if all_matched:
            print("\n✅ 모든 카테고리의 CNN 라벨과 크롭 이미지 수가 일치합니다!")
        else:
            print("\n⚠️ 일부 카테고리에서 수가 일치하지 않습니다.")
            print("   (한 이미지에 여러 개의 동일 카테고리 객체가 있을 수 있습니다)")
        
        print("\n🎉 동기화 작업이 완료되었습니다!")

def main():
    """메인 실행 함수"""
    print("🎯 CNN 라벨과 크롭 이미지 동기화 도구")
    print("크롭된 이미지가 없는 CNN 라벨 파일을 제거합니다.")
    print("=" * 60)
    
    # 사용자 확인
    response = input("\n동기화를 진행하시겠습니까? (y/N): ")
    if response.lower() != 'y':
        print("❌ 동기화가 취소되었습니다.")
        return
    
    # 동기화 실행
    synchronizer = LabelImageSynchronizer()
    synchronizer.run_sync()

if __name__ == "__main__":
    main()

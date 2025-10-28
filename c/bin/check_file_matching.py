#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN 라벨과 크롭된 이미지 파일 매칭 상태 확인 스크립트
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

def check_file_matching():
    """CNN 라벨과 크롭된 이미지 파일 매칭 상태 확인"""
    
    # 경로 설정
    cnn_labels_dir = Path("D:/converted_data/prepared_data/cnn_labels")
    cropped_images_dir = Path("D:/converted_data/prepared_data/cropped_images")
    
    categories = ['상의', '하의', '아우터', '원피스']
    
    print("🔍 CNN 라벨과 크롭된 이미지 파일 매칭 상태 확인")
    print("=" * 60)
    
    # 전체 결과 저장용
    all_results = {}
    
    for category in categories:
        print(f"\n📂 {category} 카테고리 확인 중...")
        
        category_cnn_dir = cnn_labels_dir / category
        category_images_dir = cropped_images_dir / category
        
        # 디렉토리 존재 확인
        if not category_cnn_dir.exists():
            print(f"  ❌ CNN 라벨 디렉토리가 존재하지 않습니다: {category_cnn_dir}")
            continue
            
        if not category_images_dir.exists():
            print(f"  ❌ 크롭 이미지 디렉토리가 존재하지 않습니다: {category_images_dir}")
            continue
        
        # 파일 목록 수집
        cnn_files = set()
        image_files = set()
        
        # CNN 라벨 파일들 (확장자 제거)
        for cnn_file in category_cnn_dir.glob("*.json"):
            cnn_files.add(cnn_file.stem)
        
        # 크롭 이미지 파일들 (확장자 제거)
        for img_file in category_images_dir.glob("*.jpg"):
            image_files.add(img_file.stem)
        
        # 매칭 분석
        matched_files = cnn_files & image_files  # 교집합
        cnn_only = cnn_files - image_files       # CNN에만 있는 파일
        image_only = image_files - cnn_files     # 이미지에만 있는 파일
        
        # 결과 출력
        print(f"  📊 CNN 라벨 파일: {len(cnn_files):,}개")
        print(f"  📊 크롭 이미지: {len(image_files):,}개")
        print(f"  ✅ 매칭된 파일: {len(matched_files):,}개")
        print(f"  ⚠️  CNN에만 있는 파일: {len(cnn_only):,}개")
        print(f"  ⚠️  이미지에만 있는 파일: {len(image_only):,}개")
        
        # 매칭률 계산
        if len(cnn_files) > 0:
            match_rate = (len(matched_files) / len(cnn_files)) * 100
            print(f"  📈 매칭률: {match_rate:.2f}%")
        
        # 결과 저장
        all_results[category] = {
            'cnn_files': len(cnn_files),
            'image_files': len(image_files),
            'matched_files': len(matched_files),
            'cnn_only': len(cnn_only),
            'image_only': len(image_only),
            'match_rate': match_rate if len(cnn_files) > 0 else 0,
            'cnn_only_list': list(cnn_only)[:10],  # 처음 10개만 저장
            'image_only_list': list(image_only)[:10]  # 처음 10개만 저장
        }
        
        # 매칭되지 않은 파일들 샘플 출력
        if cnn_only:
            print(f"  📝 CNN에만 있는 파일 샘플 (처음 5개):")
            for i, file in enumerate(list(cnn_only)[:5]):
                print(f"    - {file}.json")
        
        if image_only:
            print(f"  📝 이미지에만 있는 파일 샘플 (처음 5개):")
            for i, file in enumerate(list(image_only)[:5]):
                print(f"    - {file}.jpg")
    
    # 전체 요약
    print(f"\n{'='*60}")
    print("📊 전체 요약")
    print(f"{'='*60}")
    
    total_cnn = sum(result['cnn_files'] for result in all_results.values())
    total_images = sum(result['image_files'] for result in all_results.values())
    total_matched = sum(result['matched_files'] for result in all_results.values())
    
    print(f"전체 CNN 라벨 파일: {total_cnn:,}개")
    print(f"전체 크롭 이미지: {total_images:,}개")
    print(f"전체 매칭된 파일: {total_matched:,}개")
    print(f"전체 매칭률: {(total_matched/total_cnn*100):.2f}%" if total_cnn > 0 else "N/A")
    
    # 카테고리별 상세 요약
    print(f"\n📋 카테고리별 상세:")
    print(f"{'카테고리':8s} {'CNN':8s} {'이미지':8s} {'매칭':8s} {'매칭률':8s}")
    print("-" * 50)
    
    for category, result in all_results.items():
        print(f"{category:8s} {result['cnn_files']:6d}개 {result['image_files']:6d}개 {result['matched_files']:6d}개 {result['match_rate']:6.1f}%")
    
    # 결과를 JSON 파일로 저장
    output_file = Path("D:/converted_data/prepared_data/file_matching_report.json")
    
    # 상세 결과 (모든 파일 목록 포함)
    detailed_results = {}
    for category in categories:
        category_cnn_dir = cnn_labels_dir / category
        category_images_dir = cropped_images_dir / category
        
        if category_cnn_dir.exists() and category_images_dir.exists():
            cnn_files = {f.stem for f in category_cnn_dir.glob("*.json")}
            image_files = {f.stem for f in category_images_dir.glob("*.jpg")}
            
            detailed_results[category] = {
                'summary': all_results[category],
                'cnn_only_files': list(cnn_files - image_files),
                'image_only_files': list(image_files - cnn_files),
                'matched_files': list(cnn_files & image_files)
            }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 상세 결과 저장: {output_file}")
    
    return all_results

if __name__ == "__main__":
    results = check_file_matching()

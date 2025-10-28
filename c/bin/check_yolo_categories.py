#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from collections import defaultdict, Counter

def check_yolo_categories():
    """YOLO JSON 파일에서 같은 카테고리가 여러개 있는지 확인"""
    
    yolo_dir = Path("d:/converted_data/yolo")
    
    if not yolo_dir.exists():
        print("YOLO 디렉토리가 없습니다.")
        return
    
    # 카테고리별 객체 개수 통계
    category_counts = defaultdict(list)  # {category: [count1, count2, ...]}
    multi_category_files = []  # 같은 카테고리가 여러개인 파일들
    
    print("YOLO JSON 파일 분석 중...")
    
    yolo_files = list(yolo_dir.glob("*.json"))
    print(f"총 YOLO 파일: {len(yolo_files)}개")
    
    for yolo_file in yolo_files:
        try:
            with open(yolo_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 카테고리별 객체 개수 세기
            category_counter = Counter()
            
            for annotation in data.get('annotations', []):
                category_id = annotation['category_id']
                
                # category_id를 이름으로 변환
                category_name = None
                for cat in data.get('categories', []):
                    if cat['id'] == category_id:
                        category_name = cat['name']
                        break
                
                if category_name:
                    category_counter[category_name] += 1
            
            # 결과 저장
            for category, count in category_counter.items():
                category_counts[category].append(count)
            
            # 같은 카테고리가 여러개인 파일 체크
            if any(count > 1 for count in category_counter.values()):
                multi_category_files.append({
                    'file': yolo_file.name,
                    'categories': dict(category_counter)
                })
                
        except Exception as e:
            print(f"파일 처리 오류 {yolo_file}: {e}")
            continue
    
    # 결과 출력
    print(f"\n카테고리별 객체 개수 통계:")
    print("-" * 50)
    
    for category, counts in category_counts.items():
        total_objects = sum(counts)
        max_count = max(counts)
        avg_count = sum(counts) / len(counts)
        
        print(f"{category:8s}: 총 {total_objects:6d}개 객체, 최대 {max_count}개, 평균 {avg_count:.2f}개")
        
        # 개수별 분포
        count_distribution = Counter(counts)
        print(f"         분포: {dict(count_distribution)}")
    
    print(f"\n같은 카테고리가 여러개인 파일들:")
    print("-" * 50)
    
    if multi_category_files:
        print(f"총 {len(multi_category_files)}개 파일에서 같은 카테고리가 여러개 발견:")
        
        for item in multi_category_files[:10]:  # 처음 10개만 출력
            print(f"  {item['file']}: {item['categories']}")
        
        if len(multi_category_files) > 10:
            print(f"  ... 그리고 {len(multi_category_files) - 10}개 더")
    else:
        print("같은 카테고리가 여러개인 파일은 없습니다.")
    
    # 카테고리 매핑 확인
    print(f"\n카테고리 매핑:")
    print("-" * 30)
    print("YOLO -> CNN")
    print("outer -> 아우터")
    print("top -> 상의") 
    print("bottom -> 하의")
    print("dress -> 원피스")

if __name__ == "__main__":
    check_yolo_categories()

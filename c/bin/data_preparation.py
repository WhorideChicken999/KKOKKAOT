#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 준비 스크립트
1. CNN 라벨을 카테고리별로 분리하여 저장
2. YOLO 바운딩 박스 정보를 사용하여 이미지를 잘라서 저장
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class DataPreparation:
    """데이터 준비 클래스"""
    
    def __init__(self, data_dir="D:/converted_data", output_dir="D:/converted_data/prepared_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        
        # 입력 경로
        self.cnn_dir = self.data_dir / "cnn"
        self.yolo_dir = self.data_dir / "yolo"
        self.images_dir = self.data_dir / "all_images"
        
        # 출력 경로
        self.output_dir.mkdir(exist_ok=True)
        
        # 로그 파일 설정
        self.log_file = self.output_dir / f"data_preparation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.log_entries = []
        
        print(f"📁 입력 데이터 디렉토리: {self.data_dir}")
        print(f"📁 출력 데이터 디렉토리: {self.output_dir}")
        print(f"📁 CNN 라벨 디렉토리: {self.cnn_dir}")
        print(f"📁 YOLO 라벨 디렉토리: {self.yolo_dir}")
        print(f"📁 이미지 디렉토리: {self.images_dir}")
        print(f"📄 로그 파일: {self.log_file}")
    
    def log_entry(self, level, message, details=None, pbar=None):
        """로그 엔트리 추가"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message,
            'details': details
        }
        self.log_entries.append(log_entry)
        
        # WARNING과 ERROR만 콘솔에 출력 (INFO는 로그 파일에만 저장)
        if level in ['WARNING', 'ERROR']:
            if pbar:
                pbar.write(f"[{timestamp}] {level}: {message}")
                if details:
                    pbar.write(f"    상세: {details}")
            else:
                print(f"[{timestamp}] {level}: {message}")
                if details:
                    print(f"    상세: {details}")
    
    def save_logs(self):
        """로그를 파일로 저장"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("데이터 준비 로그\n")
            f.write(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # 로그 요약 생성
            log_summary = self.generate_log_summary()
            f.write("📊 로그 요약\n")
            f.write("-" * 40 + "\n")
            for level, count in log_summary.items():
                f.write(f"{level}: {count}개\n")
            f.write("\n" + "=" * 80 + "\n\n")
            
            # 상세 로그
            f.write("📝 상세 로그\n")
            f.write("-" * 40 + "\n")
            for entry in self.log_entries:
                f.write(f"[{entry['timestamp']}] {entry['level']}: {entry['message']}\n")
                if entry['details']:
                    f.write(f"    상세: {entry['details']}\n")
                f.write("\n")
        
        print(f"📄 로그 파일 저장 완료: {self.log_file}")
    
    def generate_log_summary(self):
        """로그 요약 생성"""
        summary = {}
        for entry in self.log_entries:
            level = entry['level']
            summary[level] = summary.get(level, 0) + 1
        return summary
    
    def prepare_cnn_labels(self):
        """CNN 라벨을 카테고리별로 분리하여 저장 (이미지별 독립적 시퀀스)"""
        print("\n🔄 CNN 라벨 분리 시작...")
        
        # 카테고리별 출력 디렉토리 생성
        categories = ['상의', '하의', '아우터', '원피스']
        for category in categories:
            category_dir = self.output_dir / "cnn_labels" / category
            category_dir.mkdir(parents=True, exist_ok=True)
        
        # CNN JSON 파일들 처리
        cnn_files = list(self.cnn_dir.glob("*.json"))
        print(f"📊 처리할 CNN 파일: {len(cnn_files)}개")
        
        category_counts = {cat: 0 for cat in categories}
        
        for cnn_file in tqdm(cnn_files, desc="CNN 라벨 분리 중"):
            try:
                # JSON 파일 로드
                with open(cnn_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 각 이미지별로 카테고리별 시퀀스 번호 초기화
                image_category_counts = {cat: 0 for cat in categories}
                
                # 각 카테고리별로 처리
                items = data.get('items', {})
                
                for category in categories:
                    if category in items and items[category]:
                        # 해당 카테고리에 데이터가 있으면 저장
                        category_data = {
                            'image_id': data['image_id'],
                            'file_name': data['file_name'],
                            'category': category,
                            'item_data': items[category],
                            'style': data.get('style', {})
                        }
                        
                        # 각 이미지별로 독립적인 시퀀스 번호 사용
                        sequence_num = image_category_counts[category] + 1
                        output_filename = f"{data['image_id']}_{category}_{sequence_num:03d}.json"
                        output_path = self.output_dir / "cnn_labels" / category / output_filename
                        
                        # JSON 파일 저장
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(category_data, f, ensure_ascii=False, indent=2)
                        
                        # 각 이미지별 카테고리 카운트 증가
                        image_category_counts[category] += 1
                        # 전체 카테고리 카운트도 증가
                        category_counts[category] += 1
                        
            except Exception as e:
                print(f"⚠️ CNN 파일 처리 오류 {cnn_file}: {e}")
                continue
        
        print("\n✅ CNN 라벨 분리 완료!")
        for category, count in category_counts.items():
            print(f"  {category}: {count}개 파일")
        
        return category_counts
    
    def prepare_cropped_images(self):
        """YOLO 바운딩 박스 정보를 사용하여 이미지를 잘라서 저장"""
        print("\n🔄 이미지 크롭 시작...")
        
        # 카테고리별 출력 디렉토리 생성
        categories = ['상의', '하의', '아우터', '원피스']
        category_mapping = {
            'top': '상의',
            'bottom': '하의', 
            'outer': '아우터',
            'dress': '원피스'
        }
        
        for category in categories:
            category_dir = self.output_dir / "cropped_images" / category
            category_dir.mkdir(parents=True, exist_ok=True)
        
        # YOLO JSON 파일들 처리
        yolo_files = list(self.yolo_dir.glob("*.json"))
        print(f"📊 처리할 YOLO 파일: {len(yolo_files)}개")
        
        category_counts = {cat: 0 for cat in categories}
        processed_images = 0
        
        pbar = tqdm(yolo_files, desc="이미지 크롭 중")
        for yolo_file in pbar:
            try:
                # YOLO JSON 파일 로드
                with open(yolo_file, 'r', encoding='utf-8') as f:
                    yolo_data = json.load(f)
                
                # 이미지 정보 추출
                if not yolo_data.get('images') or not yolo_data.get('annotations'):
                    self.log_entry("WARNING", f"YOLO 파일에 이미지 또는 어노테이션 정보 없음", 
                                 f"파일: {yolo_file.name}", pbar)
                    continue
                
                image_info = yolo_data['images'][0]
                image_id = image_info['id']
                image_width = image_info['width']
                image_height = image_info['height']
                
                # 원본 이미지 파일 경로
                image_path = self.images_dir / f"{image_id}.jpg"
                if not image_path.exists():
                    self.log_entry("ERROR", f"원본 이미지 파일을 찾을 수 없음", 
                                 f"이미지 ID: {image_id}, 경로: {image_path}", pbar)
                    continue
                
                # 원본 이미지 로드
                original_image = Image.open(image_path).convert('RGB')
                
                # 각 이미지별로 카테고리별 시퀀스 번호 초기화
                image_category_counts = {cat: 0 for cat in categories}
                
                # 각 어노테이션(바운딩 박스) 처리
                for annotation in yolo_data['annotations']:
                    category_id = annotation['category_id']
                    bbox = annotation['bbox']  # [x, y, width, height]
                    
                    # 카테고리 매핑
                    category_name = None
                    for cat_id, cat_name in enumerate(['outer', 'top', 'bottom', 'dress']):
                        if category_id == cat_id:
                            category_name = category_mapping[cat_name]
                            break
                    
                    if category_name is None:
                        self.log_entry("WARNING", f"알 수 없는 카테고리 ID", 
                                     f"이미지 ID: {image_id}, 카테고리 ID: {category_id}", pbar)
                        continue
                    
                    # 바운딩 박스 좌표 계산
                    x, y, w, h = bbox
                    
                    # 좌표가 이미지 범위를 벗어나지 않도록 조정
                    x = max(0, min(x, image_width - 1))
                    y = max(0, min(y, image_height - 1))
                    w = max(1, min(w, image_width - x))
                    h = max(1, min(h, image_height - y))
                    
                    # 이미지 크롭
                    cropped_image = original_image.crop((x, y, x + w, y + h))
                    
                    # 크롭된 이미지가 너무 작으면 스킵 (최소 32x32 픽셀)
                    if cropped_image.width < 32 or cropped_image.height < 32:
                        self.log_entry("WARNING", f"크롭된 이미지가 너무 작아서 스킵", 
                                     f"이미지 ID: {image_id}, 카테고리: {category_name}, 크기: {cropped_image.width}x{cropped_image.height}", pbar)
                        continue
                    
                    # 파일명 생성 (이미지ID_카테고리_순번.jpg) - 각 이미지별로 독립적인 시퀀스
                    sequence_num = image_category_counts[category_name] + 1
                    output_filename = f"{image_id}_{category_name}_{sequence_num:03d}.jpg"
                    output_path = self.output_dir / "cropped_images" / category_name / output_filename
                    
                    # 크롭된 이미지 저장
                    cropped_image.save(output_path, 'JPEG', quality=95)
                    
                    # 각 이미지별 카테고리 카운트 증가
                    image_category_counts[category_name] += 1
                    # 전체 카테고리 카운트도 증가
                    category_counts[category_name] += 1
                
                processed_images += 1
                
            except Exception as e:
                print(f"⚠️ YOLO 파일 처리 오류 {yolo_file}: {e}")
                continue
        
        print(f"\n✅ 이미지 크롭 완료! 처리된 이미지: {processed_images}개")
        for category, count in category_counts.items():
            print(f"  {category}: {count}개 크롭 이미지")
        
        return category_counts, processed_images
    
    def create_mapping_file(self, cnn_counts, image_counts, processed_images):
        """매핑 정보 파일 생성"""
        print("\n📄 매핑 정보 파일 생성 중...")
        
        mapping_info = {
            'summary': {
                'total_processed_images': processed_images,
                'cnn_label_counts': cnn_counts,
                'cropped_image_counts': image_counts,
                'categories': ['상의', '하의', '아우터', '원피스']
            },
            'directory_structure': {
                'cnn_labels': {
                    'description': '카테고리별로 분리된 CNN 라벨 파일들',
                    'format': 'imageID_category.json'
                },
                'cropped_images': {
                    'description': 'YOLO 바운딩 박스로 크롭된 이미지들',
                    'format': 'imageID_category_sequence.jpg'
                }
            },
            'usage_notes': [
                '각 카테고리별로 CNN 라벨과 크롭된 이미지가 분리되어 저장됨',
                '하나의 원본 이미지에서 여러 카테고리가 감지되면 각각 저장됨',
                'CNN 라벨 파일과 크롭된 이미지는 imageID로 매칭 가능'
            ]
        }
        
        # JSON 매핑 파일 저장
        mapping_file = self.output_dir / 'data_mapping.json'
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(mapping_info, f, ensure_ascii=False, indent=2)
        
        print(f"💾 매핑 정보 파일 저장: {mapping_file}")
        
        return mapping_file
    
    def sync_labels_with_images(self):
        """크롭된 이미지 기준으로 CNN 라벨 동기화 (시퀀스 번호 추가)"""
        print("\n🔄 CNN 라벨과 크롭 이미지 동기화 중...")
        print("📝 크롭된 이미지 파일명에 맞춰 CNN 라벨을 재생성합니다.")
        
        categories = ['상의', '하의', '아우터', '원피스']
        removed_counts = {cat: 0 for cat in categories}
        created_counts = {cat: 0 for cat in categories}
        
        for category in categories:
            print(f"\n  🔍 {category} 동기화 중...")
            
            # 크롭된 이미지 파일들 수집
            cropped_images_dir = self.output_dir / "cropped_images" / category
            cropped_files = []
            
            if cropped_images_dir.exists():
                cropped_files = list(cropped_images_dir.glob("*.jpg"))
            
            print(f"    크롭 이미지: {len(cropped_files)}개 파일")
            
            # CNN 라벨 디렉토리
            cnn_labels_dir = self.output_dir / "cnn_labels" / category
            
            if not cnn_labels_dir.exists():
                cnn_labels_dir.mkdir(parents=True, exist_ok=True)
            
            # 기존 CNN 라벨 파일들 모두 삭제
            existing_labels = list(cnn_labels_dir.glob("*.json"))
            for label_file in existing_labels:
                label_file.unlink()
                removed_counts[category] += 1
            
            print(f"    기존 라벨 제거: {removed_counts[category]}개")
            
            # 크롭된 이미지에 맞춰 CNN 라벨 재생성
            if len(cropped_files) == 0:
                self.log_entry("WARNING", f"크롭된 이미지가 없어서 기존 CNN 라벨을 시퀀스 번호 없이 유지", 
                             f"카테고리: {category}, 디렉토리: {cropped_images_dir}")
                # 크롭된 이미지가 없으면 기존 CNN 라벨을 시퀀스 번호 없이 재생성
                self._regenerate_cnn_labels_without_sequence(category, cnn_labels_dir, created_counts)
                continue
                
            pbar = tqdm(cropped_files, desc=f"  {category} 라벨 생성 중", leave=False)
            for img_file in pbar:
                try:
                    # 파일명에서 정보 추출: imageID_category_sequence.jpg
                    filename = img_file.stem
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        image_id = parts[0]
                        sequence = parts[2]  # 001, 002 등
                        
                        # 원본 CNN 라벨 데이터 찾기
                        original_cnn_file = self.cnn_dir / f"cnn_{image_id}.json"
                        
                        if original_cnn_file.exists():
                            with open(original_cnn_file, 'r', encoding='utf-8') as f:
                                original_data = json.load(f)
                            
                            # 해당 카테고리 데이터 추출
                            items = original_data.get('items', {})
                            if category in items and items[category]:
                                category_data = {
                                    'image_id': original_data['image_id'],
                                    'file_name': original_data['file_name'],
                                    'category': category,
                                    'item_data': items[category],
                                    'style': original_data.get('style', {})
                                }
                                
                                # 새로운 라벨 파일 생성 (시퀀스 번호 포함) - 3자리로 맞춤
                                new_label_filename = f"{image_id}_{category}_{sequence}.json"
                                new_label_path = cnn_labels_dir / new_label_filename
                                
                                with open(new_label_path, 'w', encoding='utf-8') as f:
                                    json.dump(category_data, f, ensure_ascii=False, indent=2)
                                
                                created_counts[category] += 1
                            else:
                                self.log_entry("WARNING", f"CNN 파일에 해당 카테고리 데이터 없음", 
                                             f"이미지 ID: {image_id}, 카테고리: {category}, CNN 파일: {original_cnn_file.name}", pbar)
                        else:
                            self.log_entry("ERROR", f"원본 CNN 파일을 찾을 수 없음", 
                                         f"이미지 ID: {image_id}, 카테고리: {category}, 경로: {original_cnn_file}", pbar)
                    else:
                        self.log_entry("WARNING", f"크롭된 이미지 파일명 형식 오류", 
                                     f"파일: {img_file.name}, 예상 형식: imageID_category_sequence.jpg", pbar)
                            
                except Exception as e:
                    self.log_entry("ERROR", f"라벨 파일 생성 오류", 
                                 f"파일: {img_file.name}, 오류: {str(e)}", pbar)
                    continue
            
            print(f"    새로 생성된 라벨: {created_counts[category]}개")
        
        print("\n동기화 완료:")
        for category in categories:
            print(f"  {category}: 제거 {removed_counts[category]}개, 생성 {created_counts[category]}개")
        
        return removed_counts, created_counts
    
    def _regenerate_cnn_labels_without_sequence(self, category, cnn_labels_dir, created_counts):
        """크롭된 이미지가 없을 때 시퀀스 번호 없이 CNN 라벨 재생성"""
        print(f"    크롭된 이미지가 없어서 시퀀스 번호 없이 CNN 라벨 재생성 중...")
        
        # 원본 CNN 파일들에서 해당 카테고리 데이터 찾기
        cnn_files = list(self.cnn_dir.glob("*.json"))
        
        for cnn_file in cnn_files:
            try:
                with open(cnn_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 해당 카테고리 데이터가 있는지 확인
                items = data.get('items', {})
                if category in items and items[category]:
                    # 시퀀스 번호 없이 CNN 라벨 생성
                    category_data = {
                        'image_id': data['image_id'],
                        'file_name': data['file_name'],
                        'category': category,
                        'item_data': items[category],
                        'style': data.get('style', {})
                    }
                    
                    # 파일명 생성 (시퀀스 번호 없이)
                    output_filename = f"{data['image_id']}_{category}.json"
                    output_path = cnn_labels_dir / output_filename
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(category_data, f, ensure_ascii=False, indent=2)
                    
                    created_counts[category] += 1
                    
            except Exception as e:
                self.log_entry("ERROR", f"CNN 라벨 재생성 오류", f"파일: {cnn_file.name}, 오류: {str(e)}")
                continue
    
    def count_final_data(self):
        """최종 데이터 개수 확인 (고유 ID 기준)"""
        categories = ['상의', '하의', '아우터', '원피스']
        
        final_cnn_counts = {}
        final_image_counts = {}
        final_unique_image_counts = {}
        
        for category in categories:
            # CNN 라벨 개수
            cnn_labels_dir = self.output_dir / "cnn_labels" / category
            if cnn_labels_dir.exists():
                final_cnn_counts[category] = len(list(cnn_labels_dir.glob("*.json")))
            else:
                final_cnn_counts[category] = 0
            
            # 크롭 이미지 개수 (전체 파일 수)
            cropped_images_dir = self.output_dir / "cropped_images" / category
            if cropped_images_dir.exists():
                final_image_counts[category] = len(list(cropped_images_dir.glob("*.jpg")))
                
                # 고유 image_id 개수
                unique_ids = set()
                for img_file in cropped_images_dir.glob("*.jpg"):
                    filename = img_file.stem
                    parts = filename.split('_')
                    if len(parts) >= 2:
                        image_id = parts[0]
                        unique_ids.add(image_id)
                final_unique_image_counts[category] = len(unique_ids)
            else:
                final_image_counts[category] = 0
                final_unique_image_counts[category] = 0
        
        return final_cnn_counts, final_image_counts, final_unique_image_counts
    
    def run_preparation(self):
        """전체 데이터 준비 프로세스 실행"""
        print("🚀 데이터 준비 시작!")
        print("=" * 60)
        
        try:
            # 1. CNN 라벨 분리
            cnn_counts = self.prepare_cnn_labels()
            
            # 2. 이미지 크롭
            image_counts, processed_images = self.prepare_cropped_images()
            
            # 3. CNN 라벨과 크롭 이미지 동기화
            removed_counts, created_counts = self.sync_labels_with_images()
            
            # 4. 최종 데이터 개수 확인
            final_cnn_counts, final_image_counts, final_unique_image_counts = self.count_final_data()
            
            # 5. 매핑 파일 생성
            mapping_file = self.create_mapping_file(final_cnn_counts, final_image_counts, processed_images)
            
            # 6. 결과 요약
            print(f"\n{'='*60}")
            print("📊 데이터 준비 완료!")
            print(f"{'='*60}")
            
            print(f"📁 출력 디렉토리: {self.output_dir}")
            print(f"📄 매핑 파일: {mapping_file}")
            print(f"🖼️ 처리된 이미지: {processed_images}개")
            
            print("\n📊 초기 데이터:")
            print("-" * 50)
            for category in ['상의', '하의', '아우터', '원피스']:
                cnn_count = cnn_counts.get(category, 0)
                img_count = image_counts.get(category, 0)
                print(f"{category:8s}: CNN 라벨 {cnn_count:5d}개, 크롭 이미지 {img_count:5d}개")
            
            print("\n📊 동기화 후 최종 데이터:")
            print("-" * 50)
            print(f"{'카테고리':8s} {'CNN라벨':8s} {'크롭파일':8s} {'매칭':4s} {'변화':12s}")
            print("-" * 50)
            
            all_matched = True
            for category in ['상의', '하의', '아우터', '원피스']:
                final_cnn = final_cnn_counts.get(category, 0)
                final_img_files = final_image_counts.get(category, 0)
                removed = removed_counts.get(category, 0)
                created = created_counts.get(category, 0)
                
                # CNN 라벨 수와 크롭 이미지 파일 수가 일치하는지 확인
                match_status = "✅" if final_cnn == final_img_files else "❌"
                if final_cnn != final_img_files:
                    all_matched = False
                
                change_info = f"제거:{removed}, 생성:{created}"
                print(f"{category:8s} {final_cnn:6d}개 {final_img_files:6d}개 {match_status:4s} {change_info}")
            
            print("-" * 50)
            
            if all_matched:
                print(f"\n✅ 모든 카테고리의 CNN 라벨과 크롭 이미지 파일 수가 일치합니다!")
                print("📝 CNN 라벨 파일명에 시퀀스 번호(0001, 0002)가 추가되었습니다.")
            else:
                print(f"\n⚠️ 일부 카테고리에서 파일 수가 일치하지 않습니다.")
            
            print(f"\n🎉 모든 작업이 완료되었습니다!")
            
            # 로그 저장
            self.save_logs()
            
        except Exception as e:
            self.log_entry("ERROR", f"데이터 준비 중 오류 발생", f"오류: {str(e)}")
            self.save_logs()
            print(f"❌ 데이터 준비 중 오류 발생: {e}")
            raise

def main():
    """메인 실행 함수"""
    print("🎯 데이터 준비 스크립트")
    print("CNN 라벨 분리 및 이미지 크롭")
    print("=" * 60)
    
    # 데이터 준비기 초기화
    preparer = DataPreparation()
    
    # 데이터 준비 실행
    preparer.run_preparation()

if __name__ == "__main__":
    main()

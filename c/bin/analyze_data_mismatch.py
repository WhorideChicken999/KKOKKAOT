#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 불일치 분석 스크립트
CNN 라벨과 크롭 이미지 수가 다른 이유를 분석합니다.
"""

import os
import json
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DataMismatchAnalyzer:
    """데이터 불일치 분석 클래스"""
    
    def __init__(self, data_dir="D:/converted_data"):
        self.data_dir = Path(data_dir)
        
        # 경로 설정
        self.cnn_labels_dir = self.data_dir / "prepared_data" / "cnn_labels"
        self.cropped_images_dir = self.data_dir / "prepared_data" / "cropped_images"
        self.original_cnn_dir = self.data_dir / "cnn"
        self.original_yolo_dir = self.data_dir / "yolo"
        self.images_dir = self.data_dir / "all_images"
        
        print(f"📁 분석 대상 디렉토리: {self.data_dir}")
    
    def analyze_cnn_labels(self):
        """CNN 라벨 분석"""
        print("\n🔍 CNN 라벨 분석 중...")
        
        cnn_label_files = defaultdict(list)
        cnn_image_ids = defaultdict(set)
        
        categories = ['상의', '하의', '아우터', '원피스']
        
        for category in categories:
            category_dir = self.cnn_labels_dir / category
            if category_dir.exists():
                files = list(category_dir.glob("*.json"))
                cnn_label_files[category] = files
                
                # 이미지 ID 추출
                for file in files:
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        image_id = data.get('image_id')
                        if image_id:
                            cnn_image_ids[category].add(image_id)
                    except Exception as e:
                        print(f"⚠️ CNN 라벨 파일 읽기 오류 {file}: {e}")
        
        print("CNN 라벨 통계:")
        for category in categories:
            file_count = len(cnn_label_files[category])
            unique_images = len(cnn_image_ids[category])
            print(f"  {category}: {file_count}개 파일, {unique_images}개 고유 이미지")
        
        return cnn_label_files, cnn_image_ids
    
    def analyze_cropped_images(self):
        """크롭된 이미지 분석"""
        print("\n🔍 크롭된 이미지 분석 중...")
        
        cropped_files = defaultdict(list)
        cropped_image_ids = defaultdict(set)
        
        categories = ['상의', '하의', '아우터', '원피스']
        
        for category in categories:
            category_dir = self.cropped_images_dir / category
            if category_dir.exists():
                files = list(category_dir.glob("*.jpg"))
                cropped_files[category] = files
                
                # 이미지 ID 추출 (파일명에서)
                for file in files:
                    try:
                        # 파일명 형식: imageID_category_sequence.jpg
                        filename = file.stem
                        parts = filename.split('_')
                        if len(parts) >= 2:
                            image_id = parts[0]
                            cropped_image_ids[category].add(image_id)
                    except Exception as e:
                        print(f"⚠️ 크롭 이미지 파일명 파싱 오류 {file}: {e}")
        
        print("크롭된 이미지 통계:")
        for category in categories:
            file_count = len(cropped_files[category])
            unique_images = len(cropped_image_ids[category])
            print(f"  {category}: {file_count}개 파일, {unique_images}개 고유 이미지")
        
        return cropped_files, cropped_image_ids
    
    def analyze_original_data(self):
        """원본 데이터 분석"""
        print("\n🔍 원본 데이터 분석 중...")
        
        # 원본 CNN 파일들
        original_cnn_files = list(self.original_cnn_dir.glob("*.json"))
        print(f"원본 CNN 파일: {len(original_cnn_files)}개")
        
        # 원본 YOLO 파일들
        original_yolo_files = list(self.original_yolo_dir.glob("*.json"))
        print(f"원본 YOLO 파일: {len(original_yolo_files)}개")
        
        # 원본 이미지 파일들
        original_images = list(self.images_dir.glob("*.jpg"))
        print(f"원본 이미지 파일: {len(original_images)}개")
        
        return original_cnn_files, original_yolo_files, original_images
    
    def analyze_missing_images(self, cnn_image_ids, cropped_image_ids):
        """누락된 이미지 분석"""
        print("\n🔍 누락된 이미지 분석 중...")
        
        categories = ['상의', '하의', '아우터', '원피스']
        
        for category in categories:
            cnn_images = cnn_image_ids[category]
            cropped_images = cropped_image_ids[category]
            
            # CNN에는 있지만 크롭되지 않은 이미지들
            missing_in_cropped = cnn_images - cropped_images
            # 크롭되었지만 CNN에 없는 이미지들 (일반적으로 없어야 함)
            extra_in_cropped = cropped_images - cnn_images
            
            print(f"\n{category} 분석:")
            print(f"  CNN 라벨에만 있는 이미지: {len(missing_in_cropped)}개")
            print(f"  크롭 이미지에만 있는 이미지: {len(extra_in_cropped)}개")
            
            if missing_in_cropped:
                print(f"  누락된 이미지 ID 예시: {list(missing_in_cropped)[:10]}")
    
    def analyze_yolo_detection_failures(self, missing_image_ids):
        """YOLO 감지 실패 분석"""
        print("\n🔍 YOLO 감지 실패 분석 중...")
        
        detection_failures = []
        confidence_issues = []
        bbox_issues = []
        
        for image_id in tqdm(list(missing_image_ids)[:1000], desc="YOLO 분석 중"):  # 샘플 분석
            yolo_file = self.original_yolo_dir / f"yolo_{image_id}.json"
            
            if not yolo_file.exists():
                detection_failures.append(image_id)
                continue
            
            try:
                with open(yolo_file, 'r', encoding='utf-8') as f:
                    yolo_data = json.load(f)
                
                if not yolo_data.get('annotations'):
                    detection_failures.append(image_id)
                    continue
                
                # 신뢰도 분석
                low_confidence = True
                small_bbox = True
                
                for annotation in yolo_data['annotations']:
                    confidence = annotation.get('confidence', 0)
                    bbox = annotation.get('bbox', [0, 0, 0, 0])
                    
                    if confidence >= 0.5:
                        low_confidence = False
                    
                    if bbox[2] >= 32 and bbox[3] >= 32:  # width, height
                        small_bbox = False
                
                if low_confidence:
                    confidence_issues.append(image_id)
                if small_bbox:
                    bbox_issues.append(image_id)
                    
            except Exception as e:
                print(f"⚠️ YOLO 파일 분석 오류 {yolo_file}: {e}")
        
        print(f"YOLO 감지 실패 원인 (샘플 1000개 분석):")
        print(f"  파일 없음: {len(detection_failures)}개")
        print(f"  신뢰도 낮음: {len(confidence_issues)}개")
        print(f"  바운딩 박스 작음: {len(bbox_issues)}개")
    
    def check_image_file_availability(self, missing_image_ids):
        """원본 이미지 파일 가용성 확인"""
        print("\n🔍 원본 이미지 파일 가용성 확인 중...")
        
        missing_files = []
        available_files = []
        
        for image_id in tqdm(list(missing_image_ids)[:1000], desc="이미지 파일 확인 중"):
            image_file = self.images_dir / f"{image_id}.jpg"
            
            if image_file.exists():
                available_files.append(image_id)
            else:
                missing_files.append(image_id)
        
        print(f"원본 이미지 파일 상태 (샘플 1000개):")
        print(f"  파일 있음: {len(available_files)}개")
        print(f"  파일 없음: {len(missing_files)}개")
        
        if missing_files:
            print(f"  누락된 파일 예시: {missing_files[:10]}")
    
    def generate_report(self, cnn_image_ids, cropped_image_ids):
        """분석 보고서 생성"""
        print("\n📊 분석 보고서 생성 중...")
        
        categories = ['상의', '하의', '아우터', '원피스']
        
        report = {
            "summary": {
                "total_cnn_labels": sum(len(cnn_image_ids[cat]) for cat in categories),
                "total_cropped_images": sum(len(cropped_image_ids[cat]) for cat in categories),
                "total_missing": 0
            },
            "category_analysis": {},
            "recommendations": []
        }
        
        for category in categories:
            cnn_count = len(cnn_image_ids[category])
            cropped_count = len(cropped_image_ids[category])
            missing_count = cnn_count - cropped_count
            
            report["category_analysis"][category] = {
                "cnn_labels": cnn_count,
                "cropped_images": cropped_count,
                "missing": missing_count,
                "missing_rate": (missing_count / cnn_count * 100) if cnn_count > 0 else 0
            }
            
            report["summary"]["total_missing"] += missing_count
        
        # 권장사항 생성
        if report["summary"]["total_missing"] > 0:
            report["recommendations"].append("YOLO 모델의 신뢰도 임계값을 낮춰보세요 (현재 0.5)")
            report["recommendations"].append("바운딩 박스 최소 크기 임계값을 줄여보세요 (현재 32x32)")
            report["recommendations"].append("원본 이미지 파일의 가용성을 확인하세요")
            report["recommendations"].append("CNN 라벨과 YOLO 라벨의 데이터 일관성을 검토하세요")
        
        # 보고서 저장
        report_file = self.data_dir / "prepared_data" / "data_mismatch_analysis.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📄 분석 보고서 저장: {report_file}")
        
        return report
    
    def run_analysis(self):
        """전체 분석 실행"""
        print("🔍 데이터 불일치 분석 시작!")
        print("=" * 60)
        
        try:
            # 1. CNN 라벨 분석
            cnn_label_files, cnn_image_ids = self.analyze_cnn_labels()
            
            # 2. 크롭된 이미지 분석
            cropped_files, cropped_image_ids = self.analyze_cropped_images()
            
            # 3. 원본 데이터 분석
            self.analyze_original_data()
            
            # 4. 누락된 이미지 분석
            self.analyze_missing_images(cnn_image_ids, cropped_image_ids)
            
            # 5. YOLO 감지 실패 분석 (샘플)
            all_cnn_images = set()
            for category_images in cnn_image_ids.values():
                all_cnn_images.update(category_images)
            
            all_cropped_images = set()
            for category_images in cropped_image_ids.values():
                all_cropped_images.update(category_images)
            
            missing_images = all_cnn_images - all_cropped_images
            
            if missing_images:
                self.analyze_yolo_detection_failures(missing_images)
                self.check_image_file_availability(missing_images)
            
            # 6. 보고서 생성
            report = self.generate_report(cnn_image_ids, cropped_image_ids)
            
            print("\n" + "=" * 60)
            print("📊 분석 완료!")
            print("=" * 60)
            
            print(f"총 CNN 라벨: {report['summary']['total_cnn_labels']:,}개")
            print(f"총 크롭 이미지: {report['summary']['total_cropped_images']:,}개")
            print(f"총 누락: {report['summary']['total_missing']:,}개")
            
            print("\n카테고리별 상세:")
            for category, data in report["category_analysis"].items():
                print(f"  {category}: {data['missing']:,}개 누락 ({data['missing_rate']:.1f}%)")
            
            if report["recommendations"]:
                print("\n권장사항:")
                for i, rec in enumerate(report["recommendations"], 1):
                    print(f"  {i}. {rec}")
            
            return report
            
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """메인 실행 함수"""
    print("🎯 데이터 불일치 분석 도구")
    print("CNN 라벨과 크롭 이미지 수가 다른 이유를 분석합니다.")
    print("=" * 60)
    
    analyzer = DataMismatchAnalyzer()
    report = analyzer.run_analysis()
    
    if report:
        print("\n🎉 분석이 완료되었습니다!")
    else:
        print("\n💥 분석에 실패했습니다.")

if __name__ == "__main__":
    main()

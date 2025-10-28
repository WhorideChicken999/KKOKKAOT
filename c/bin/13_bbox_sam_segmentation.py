#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 bbox 이미지들을 SAM으로 누끼따기
processed_images/ 폴더의 bbox 크롭 이미지들을 SAM으로 세그멘테이션하여 배경 제거
마네킹 합성을 위한 투명 배경 이미지 생성
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# SAM2 관련 import
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    print("⚠️ sam2가 설치되지 않았습니다.")
    print("설치 방법: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    SAM2_AVAILABLE = False

class BboxSAM2Segmentation:
    """bbox 이미지들을 SAM2로 세그멘테이션하는 클래스"""
    
    def __init__(self, sam_model_path, config_path, device="cuda"):
        """
        SAM2 모델 초기화
        
        Args:
            sam_model_path: SAM2 모델 가중치 파일 경로
            config_path: SAM2 설정 파일 경로
            device: 사용할 디바이스 ("cuda" 또는 "cpu")
        """
        if not SAM2_AVAILABLE:
            raise ImportError("sam2 라이브러리가 필요합니다.")
        
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"🚀 사용 중인 디바이스: {self.device}")
        
        # SAM2 모델 로드
        print(f"📦 SAM2 모델 로딩 중...")
        self.sam2 = build_sam2(sam2_cfg_file=config_path, 
                              sam2_ckpt_path=sam_model_path, 
                              device=self.device)
        
        # SAM2 예측기 초기화
        self.predictor = SAM2ImagePredictor(self.sam2)
        print("✅ SAM2 모델 로딩 완료!")
    
    def segment_bbox_image(self, image_path, output_path, use_center_point=True):
        """
        bbox 이미지를 세그멘테이션
        
        Args:
            image_path: bbox 이미지 경로
            output_path: 출력 경로
            use_center_point: 중앙점을 사용할지 여부
        
        Returns:
            success: 성공 여부
            mask_score: 마스크 점수
        """
        try:
            # 이미지 로드
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
                return False, 0.0
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image_rgb.shape[:2]
            
            # 이미지 설정
            self.predictor.set_image(image_rgb)
            
            # 전체 이미지를 bbox로 사용
            bbox = np.array([0, 0, width, height])
            
            # 중앙점 추가 (선택사항)
            point_coords = None
            point_labels = None
            if use_center_point:
                center_x, center_y = width // 2, height // 2
                point_coords = np.array([[center_x, center_y]])
                point_labels = np.array([1])  # 포그라운드 포인트
            
            # 세그멘테이션 예측
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=bbox,
                multimask_output=True,
            )
            
            # 가장 높은 점수의 마스크 선택
            best_mask = masks[np.argmax(scores)]
            best_score = scores[np.argmax(scores)]
            
            # 투명 배경 이미지 생성
            transparent_image = self.create_transparent_image(image_rgb, best_mask)
            
            # PNG로 저장
            pil_image = Image.fromarray(transparent_image, 'RGBA')
            pil_image.save(output_path, "PNG")
            
            return True, best_score
            
        except Exception as e:
            print(f"❌ 세그멘테이션 실패 {image_path}: {e}")
            return False, 0.0
    
    def create_transparent_image(self, image, mask, alpha=1.0):
        """
        마스크를 사용하여 투명 배경 이미지 생성
        
        Args:
            image: 원본 이미지
            mask: 세그멘테이션 마스크
            alpha: 투명도 (0.0 ~ 1.0)
        
        Returns:
            transparent_image: 투명 배경 이미지 (RGBA)
        """
        # 이미지를 RGBA로 변환
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # RGB -> RGBA
                rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
                rgba_image[:, :, :3] = image
                rgba_image[:, :, 3] = 255  # 알파 채널을 255로 설정
            else:
                rgba_image = image.copy()
        else:
            # 그레이스케일 -> RGBA
            rgba_image = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            rgba_image[:, :, :3] = np.stack([image] * 3, axis=2)
            rgba_image[:, :, 3] = 255
        
        # 마스크를 알파 채널에 적용
        rgba_image[:, :, 3] = (mask * 255 * alpha).astype(np.uint8)
        
        return rgba_image
    
    def process_category_folder(self, input_folder, output_folder, category_name):
        """
        특정 카테고리 폴더의 모든 이미지 처리
        
        Args:
            input_folder: 입력 폴더 경로
            output_folder: 출력 폴더 경로
            category_name: 카테고리 이름
        
        Returns:
            results: 처리 결과 리스트
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        
        if not input_path.exists():
            print(f"⚠️ 입력 폴더가 존재하지 않습니다: {input_folder}")
            return []
        
        # 출력 폴더 생성
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 이미지 파일 찾기
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        if len(image_files) == 0:
            print(f"⚠️ {category_name} 폴더에 이미지가 없습니다: {input_folder}")
            return []
        
        print(f"📊 {category_name} 처리할 이미지: {len(image_files)}개")
        
        results = []
        success_count = 0
        
        for image_file in tqdm(image_files, desc=f"{category_name} 처리 중"):
            # 출력 파일명 생성 (확장자를 .png로 변경)
            output_filename = image_file.stem + "_segmented.png"
            output_file_path = output_path / output_filename
            
            # 세그멘테이션 수행
            success, score = self.segment_bbox_image(
                image_file, 
                output_file_path,
                use_center_point=True
            )
            
            if success:
                success_count += 1
                results.append({
                    'input_path': str(image_file),
                    'output_path': str(output_file_path),
                    'category': category_name,
                    'mask_score': float(score),
                    'success': True
                })
                print(f"  ✅ {image_file.name} -> {output_filename} (점수: {score:.3f})")
            else:
                results.append({
                    'input_path': str(image_file),
                    'output_path': str(output_file_path),
                    'category': category_name,
                    'mask_score': 0.0,
                    'success': False
                })
                print(f"  ❌ {image_file.name} 처리 실패")
        
        print(f"📊 {category_name} 처리 완료: {success_count}/{len(image_files)}개 성공")
        return results
    
    def process_all_categories(self, processed_images_dir, output_dir):
        """
        모든 카테고리의 bbox 이미지 처리
        
        Args:
            processed_images_dir: processed_images 디렉토리 경로
            output_dir: 출력 디렉토리 경로
        
        Returns:
            all_results: 모든 처리 결과
        """
        processed_path = Path(processed_images_dir)
        output_path = Path(output_dir)
        
        if not processed_path.exists():
            print(f"❌ processed_images 디렉토리가 존재하지 않습니다: {processed_images_dir}")
            return []
        
        # 카테고리별 처리
        categories = ['top', 'bottom', 'outer', 'dress']
        all_results = []
        
        for category in categories:
            input_folder = processed_path / category
            output_folder = output_path / f"{category}_segmented"
            
            print(f"\n🎯 {category.upper()} 카테고리 처리 시작")
            print(f"  입력: {input_folder}")
            print(f"  출력: {output_folder}")
            
            category_results = self.process_category_folder(
                input_folder, 
                output_folder, 
                category
            )
            
            all_results.extend(category_results)
        
        return all_results
    
    def generate_summary_report(self, results, output_dir):
        """
        처리 결과 요약 리포트 생성
        
        Args:
            results: 처리 결과 리스트
            output_dir: 출력 디렉토리
        """
        if not results:
            print("⚠️ 생성할 리포트가 없습니다.")
            return
        
        # 통계 계산
        total_files = len(results)
        successful_files = sum(1 for r in results if r['success'])
        failed_files = total_files - successful_files
        
        # 카테고리별 통계
        category_stats = {}
        for result in results:
            category = result['category']
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'success': 0, 'failed': 0, 'scores': []}
            
            category_stats[category]['total'] += 1
            if result['success']:
                category_stats[category]['success'] += 1
                category_stats[category]['scores'].append(result['mask_score'])
            else:
                category_stats[category]['failed'] += 1
        
        # 리포트 생성
        report = {
            'summary': {
                'total_files': total_files,
                'successful_files': successful_files,
                'failed_files': failed_files,
                'success_rate': successful_files / total_files * 100 if total_files > 0 else 0
            },
            'category_stats': {}
        }
        
        for category, stats in category_stats.items():
            avg_score = np.mean(stats['scores']) if stats['scores'] else 0.0
            report['category_stats'][category] = {
                'total': stats['total'],
                'success': stats['success'],
                'failed': stats['failed'],
                'success_rate': stats['success'] / stats['total'] * 100 if stats['total'] > 0 else 0,
                'average_mask_score': float(avg_score)
            }
        
        # JSON 리포트 저장
        report_path = Path(output_dir) / "segmentation_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 콘솔 출력
        print(f"\n📊 처리 결과 요약")
        print(f"{'='*50}")
        print(f"전체 파일: {total_files}개")
        print(f"성공: {successful_files}개 ({successful_files/total_files*100:.1f}%)")
        print(f"실패: {failed_files}개 ({failed_files/total_files*100:.1f}%)")
        
        print(f"\n📈 카테고리별 통계:")
        for category, stats in category_stats.items():
            print(f"  {category.upper()}:")
            print(f"    전체: {stats['total']}개")
            print(f"    성공: {stats['success']}개 ({stats['success']/stats['total']*100:.1f}%)")
            print(f"    평균 점수: {np.mean(stats['scores']):.3f}" if stats['scores'] else "    평균 점수: 0.000")
        
        print(f"\n💾 리포트 저장: {report_path}")

def main():
    """메인 실행 함수"""
    print("🎯 bbox 이미지 SAM 세그멘테이션 시작!")
    print("=" * 60)
    
    # 모델 경로 설정
    sam_model_path = "API/pre_trained_weights/sam2_best.pt"
    sam_config_path = "API/pre_trained_weights/sam2_hiera_t.yaml"
    
    # 경로 확인
    if not Path(sam_model_path).exists():
        print(f"❌ SAM2 모델을 찾을 수 없습니다: {sam_model_path}")
        return
    
    if not Path(sam_config_path).exists():
        print(f"❌ SAM2 설정 파일을 찾을 수 없습니다: {sam_config_path}")
        return
    
    # 입력/출력 디렉토리 설정
    processed_images_dir = "API/processed_images"
    output_dir = "segmented_bbox_images"
    
    # 입력 디렉토리 확인
    if not Path(processed_images_dir).exists():
        print(f"❌ processed_images 디렉토리가 존재하지 않습니다: {processed_images_dir}")
        return
    
    # 세그멘테이션 프로세서 초기화
    try:
        segmenter = BboxSAM2Segmentation(sam_model_path, sam_config_path)
    except Exception as e:
        print(f"❌ SAM2 모델 초기화 실패: {e}")
        return
    
    # 모든 카테고리 처리
    print(f"\n📁 입력 디렉토리: {processed_images_dir}")
    print(f"📁 출력 디렉토리: {output_dir}")
    
    results = segmenter.process_all_categories(processed_images_dir, output_dir)
    
    if results:
        # 결과 요약 리포트 생성
        segmenter.generate_summary_report(results, output_dir)
        
        # 상세 결과를 JSON으로 저장
        results_path = Path(output_dir) / "detailed_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 상세 결과 저장: {results_path}")
    
    print(f"\n🎉 모든 처리가 완료되었습니다!")
    print(f"📁 세그멘테이션 결과: {output_dir}/")
    print(f"🎭 마네킹 합성 준비 완료!")

if __name__ == "__main__":
    main()

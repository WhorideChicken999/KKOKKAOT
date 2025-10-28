#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM 모델을 사용한 이미지 세그멘테이션 및 누끼따기
YOLO로 감지된 bbox 영역을 SAM으로 정확히 세그멘테이션하여 배경 제거
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

class SAM2Segmentation:
    """SAM2 모델을 사용한 이미지 세그멘테이션 클래스"""
    
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
    
    def segment_bbox(self, image, bbox, point_coords=None, point_labels=None):
        """
        bbox 영역을 세그멘테이션
        
        Args:
            image: 입력 이미지 (numpy array)
            bbox: 바운딩 박스 [x1, y1, x2, y2]
            point_coords: 포인트 좌표 (선택사항)
            point_labels: 포인트 라벨 (선택사항)
        
        Returns:
            mask: 세그멘테이션 마스크
        """
        # 이미지 설정
        self.predictor.set_image(image)
        
        # bbox를 SAM 형식으로 변환 [x1, y1, x2, y2] -> [x, y, w, h]
        x1, y1, x2, y2 = bbox
        sam_bbox = np.array([x1, y1, x2, y2])
        
        # 세그멘테이션 예측
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=sam_bbox,
            multimask_output=True,
        )
        
        # 가장 높은 점수의 마스크 선택
        best_mask = masks[np.argmax(scores)]
        
        return best_mask, scores[np.argmax(scores)]
    
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
    
    def save_segmented_image(self, image, mask, output_path, format="PNG"):
        """
        세그멘테이션된 이미지 저장
        
        Args:
            image: 원본 이미지
            mask: 세그멘테이션 마스크
            output_path: 저장 경로
            format: 저장 형식 ("PNG", "JPG")
        """
        if format.upper() == "PNG":
            # 투명 배경으로 PNG 저장
            transparent_image = self.create_transparent_image(image, mask)
            pil_image = Image.fromarray(transparent_image, 'RGBA')
            pil_image.save(output_path, "PNG")
        else:
            # 마스크를 적용한 이미지 저장
            masked_image = image.copy()
            masked_image[~mask] = [0, 0, 0]  # 배경을 검은색으로
            pil_image = Image.fromarray(masked_image)
            pil_image.save(output_path, "JPEG")

class YOLOSAM2Processor:
    """YOLO + SAM2를 사용한 이미지 처리 클래스"""
    
    def __init__(self, yolo_model_path, sam_model_path, sam_config_path):
        """
        YOLO + SAM2 프로세서 초기화
        
        Args:
            yolo_model_path: YOLO 모델 경로
            sam_model_path: SAM2 모델 경로
            sam_config_path: SAM2 설정 파일 경로
        """
        # YOLO 모델 로드
        print("📦 YOLO 모델 로딩 중...")
        try:
            self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path)
            print("✅ YOLO 모델 로딩 완료!")
        except Exception as e:
            print(f"❌ YOLO 모델 로딩 실패: {e}")
            self.yolo_model = None
        
        # SAM2 모델 로드
        self.sam2_segmenter = SAM2Segmentation(sam_model_path, sam_config_path)
        
        # 클래스 이름 매핑 (YOLO 모델에 따라 조정 필요)
        self.class_names = {
            0: 'top',
            1: 'bottom', 
            2: 'outer',
            3: 'dress'
        }
    
    def detect_and_segment(self, image_path, output_dir, confidence_threshold=0.3):
        """
        이미지에서 객체를 감지하고 세그멘테이션
        
        Args:
            image_path: 입력 이미지 경로
            output_dir: 출력 디렉토리
            confidence_threshold: 신뢰도 임계값
        
        Returns:
            results: 처리 결과 정보
        """
        if self.yolo_model is None:
            print("❌ YOLO 모델이 로드되지 않았습니다.")
            return None
        
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # YOLO로 객체 감지
        results = self.yolo_model(image_rgb)
        detections = results.pandas().xyxy[0]
        
        # 신뢰도 필터링
        detections = detections[detections['confidence'] >= confidence_threshold]
        
        if len(detections) == 0:
            print(f"⚠️ 감지된 객체가 없습니다: {image_path}")
            return None
        
        # 출력 디렉토리 생성
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 이미지 이름 (확장자 제외)
        image_name = Path(image_path).stem
        
        processed_results = []
        
        # 각 감지된 객체에 대해 세그멘테이션 수행
        for idx, detection in detections.iterrows():
            # bbox 좌표
            x1, y1, x2, y2 = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            
            # 클래스 정보
            class_id = int(detection['class'])
            class_name = self.class_names.get(class_id, f'class_{class_id}')
            confidence = detection['confidence']
            
            print(f"  🎯 {class_name} 감지 (신뢰도: {confidence:.3f})")
            
            # SAM으로 세그멘테이션
            try:
                mask, score = self.sam_segmenter.segment_bbox(image_rgb, bbox)
                
                # 세그멘테이션된 이미지 저장
                output_filename = f"{image_name}_{class_name}_{idx}.png"
                output_path = output_dir / output_filename
                
                self.sam_segmenter.save_segmented_image(
                    image_rgb, mask, output_path, format="PNG"
                )
                
                processed_results.append({
                    'class_name': class_name,
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': bbox,
                    'mask_score': score,
                    'output_path': str(output_path)
                })
                
                print(f"    ✅ 세그멘테이션 완료: {output_filename}")
                
            except Exception as e:
                print(f"    ❌ 세그멘테이션 실패: {e}")
                continue
        
        return processed_results
    
    def process_batch(self, input_dir, output_dir, confidence_threshold=0.3):
        """
        배치 처리로 여러 이미지 처리
        
        Args:
            input_dir: 입력 이미지 디렉토리
            output_dir: 출력 디렉토리
            confidence_threshold: 신뢰도 임계값
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # 지원하는 이미지 확장자
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 이미지 파일 찾기
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
            image_files.extend(input_dir.glob(f"*{ext.upper()}"))
        
        if len(image_files) == 0:
            print(f"❌ 입력 디렉토리에 이미지가 없습니다: {input_dir}")
            return
        
        print(f"📊 처리할 이미지: {len(image_files)}개")
        
        # 배치 처리
        all_results = []
        for image_file in tqdm(image_files, desc="이미지 처리 중"):
            print(f"\n🔄 처리 중: {image_file.name}")
            
            results = self.detect_and_segment(
                image_file, output_dir, confidence_threshold
            )
            
            if results:
                all_results.extend(results)
        
        # 결과 요약
        print(f"\n📊 처리 완료!")
        print(f"  - 처리된 이미지: {len(image_files)}개")
        print(f"  - 생성된 세그멘테이션: {len(all_results)}개")
        
        # 클래스별 통계
        class_counts = {}
        for result in all_results:
            class_name = result['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"  - 클래스별 통계:")
        for class_name, count in class_counts.items():
            print(f"    {class_name}: {count}개")
        
        return all_results

def main():
    """메인 실행 함수"""
    print("🎯 YOLO + SAM 세그멘테이션 시작!")
    print("=" * 60)
    
    # 모델 경로 설정
    yolo_model_path = "API/pre_trained_weights/yolo_best.pt"
    sam_model_path = "API/pre_trained_weights/sam2_best.pt"
    sam_config_path = "API/pre_trained_weights/sam2_hiera_t.yaml"
    
    # 경로 확인
    if not Path(yolo_model_path).exists():
        print(f"❌ YOLO 모델을 찾을 수 없습니다: {yolo_model_path}")
        return
    
    if not Path(sam_model_path).exists():
        print(f"❌ SAM2 모델을 찾을 수 없습니다: {sam_model_path}")
        return
    
    if not Path(sam_config_path).exists():
        print(f"❌ SAM2 설정 파일을 찾을 수 없습니다: {sam_config_path}")
        return
    
    # 프로세서 초기화
    try:
        processor = YOLOSAM2Processor(yolo_model_path, sam_model_path, sam_config_path)
    except Exception as e:
        print(f"❌ 모델 초기화 실패: {e}")
        return
    
    # 처리할 이미지 디렉토리 설정
    input_dirs = [
        "converted_data/all_images",  # converted_data의 이미지들
        "API/uploaded_images",        # 업로드된 이미지들
        "API/default_items"           # 기본 아이템들
    ]
    
    output_base_dir = "segmented_images"
    
    # 각 입력 디렉토리에 대해 처리
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"⚠️ 입력 디렉토리가 존재하지 않습니다: {input_dir}")
            continue
        
        print(f"\n📁 처리할 디렉토리: {input_dir}")
        
        # 출력 디렉토리 설정
        output_dir = Path(output_base_dir) / input_path.name
        
        # 배치 처리 실행
        results = processor.process_batch(
            input_path, 
            output_dir, 
            confidence_threshold=0.3
        )
        
        if results:
            # 결과를 JSON으로 저장
            results_file = output_dir / "segmentation_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 결과 저장: {results_file}")
    
    print(f"\n🎉 모든 처리가 완료되었습니다!")
    print(f"📁 세그멘테이션 결과: {output_base_dir}/")

if __name__ == "__main__":
    main()

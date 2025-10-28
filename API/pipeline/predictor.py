"""
예측 파이프라인
- 전체 흐름: 이미지 입력 → Gender → Style → YOLO → Crop → 속성 예측
"""
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import Dict, Tuple, Optional
from .models import STYLE_CLASSES, GENDER_CLASSES


class FashionPredictor:
    """패션 아이템 예측 파이프라인"""
    
    def __init__(self, model_loader):
        """
        Args:
            model_loader: ModelLoader 인스턴스
        """
        self.loader = model_loader
        self.device = model_loader.device
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def predict_gender(self, image: np.ndarray) -> Dict:
        """
        1단계: 성별 예측 (전체 이미지)
        
        Args:
            image: RGB 이미지 (numpy array)
        
        Returns:
            {'gender': 'male', 'confidence': 0.95}
        """
        print("\n[1/6] 성별 예측 중...")
        
        if self.loader.gender_model is None:
            print("  ⚠️ 성별 모델 없음 - 기본값 사용")
            return {'gender': 'female', 'confidence': 0.5}
        
        try:
            # 전처리
            pil_image = Image.fromarray(image)
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # 예측
            with torch.no_grad():
                outputs = self.loader.gender_model(image_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                gender_idx = probs.argmax().item()
                confidence = probs[gender_idx].item()
                gender = GENDER_CLASSES[gender_idx]
            
            result = {
                'gender': gender,
                'confidence': confidence
            }
            
            print(f"  ✅ 성별: {gender} ({confidence:.2%})")
            return result
            
        except Exception as e:
            print(f"  ❌ 성별 예측 실패: {e}")
            return {'gender': 'female', 'confidence': 0.0}
    
    def predict_style(self, image: np.ndarray) -> Dict:
        """
        2단계: 스타일 예측 (전체 이미지)
        
        Args:
            image: RGB 이미지
        
        Returns:
            {'style': '스트리트', 'confidence': 0.89}
        """
        print("\n[2/6] 스타일 예측 중...")
        
        if self.loader.style_model is None:
            print("  ⚠️ 스타일 모델 없음")
            return {'style': 'Unknown', 'confidence': 0.0}
        
        try:
            # 전처리
            pil_image = Image.fromarray(image)
            image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # 예측
            with torch.no_grad():
                outputs = self.loader.style_model(image_tensor)
                probs = torch.softmax(outputs, dim=1)[0]
                style_idx = probs.argmax().item()
                confidence = probs[style_idx].item()
                style = STYLE_CLASSES[style_idx]
            
            result = {
                'style': style,
                'confidence': confidence
            }
            
            print(f"  ✅ 스타일: {style} ({confidence:.2%})")
            return result
            
        except Exception as e:
            print(f"  ❌ 스타일 예측 실패: {e}")
            return {'style': 'Unknown', 'confidence': 0.0}
    
    def detect_items(self, image_path: str) -> Dict:
        """
        3단계: YOLO 디텍팅 (상의/하의/아우터/원피스)
        
        Args:
            image_path: 이미지 경로
        
        Returns:
            {
                'detected_items': {
                    'top': {'bbox': (x1,y1,x2,y2), 'confidence': 0.9, 'cropped_image': np.array},
                    'bottom': {...},
                },
                'has_top': True,
                'has_bottom': True,
                'has_outer': False,
                'has_dress': False
            }
        """
        print("\n[3/6] YOLO 디텍팅 중...")
        
        if self.loader.yolo_model is None:
            print("  ⚠️ YOLO 모델 없음")
            return {'detected_items': {}, 'has_top': False, 'has_bottom': False, 'has_outer': False, 'has_dress': False}
        
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지 로드 실패: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # YOLO 추론
            results = self.loader.yolo_model(image_path, verbose=False)
            
            # 클래스 매핑 (YOLO 모델의 클래스 순서)
            class_names = ['outer', 'top', 'bottom', 'dress']
            category_mapping = {
                'outer': 'outer',
                'top': 'top',
                'bottom': 'bottom',
                'dress': 'dress'
            }
            
            detected_items = {
                'top': [],
                'bottom': [],
                'outer': [],
                'dress': []
            }
            
            # 바운딩 박스 추출
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes
                print(f"  📦 감지된 박스: {len(boxes)}개")
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if confidence >= 0.3 and class_id < len(class_names):
                        class_name = class_names[class_id]
                        category = category_mapping.get(class_name)
                        
                        if category:
                            # 바운딩 박스 좌표
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # 유효성 검사
                            if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
                                # 이미지 크롭
                                cropped = image_rgb[y1:y2, x1:x2]
                                
                                if cropped.size > 0:
                                    detected_items[category].append({
                                        'bbox': (x1, y1, x2, y2),
                                        'confidence': confidence,
                                        'cropped_image': cropped
                                    })
                                    print(f"  ✅ {category}: conf={confidence:.2%}, bbox=({x1},{y1},{x2},{y2})")
            
            # 각 카테고리별로 가장 높은 confidence 선택
            final_items = {}
            for category, items in detected_items.items():
                if items:
                    best_item = max(items, key=lambda x: x['confidence'])
                    final_items[category] = best_item
            
            return {
                'detected_items': final_items,
                'has_top': 'top' in final_items,
                'has_bottom': 'bottom' in final_items,
                'has_outer': 'outer' in final_items,
                'has_dress': 'dress' in final_items
            }
            
        except Exception as e:
            print(f"  ❌ YOLO 디텍팅 실패: {e}")
            import traceback
            traceback.print_exc()
            return {'detected_items': {}, 'has_top': False, 'has_bottom': False, 'has_outer': False, 'has_dress': False}
    
    def save_cropped_images(self, detected_items: Dict, user_id: int, item_id: int, save_dir: Path) -> Dict:
        """
        4단계: Crop된 이미지 저장
        
        Args:
            detected_items: detect_items()의 결과
            user_id: 사용자 ID
            item_id: 아이템 ID
            save_dir: 저장 경로
        
        Returns:
            {'top': 'path/to/top.jpg', 'bottom': 'path/to/bottom.jpg', ...}
        """
        print("\n[4/6] Crop 이미지 저장 중...")
        
        saved_paths = {}
        
        for category, item in detected_items.items():
            try:
                # 저장 디렉토리 생성
                category_dir = save_dir / f"user_{user_id}" / category
                category_dir.mkdir(parents=True, exist_ok=True)
                
                # 파일 저장
                filename = f"item_{item_id}_{category}.jpg"
                file_path = category_dir / filename
                
                # numpy array → PIL Image → 저장
                cropped_image = item['cropped_image']
                pil_image = Image.fromarray(cropped_image)
                pil_image.save(file_path)
                
                saved_paths[category] = str(file_path)
                print(f"  ✅ {category}: {file_path}")
                
            except Exception as e:
                print(f"  ❌ {category} 저장 실패: {e}")
        
        return saved_paths
    
    def predict_attributes(self, detected_items: Dict, predicted_gender: str) -> Dict:
        """
        5단계: 각 Crop별 속성 예측
        
        Args:
            detected_items: detect_items()의 결과
            predicted_gender: 1단계에서 예측한 성별 ('male' or 'female')
        
        Returns:
            {
                'top': {'category': 'T셔츠', 'color': '검정', 'gender': 'male', ...},
                'bottom': {...},
            }
        """
        print(f"\n[5/6] 속성 예측 중... (gender={predicted_gender})")
        
        all_attributes = {}
        
        for category, item in detected_items.items():
            try:
                if category not in self.loader.attribute_models:
                    print(f"  ⚠️ {category} 모델 없음")
                    continue
                
                model_info = self.loader.attribute_models[category]
                model = model_info['model']
                encoders = model_info['encoders']
                attributes_list = model_info['attributes']
                
                # 전처리
                cropped_image = item['cropped_image']
                pil_image = Image.fromarray(cropped_image)
                image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
                
                # 예측
                with torch.no_grad():
                    outputs = model(image_tensor)
                    
                    category_attrs = {}
                    
                    for attr in attributes_list:
                        if attr == 'gender':
                            # gender는 예측하지 않고 이미 예측한 값 사용
                            category_attrs['gender'] = {
                                'value': predicted_gender,
                                'confidence': 1.0  # 전체 이미지에서 예측한 값이므로 신뢰도 높음
                            }
                        elif attr in outputs:
                            # 다른 속성은 모델로 예측
                            attr_output = outputs[attr]
                            probs = torch.softmax(attr_output, dim=1)[0]
                            pred_idx = probs.argmax().item()
                            confidence = probs[pred_idx].item()
                            
                            # 디코딩
                            predicted_class = encoders[attr].inverse_transform([pred_idx])[0]
                            
                            category_attrs[attr] = {
                                'value': predicted_class,
                                'confidence': confidence
                            }
                    
                    all_attributes[category] = category_attrs
                    print(f"  ✅ {category}: {len(category_attrs)}개 속성 예측 완료")
                
            except Exception as e:
                print(f"  ❌ {category} 속성 예측 실패: {e}")
                import traceback
                traceback.print_exc()
        
        return all_attributes
    
    def process_image(self, image_path: str, user_id: int, save_dir: Path = Path("./processed_images")) -> Dict:
        """
        전체 파이프라인 실행
        
        Args:
            image_path: 입력 이미지 경로
            user_id: 사용자 ID
            save_dir: 저장 경로
        
        Returns:
            {
                'success': True,
                'gender': {'gender': 'male', 'confidence': 0.95},
                'style': {'style': '스트리트', 'confidence': 0.89},
                'detection': {...},
                'saved_paths': {...},
                'attributes': {
                    'top': {'category': 'T셔츠', 'gender': 'male', ...},
                    'bottom': {...}
                }
            }
        """
        print(f"\n{'='*60}")
        print(f"🎯 패션 분석 파이프라인 시작")
        print(f"  - 이미지: {image_path}")
        print(f"  - 사용자: {user_id}")
        print(f"{'='*60}")
        
        try:
            # 이미지 로드
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지 로드 실패: {image_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 1. Gender 예측 (전체 이미지)
            gender_result = self.predict_gender(image_rgb)
            predicted_gender = gender_result['gender']
            
            # 2. Style 예측 (전체 이미지)
            style_result = self.predict_style(image_rgb)
            
            # 3. YOLO 디텍팅
            detection_result = self.detect_items(image_path)
            
            if not detection_result['detected_items']:
                print("\n❌ 의류 감지 실패")
                return {
                    'success': False,
                    'error': '의류를 감지하지 못했습니다.'
                }
            
            # 4. Crop 이미지 저장 (임시 item_id 사용, DB 저장 후 업데이트)
            # saved_paths = self.save_cropped_images(
            #     detection_result['detected_items'],
            #     user_id,
            #     item_id=0,  # 임시 ID
            #     save_dir=save_dir
            # )
            
            # 5. 속성 예측 (gender 포함)
            attributes = self.predict_attributes(
                detection_result['detected_items'],
                predicted_gender  # 👈 1단계에서 예측한 gender 전달
            )
            
            print(f"\n{'='*60}")
            print(f"✅ 파이프라인 완료")
            print(f"{'='*60}\n")
            
            return {
                'success': True,
                'gender': gender_result,
                'style': style_result,
                'detection': detection_result,
                # 'saved_paths': saved_paths,
                'attributes': attributes
            }
            
        except Exception as e:
            print(f"\n❌ 파이프라인 실패: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e)
            }


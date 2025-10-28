#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
새로운 메인 파이프라인
1. 전체 이미지 스타일 예측 (22개 스타일)
2. YOLO로 상의/하의/아우터/드레스 크롭 (confidence 0.5 이상)
3. 크롭된 이미지별 카테고리 속성 예측
4. 데이터베이스 저장 및 ChromaDB 임베딩
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from torchvision import models, transforms
import chromadb
import psycopg2
from psycopg2.extras import Json
import json
import pandas as pd
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

# GPU 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {DEVICE}")

class NewFashionPipeline:
    """새로운 패션 파이프라인"""
    
    def __init__(self, 
                style_model_path: str = "D:/kkokkaot/API/pre_trained_weights/k_fashion_best_model.pth",
                yolo_detection_path: str = "D:/kkokkaot/API/pre_trained_weights/yolo_best.pt",
                category_models_dir: str = "D:/kkokkaot/API/pre_trained_weights/category_attributes",
                schema_path: str = "D:/kkokkaot/API/kfashion_attributes_schema.csv",
                chroma_path: str = "./chroma_db",
                db_config: dict = None):
        """초기화"""
        
        print("\n=== 새로운 패션 파이프라인 모델 로딩 중 ===")
        
        # 1. 스타일 예측 모델 로드
        print("1. 스타일 예측 모델 로드...")
        self.style_model, self.style_classes = self.load_style_model(style_model_path)
        
        # 2. YOLO Detection 모델 로드
        print("2. YOLO Detection 모델 로드...")
        self.yolo_detection_model = YOLO(yolo_detection_path)
        
        # 3. 카테고리별 속성 모델 로드
        print("3. 카테고리별 속성 모델 로드...")
        self.category_models = self.load_category_models(category_models_dir)
        
        # 4. 스키마 로드
        print("4. 속성 스키마 로드...")
        self.schema = self.load_schema(schema_path)
        
        # 5. CLIP 모델 (임베딩용)
        print("5. CLIP 모델 로드...")
        self.clip_model, self.clip_processor = self.load_clip_model()
        
        # 6. ChromaDB
        print("6. ChromaDB 연결...")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        try:
            self.chroma_collection = self.chroma_client.get_collection(name="fashion_collection")
        except:
            self.chroma_collection = self.chroma_client.create_collection(name="fashion_collection")
        
        # 7. PostgreSQL
        print("7. PostgreSQL 연결...")
        if db_config is None:
            db_config = {
                'host': 'localhost',
                'port': 5432,
                'database': 'kkokkaot_closet',
                'user': 'postgres',
                'password': '000000'
            }
        self.db_config = db_config
        self.db_conn = psycopg2.connect(**db_config)
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("\n✓ 모든 모델 로드 완료\n")
    
    def load_style_model(self, model_path: str):
        """스타일 예측 모델 로드"""
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            
            # 모델 구조 추정 (일반적인 ResNet 기반)
            num_classes = 22  # 22개 스타일
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(checkpoint)
            model.to(DEVICE)
            model.eval()
            
            # 스타일 클래스 정의 (스키마에서 추출)
            style_classes = [
                '로맨틱', '페미닌', '섹시', '젠더리스/젠더플루이드', '매스큘린', '톰보이',
                '히피', '오리엔탈', '웨스턴', '컨트리', '리조트', '모던',
                '소피스트케이티드', '아방가르드', '펑크', '키치/키덜트', '레트로',
                '힙합', '클래식', '프레피', '스트리트', '밀리터리', '스포티'
            ]
            
            return model, style_classes
            
        except Exception as e:
            print(f"❌ 스타일 모델 로드 실패: {e}")
            return None, None
    
    def load_category_models(self, models_dir: str):
        """카테고리별 속성 모델 로드"""
        models_dir = Path(models_dir)
        category_models = {}
        
        categories = ['상의', '하의', '아우터', '원피스']
        
        for category in categories:
            category_models[category] = {}
            
            # 각 속성별 모델 로드
            attributes = ['카테고리', '색상', '핏', '소재', '기장', '소매기장', '넥라인', '프린트']
            
            for attribute in attributes:
                model_file = models_dir / f"best_model_{category}_{attribute}.pth"
                info_file = models_dir / f"model_info_{category}_{attribute}.json"
                
                if model_file.exists() and info_file.exists():
                    try:
                        # 모델 정보 로드
                        with open(info_file, 'r', encoding='utf-8') as f:
                            model_info = json.load(f)
                        
                        # 모델 로드
                        model = CategoryAttributeCNN(model_info['num_classes'])
                        model.load_state_dict(torch.load(model_file, map_location=DEVICE))
                        model.to(DEVICE)
                        model.eval()
                        
                        category_models[category][attribute] = {
                            'model': model,
                            'num_classes': model_info['num_classes'],
                            'class_names': model_info['class_names']
                        }
                        
                        print(f"  ✓ {category}_{attribute} 모델 로드 완료")
                        
                    except Exception as e:
                        print(f"  ❌ {category}_{attribute} 모델 로드 실패: {e}")
                        continue
                else:
                    print(f"  ⚠️ {category}_{attribute} 모델 파일 없음")
        
        return category_models
    
    def load_schema(self, schema_path: str):
        """속성 스키마 로드"""
        try:
            schema_df = pd.read_csv(schema_path)
            return schema_df
        except Exception as e:
            print(f"❌ 스키마 로드 실패: {e}")
            return None
    
    def load_clip_model(self):
        """CLIP 모델 로드"""
        try:
            clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(DEVICE)
            clip_processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        except:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        
        clip_model.eval()
        return clip_model, clip_processor
    
    def predict_style(self, image: np.ndarray) -> Dict:
        """전체 이미지 스타일 예측"""
        print("[1/7] 전체 이미지 스타일 예측 중...")
        
        if self.style_model is None:
            return {'style': 'Unknown', 'confidence': 0.0}
        
        # PIL 이미지로 변환
        pil_image = Image.fromarray(image)
        image_tensor = self.transform(pil_image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.style_model(image_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            style_idx = probs.argmax().item()
            confidence = probs[style_idx].item()
            style = self.style_classes[style_idx]
        
        result = {
            'style': style,
            'confidence': confidence
        }
        
        print(f"  - 스타일: {style} ({confidence:.2f})")
        return result
    
    def detect_and_crop_categories(self, image_path: str) -> Dict:
        """YOLO로 카테고리 감지 및 크롭"""
        print("[2/7] YOLO 카테고리 감지 및 크롭 중...")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 열 수 없습니다: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # YOLO Detection 추론
        results = self.yolo_detection_model(image_path, verbose=False)
        
        detected_items = {
            '상의': [],
            '하의': [],
            '아우터': [],
            '원피스': []
        }
        
        # 클래스 이름 매핑
        class_names = ['아우터', '상의', '하의', '원피스']
        category_mapping = {
            'outer': '아우터',
            'top': '상의',
            'bottom': '하의',
            'dress': '원피스'
        }
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # confidence 0.5 이상만 처리
                if confidence >= 0.5 and class_id < len(class_names):
                    class_name_en = class_names[class_id]
                    class_name_ko = category_mapping.get(class_name_en)
                    
                    if class_name_ko:
                        # 바운딩 박스 좌표
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # 이미지 크롭
                        cropped_image = image_rgb[y1:y2, x1:x2]
                        
                        detected_items[class_name_ko].append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'cropped_image': cropped_image
                        })
                        
                        print(f"  - {class_name_ko}: confidence={confidence:.2f}, bbox=({x1},{y1},{x2},{y2})")
        
        # 각 카테고리별로 가장 높은 confidence 선택
        final_items = {}
        for category, items in detected_items.items():
            if items:
                best_item = max(items, key=lambda x: x['confidence'])
                final_items[category] = best_item
                print(f"  ✅ {category} 선택: confidence={best_item['confidence']:.2f}")
        
        return {
            'original': image_rgb,
            'detected_items': final_items,
            'has_상의': '상의' in final_items,
            'has_하의': '하의' in final_items,
            'has_아우터': '아우터' in final_items,
            'has_원피스': '원피스' in final_items
        }
    
    def predict_category_attributes(self, category: str, cropped_image: np.ndarray) -> Dict:
        """특정 카테고리의 속성 예측"""
        print(f"  [3/7] {category} 속성 예측 중...")
        
        if category not in self.category_models:
            return {}
        
        # PIL 이미지로 변환
        pil_image = Image.fromarray(cropped_image)
        image_tensor = self.transform(pil_image).unsqueeze(0).to(DEVICE)
        
        attributes = {}
        
        # 각 속성별 예측
        for attribute, model_info in self.category_models[category].items():
            try:
                model = model_info['model']
                class_names = model_info['class_names']
                
                with torch.no_grad():
                    outputs = model(image_tensor)
                    probs = torch.softmax(outputs, dim=1)[0]
                    pred_idx = probs.argmax().item()
                    confidence = probs[pred_idx].item()
                    predicted_class = class_names[pred_idx]
                    
                    attributes[attribute] = {
                        'value': predicted_class,
                        'confidence': confidence
                    }
                    
                    print(f"    - {attribute}: {predicted_class} ({confidence:.2f})")
                    
            except Exception as e:
                print(f"    ❌ {category}_{attribute} 예측 실패: {e}")
                continue
        
        return attributes
    
    def create_embedding(self, image: np.ndarray) -> np.ndarray:
        """이미지 임베딩 생성"""
        print("[4/7] 이미지 임베딩 생성 중...")
        
        pil_image = Image.fromarray(image)
        
        inputs = self.clip_processor(
            images=pil_image,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)
        
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        embedding = image_features.cpu().numpy()[0]
        print(f"  - 임베딩 차원: {embedding.shape}")
        
        return embedding
    
    def save_to_postgresql(self, user_id: int, image_path: str, 
                          style_result: Dict, detection_result: Dict,
                          category_attributes: Dict, chroma_id: str = None) -> int:
        """PostgreSQL에 저장"""
        print("[5/7] PostgreSQL에 저장 중...")
        
        try:
            self.db_conn.rollback()
        except:
            pass
        
        try:
            with self.db_conn.cursor() as cur:
                # wardrobe_items 삽입
                cur.execute("""
                    INSERT INTO wardrobe_items (
                        user_id, original_image_path, style, style_confidence,
                        has_top, has_bottom, has_outer, has_dress,
                        chroma_embedding_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING item_id
                """, (
                    user_id, 
                    image_path,
                    style_result['style'],
                    style_result['confidence'],
                    detection_result['has_상의'],
                    detection_result['has_하의'],
                    detection_result['has_아우터'],
                    detection_result['has_원피스'],
                    chroma_id
                ))
                
                item_id = cur.fetchone()[0]
                
                # 각 카테고리별 속성 저장
                for category, attributes in category_attributes.items():
                    if attributes:
                        # 카테고리별 테이블에 저장
                        table_name = f"{category.lower()}_attributes"
                        
                        # 속성 값 추출
                        category_val = attributes.get('카테고리', {}).get('value', 'Unknown')
                        color_val = attributes.get('색상', {}).get('value', 'Unknown')
                        fit_val = attributes.get('핏', {}).get('value', 'Unknown')
                        material_val = attributes.get('소재', {}).get('value', 'Unknown')
                        length_val = attributes.get('기장', {}).get('value', 'Unknown')
                        sleeve_length_val = attributes.get('소매기장', {}).get('value', 'Unknown')
                        neckline_val = attributes.get('넥라인', {}).get('value', 'Unknown')
                        print_val = attributes.get('프린트', {}).get('value', 'Unknown')
                        
                        # 신뢰도 추출
                        category_conf = attributes.get('카테고리', {}).get('confidence', 0.0)
                        color_conf = attributes.get('색상', {}).get('confidence', 0.0)
                        fit_conf = attributes.get('핏', {}).get('confidence', 0.0)
                        
                        cur.execute(f"""
                            INSERT INTO {table_name} (
                                item_id, category, color, fit, material, length,
                                sleeve_length, neckline, print_pattern,
                                category_confidence, color_confidence, fit_confidence
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            item_id, category_val, color_val, fit_val, material_val,
                            length_val, sleeve_length_val, neckline_val, print_val,
                            category_conf, color_conf, fit_conf
                        ))
                
                self.db_conn.commit()
                
            print(f"  - 아이템 ID: {item_id}")
            return item_id
            
        except Exception as e:
            try:
                self.db_conn.rollback()
                print(f"  ❌ PostgreSQL 저장 실패 (rollback 완료): {e}")
            except:
                print(f"  ❌ PostgreSQL 저장 실패: {e}")
            raise e
    
    def save_to_chromadb(self, item_id: int, embedding: np.ndarray, 
                        metadata: Dict) -> str:
        """ChromaDB에 저장"""
        print("[6/7] ChromaDB에 저장 중...")
        
        chroma_id = f"item_{item_id}"
        
        # 메타데이터를 문서로 변환
        doc_parts = []
        if metadata.get('style'):
            doc_parts.append(f"스타일: {metadata['style']}")
        
        for category in ['상의', '하의', '아우터', '원피스']:
            if metadata.get(f'{category}_category'):
                doc_parts.append(f"{category}: {metadata[f'{category}_category']}")
        
        document = " | ".join(doc_parts)
        
        self.chroma_collection.add(
            ids=[chroma_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
            documents=[document]
        )
        
        print(f"  - Chroma ID: {chroma_id}")
        return chroma_id
    
    def search_similar(self, embedding: np.ndarray, n_results: int = 5) -> list:
        """유사 아이템 검색"""
        print(f"[7/7] 유사 아이템 검색 중 (Top {n_results})...")
        
        results = self.chroma_collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=n_results
        )
        
        print(f"  - 검색 완료: {len(results['ids'][0])}개")
        return results
    
    def process_image(self, image_path: str, user_id: int) -> Dict:
        """전체 파이프라인 실행"""
        print(f"\n{'='*60}")
        print(f"새로운 패션 파이프라인 시작: {image_path}")
        print(f"{'='*60}\n")
        
        try:
            # 1. 스타일 예측
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 열 수 없습니다: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            style_result = self.predict_style(image_rgb)
            
            # 2. 카테고리 감지 및 크롭
            detection_result = self.detect_and_crop_categories(image_path)
            
            # 3. 각 카테고리별 속성 예측
            category_attributes = {}
            for category, item in detection_result['detected_items'].items():
                cropped_image = item['cropped_image']
                attributes = self.predict_category_attributes(category, cropped_image)
                category_attributes[category] = attributes
            
            # 4. 임베딩 생성
            embedding = self.create_embedding(image_rgb)
            
            # 5. 메타데이터 준비
            metadata = {
                'user_id': str(user_id),
                'style': style_result['style'],
                'style_confidence': style_result['confidence']
            }
            
            for category, attributes in category_attributes.items():
                for attr_name, attr_data in attributes.items():
                    metadata[f'{category}_{attr_name}'] = attr_data['value']
                    metadata[f'{category}_{attr_name}_confidence'] = attr_data['confidence']
            
            # 6. PostgreSQL 저장
            item_id = self.save_to_postgresql(
                user_id, image_path, style_result, detection_result,
                category_attributes, chroma_id=None
            )
            
            # 7. ChromaDB 저장
            chroma_id = self.save_to_chromadb(item_id, embedding, metadata)
            
            # 8. PostgreSQL에 chroma_id 업데이트
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    UPDATE wardrobe_items 
                    SET chroma_embedding_id = %s 
                    WHERE item_id = %s
                """, (chroma_id, item_id))
                self.db_conn.commit()
            
            # 9. 유사 아이템 검색
            similar_items = self.search_similar(embedding, n_results=5)
            
            print(f"\n{'='*60}")
            print(f"✔ 파이프라인 완료!")
            print(f"  - 아이템 ID: {item_id}")
            print(f"  - Chroma ID: {chroma_id}")
            print(f"  - 스타일: {style_result['style']}")
            
            detected_categories = list(detection_result['detected_items'].keys())
            print(f"  - 감지된 의류: {', '.join(detected_categories)}")
            
            for category, attributes in category_attributes.items():
                if attributes:
                    print(f"  - {category}:")
                    for attr_name, attr_data in attributes.items():
                        print(f"    {attr_name}: {attr_data['value']}")
            
            print(f"{'='*60}\n")
            
            return {
                'success': True,
                'item_id': item_id,
                'chroma_id': chroma_id,
                'style_result': style_result,
                'detection_result': detection_result,
                'category_attributes': category_attributes,
                'similar_items': similar_items
            }
            
        except Exception as e:
            print(f"\n✗ 에러 발생: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                if hasattr(self, 'db_conn') and self.db_conn:
                    self.db_conn.rollback()
            except:
                pass
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_image_info(self, item_id: int) -> Dict:
        """이미지 정보 조회 (전체 이미지용)"""
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    SELECT item_id, original_image_path, style, style_confidence,
                           has_top, has_bottom, has_outer, has_dress
                    FROM wardrobe_items 
                    WHERE item_id = %s
                """, (item_id,))
                
                result = cur.fetchone()
                if result:
                    return {
                        'item_id': result[0],
                        'image_path': result[1],
                        'style': result[2],
                        'style_confidence': result[3],
                        'has_top': result[4],
                        'has_bottom': result[5],
                        'has_outer': result[6],
                        'has_dress': result[7]
                    }
                else:
                    return None
                    
        except Exception as e:
            print(f"❌ 이미지 정보 조회 실패: {e}")
            return None
    
    def get_category_info(self, item_id: int, category: str) -> Dict:
        """카테고리별 속성 정보 조회 (크롭된 이미지용)"""
        try:
            table_name = f"{category.lower()}_attributes"
            
            with self.db_conn.cursor() as cur:
                cur.execute(f"""
                    SELECT category, color, fit, material, length,
                           sleeve_length, neckline, print_pattern,
                           category_confidence, color_confidence, fit_confidence
                    FROM {table_name} 
                    WHERE item_id = %s
                """, (item_id,))
                
                result = cur.fetchone()
                if result:
                    return {
                        'category': result[0],
                        'color': result[1],
                        'fit': result[2],
                        'material': result[3],
                        'length': result[4],
                        'sleeve_length': result[5],
                        'neckline': result[6],
                        'print_pattern': result[7],
                        'category_confidence': result[8],
                        'color_confidence': result[9],
                        'fit_confidence': result[10]
                    }
                else:
                    return None
                    
        except Exception as e:
            print(f"❌ {category} 정보 조회 실패: {e}")
            return None
    
    def close(self):
        """연결 종료"""
        if self.db_conn:
            self.db_conn.close()
            print("PostgreSQL 연결 종료")


class CategoryAttributeCNN(nn.Module):
    """카테고리별 속성 분류 CNN 모델"""
    
    def __init__(self, num_classes, pretrained=True):
        super(CategoryAttributeCNN, self).__init__()
        
        # ResNet50 백본 사용
        self.backbone = models.resnet50(pretrained=pretrained)
        num_features = self.backbone.fc.in_features
        
        # 기존 fc 레이어 제거
        self.backbone.fc = nn.Identity()
        
        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def main():
    """테스트용 메인 함수"""
    print("🎯 새로운 패션 파이프라인 테스트")
    
    # 파이프라인 초기화
    pipeline = NewFashionPipeline()
    
    # 테스트 이미지 처리
    test_image_path = "test_image.jpg"  # 실제 이미지 경로로 변경
    user_id = 1
    
    result = pipeline.process_image(test_image_path, user_id)
    
    if result['success']:
        print("✅ 파이프라인 실행 성공!")
    else:
        print("❌ 파이프라인 실행 실패!")
    
    pipeline.close()


if __name__ == "__main__":
    main()

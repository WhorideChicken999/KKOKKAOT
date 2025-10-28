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

# Background Remover 관련 import
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    print("⚠️ rembg가 설치되지 않았습니다. Background Remover 기능을 사용할 수 없습니다.")
    REMBG_AVAILABLE = False

# GPU 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {DEVICE}")


class FashionPipeline:
    """가상옷장 전체 파이프라인 (업데이트된 버전)"""
    
    def __init__(self, 
                style_model_path: str = "D:/kkokkaot/API/pre_trained_weights/k_fashion_best_model.pth",
                yolo_detection_path: str = "D:/kkokkaot/API/pre_trained_weights/yolo_best.pt",
                # 새로운 의류별 모델 경로
                top_model_path: str = "D:/kkokkaot/models/top/best_model.pth",
                bottom_model_path: str = "D:/kkokkaot/models/bottom/best_model.pth",
                outer_model_path: str = "D:/kkokkaot/models/outer/best_model.pth",
                dress_model_path: str = "D:/kkokkaot/models/dress/best_model.pth",
                schema_path: str = "D:/kkokkaot/API/kfashion_attributes_schema.csv",
                yolo_pose_path: str = None,  # 기존 호환성을 위해 유지
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
        
        # 3. 의류별 속성 모델 로드
        print("3. 의류별 속성 모델 로드...")
        self.category_models = self.load_category_models(
            top_model_path, bottom_model_path, outer_model_path, dress_model_path
        )
        
        # 4. 스키마 로드
        print("4. 속성 스키마 로드...")
        self.schema = self.load_schema(schema_path)
        
        # 5. YOLO Pose 모델 (선택적, 기존 호환성을 위해)
        if yolo_pose_path:
            print("5. YOLO Pose 로드...")
            self.yolo_model = YOLO(yolo_pose_path)
        else:
            print("5. YOLO Pose 건너뛰기...")
            self.yolo_model = None
        
        # 6. Background Remover (누끼따기)
        if REMBG_AVAILABLE:
            print("6. Background Remover 초기화...")
            try:
                # Background Remover 세션 초기화 (u2net 모델 사용)
                self.rembg_session = new_session('u2net')
                self.rembg_available = True
                print("✅ Background Remover 초기화 완료!")
            except Exception as e:
                print(f"⚠️ Background Remover 초기화 실패: {e}")
                self.rembg_session = None
                self.rembg_available = False
        else:
            print("6. Background Remover 건너뛰기...")
            self.rembg_session = None
            self.rembg_available = False
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 7. CLIP 모델 (임베딩용)
        print("7. CLIP 모델 로드...")
        self.clip_model, self.clip_processor = self.load_clip_model()
        
        # 8. ChromaDB
        print("8. ChromaDB 연결...")
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        try:
            self.chroma_collection = self.chroma_client.get_collection(name="fashion_collection")
        except:
            self.chroma_collection = self.chroma_client.create_collection(name="fashion_collection")
        
        # 9. PostgreSQL
        print("9. PostgreSQL 연결...")
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
        
        print("\n✓ 모든 모델 로드 완료\n")
    
    def load_style_model(self, model_path: str):
        """스타일 예측 모델 로드"""
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            
            # 체크포인트 구조 확인
            if 'model_state_dict' in checkpoint:
                # 새로운 체크포인트 구조 (학습된 모델)
                model_state_dict = checkpoint['model_state_dict']
                class_to_idx = checkpoint.get('class_to_idx', {})
                
                # 클래스 개수 확인
                num_classes = len(class_to_idx) if class_to_idx else 22
                
                # KFashionModel 커스텀 모델 구조 사용
                model = self._create_kfashion_model(num_classes)
                model.load_state_dict(model_state_dict)
                model.to(DEVICE)
                model.eval()
                
                # 클래스 이름 추출
                if class_to_idx:
                    style_classes = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]
                else:
                    # 기본 스타일 클래스
                    style_classes = [
                        '로맨틱', '페미닌', '섹시', '젠더리스/젠더플루이드', '매스큘린', '톰보이',
                        '히피', '오리엔탈', '웨스턴', '컨트리', '리조트', '모던',
                        '소피스트케이티드', '아방가르드', '펑크', '키치/키덜트', '레트로',
                        '힙합', '클래식', '프레피', '스트리트', '밀리터리', '스포티'
                    ]
                
            else:
                # 기존 구조 (직접 state_dict)
                num_classes = 22
                model = models.resnet50(pretrained=False)
                model.fc = nn.Linear(model.fc.in_features, num_classes)
                model.load_state_dict(checkpoint)
                model.to(DEVICE)
                model.eval()
                
                style_classes = [
                    '로맨틱', '페미닌', '섹시', '젠더리스/젠더플루이드', '매스큘린', '톰보이',
                    '히피', '오리엔탈', '웨스턴', '컨트리', '리조트', '모던',
                    '소피스트케이티드', '아방가르드', '펑크', '키치/키덜트', '레트로',
                    '힙합', '클래식', '프레피', '스트리트', '밀리터리', '스포티'
                ]
            
            print(f"  ✓ 스타일 모델 로드 완료 ({len(style_classes)}개 클래스)")
            return model, style_classes
            
        except Exception as e:
            print(f"❌ 스타일 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def load_category_models(self, top_model_path: str, bottom_model_path: str, 
                           outer_model_path: str, dress_model_path: str):
        """의류별 속성 모델 로드 (새로운 구조)"""
        category_models = {}
        
        # 각 의류별 모델 로드
        model_paths = {
            '상의': top_model_path,
            '하의': bottom_model_path, 
            '아우터': outer_model_path,
            '원피스': dress_model_path
        }
        
        for category, model_path in model_paths.items():
            try:
                if not Path(model_path).exists():
                    print(f"  ⚠️ {category} 모델 파일 없음: {model_path}")
                    continue
                
                # 체크포인트 로드
                checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
                
                # 모델 구조 및 인코더 정보 추출
                encoders = checkpoint.get('encoders', {})
                schema = checkpoint.get('schema', {})
                
                # 각 의류별 속성 정의
                if category == '상의':
                    attributes = ['category', 'color', 'material', 'print', 'fit', 'style', 'sleeve']
                elif category == '하의':
                    attributes = ['category', 'color', 'material', 'print', 'fit', 'style', 'length']
                elif category == '아우터':
                    attributes = ['category', 'color', 'material', 'print', 'fit', 'style', 'sleeve']
                elif category == '원피스':
                    attributes = ['category', 'color', 'material', 'print', 'style']
                
                category_models[category] = {
                    'model': None,  # 실제 모델은 predict_category_attributes에서 로드
                    'checkpoint': checkpoint,
                    'encoders': encoders,
                    'schema': schema,
                    'attributes': attributes
                }
                
                print(f"  ✓ {category} 모델 로드 완료 ({len(attributes)}개 속성)")
                
            except Exception as e:
                print(f"  ❌ {category} 모델 로드 실패: {e}")
                continue
        
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
    
    def predict_category_attributes(self, category: str, cropped_image: np.ndarray) -> Dict:
        """특정 카테고리의 속성 예측 (새로운 구조)"""
        print(f"  [3/7] {category} 속성 예측 중...")
        
        if category not in self.category_models:
            return {}
        
        # PIL 이미지로 변환
        pil_image = Image.fromarray(cropped_image)
        image_tensor = self.transform(pil_image).unsqueeze(0).to(DEVICE)
        
        attributes = {}
        
        try:
            # 체크포인트에서 모델 정보 가져오기
            model_info = self.category_models[category]
            checkpoint = model_info['checkpoint']
            encoders = model_info['encoders']
            attributes_list = model_info['attributes']
            
            # 모델 구조 재구성 (학습 시와 동일한 구조)
            if category == '상의':
                model = self._create_top_model(encoders)
            elif category == '하의':
                model = self._create_bottom_model(encoders)
            elif category == '아우터':
                model = self._create_outer_model(encoders)
            elif category == '원피스':
                model = self._create_dress_model(encoders)
            
            # 모델 가중치 로드
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)
            model.eval()
            
            # 예측 수행
            with torch.no_grad():
                outputs = model(image_tensor)
                
                # 각 속성별 예측 결과 처리
                for attr in attributes_list:
                    if attr in outputs:
                        attr_output = outputs[attr]
                        probs = torch.softmax(attr_output, dim=1)[0]
                        pred_idx = probs.argmax().item()
                        confidence = probs[pred_idx].item()
                        
                        # 인코더로 디코딩
                        predicted_class = encoders[attr].inverse_transform([pred_idx])[0]
                        
                        attributes[attr] = {
                            'value': predicted_class,
                            'confidence': confidence
                        }
                        
                        print(f"    - {attr}: {predicted_class} ({confidence:.2f})")
            
        except Exception as e:
            print(f"    ❌ {category} 속성 예측 실패: {e}")
            import traceback
            traceback.print_exc()
        
        return attributes
    
    def _create_top_model(self, encoders):
        """상의 모델 생성"""
        from torchvision import models
        import torch.nn as nn
        
        class TopFashionModel(nn.Module):
            def __init__(self, num_category, num_color, num_material, num_print, num_fit, num_style, num_sleeve):
                super(TopFashionModel, self).__init__()
                
                # EfficientNet-B0를 백본으로 사용
                self.backbone = models.efficientnet_b0(pretrained=True)
                
                # 특징 추출기 (마지막 분류층 제거)
                self.features = nn.Sequential(*list(self.backbone.children())[:-1])
                
                # 특징 차원
                feature_dim = 1280
                
                # 공유 특징 변환층
                self.shared_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(feature_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                
                # 각 태스크별 헤드
                self.category_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_category)
                )
                
                self.color_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_color)
                )
                
                self.material_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_material)
                )
                
                self.print_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_print)
                )
                
                self.fit_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_fit)
                )
                
                self.style_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_style)
                )
                
                self.sleeve_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_sleeve)
                )
            
            def forward(self, x):
                # 공유 특징 추출
                features = self.features(x)
                shared_features = self.shared_fc(features)
                
                # 각 태스크별 예측
                category_out = self.category_head(shared_features)
                color_out = self.color_head(shared_features)
                material_out = self.material_head(shared_features)
                print_out = self.print_head(shared_features)
                fit_out = self.fit_head(shared_features)
                style_out = self.style_head(shared_features)
                sleeve_out = self.sleeve_head(shared_features)
                
                return {
                    'category': category_out,
                    'color': color_out,
                    'material': material_out,
                    'print': print_out,
                    'fit': fit_out,
                    'style': style_out,
                    'sleeve': sleeve_out
                }
        
        return TopFashionModel(
            num_category=len(encoders['category'].classes_),
            num_color=len(encoders['color'].classes_),
            num_material=len(encoders['material'].classes_),
            num_print=len(encoders['print'].classes_),
            num_fit=len(encoders['fit'].classes_),
            num_style=len(encoders['style'].classes_),
            num_sleeve=len(encoders['sleeve'].classes_)
        )
    
    def _create_bottom_model(self, encoders):
        """하의 모델 생성"""
        from torchvision import models
        import torch.nn as nn
        
        class BottomFashionModel(nn.Module):
            def __init__(self, num_category, num_color, num_material, num_print, num_fit, num_style, num_length):
                super(BottomFashionModel, self).__init__()
                
                # EfficientNet-B0를 백본으로 사용
                self.backbone = models.efficientnet_b0(pretrained=True)
                
                # 특징 추출기 (마지막 분류층 제거)
                self.features = nn.Sequential(*list(self.backbone.children())[:-1])
                
                # 특징 차원
                feature_dim = 1280
                
                # 공유 특징 변환층
                self.shared_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(feature_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                
                # 각 태스크별 헤드
                self.category_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_category)
                )
                
                self.color_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_color)
                )
                
                self.material_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_material)
                )
                
                self.print_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_print)
                )
                
                self.fit_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_fit)
                )
                
                self.style_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_style)
                )
                
                self.length_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_length)
                )
            
            def forward(self, x):
                # 공유 특징 추출
                features = self.features(x)
                shared_features = self.shared_fc(features)
                
                # 각 태스크별 예측
                category_out = self.category_head(shared_features)
                color_out = self.color_head(shared_features)
                material_out = self.material_head(shared_features)
                print_out = self.print_head(shared_features)
                fit_out = self.fit_head(shared_features)
                style_out = self.style_head(shared_features)
                length_out = self.length_head(shared_features)
                
                return {
                    'category': category_out,
                    'color': color_out,
                    'material': material_out,
                    'print': print_out,
                    'fit': fit_out,
                    'style': style_out,
                    'length': length_out
                }
        
        return BottomFashionModel(
            num_category=len(encoders['category'].classes_),
            num_color=len(encoders['color'].classes_),
            num_material=len(encoders['material'].classes_),
            num_print=len(encoders['print'].classes_),
            num_fit=len(encoders['fit'].classes_),
            num_style=len(encoders['style'].classes_),
            num_length=len(encoders['length'].classes_)
        )
    
    def _create_outer_model(self, encoders):
        """아우터 모델 생성"""
        from torchvision import models
        import torch.nn as nn
        
        class OuterFashionModel(nn.Module):
            def __init__(self, num_category, num_color, num_material, num_print, num_fit, num_style, num_sleeve):
                super(OuterFashionModel, self).__init__()
                
                # EfficientNet-B0를 백본으로 사용
                self.backbone = models.efficientnet_b0(pretrained=True)
                
                # 특징 추출기 (마지막 분류층 제거)
                self.features = nn.Sequential(*list(self.backbone.children())[:-1])
                
                # 특징 차원
                feature_dim = 1280
                
                # 공유 특징 변환층
                self.shared_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(feature_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                
                # 각 태스크별 헤드
                self.category_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_category)
                )
                
                self.color_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_color)
                )
                
                self.material_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_material)
                )
                
                self.print_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_print)
                )
                
                self.fit_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_fit)
                )
                
                self.style_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_style)
                )
                
                self.sleeve_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_sleeve)
                )
            
            def forward(self, x):
                # 공유 특징 추출
                features = self.features(x)
                shared_features = self.shared_fc(features)
                
                # 각 태스크별 예측
                category_out = self.category_head(shared_features)
                color_out = self.color_head(shared_features)
                material_out = self.material_head(shared_features)
                print_out = self.print_head(shared_features)
                fit_out = self.fit_head(shared_features)
                style_out = self.style_head(shared_features)
                sleeve_out = self.sleeve_head(shared_features)
                
                return {
                    'category': category_out,
                    'color': color_out,
                    'material': material_out,
                    'print': print_out,
                    'fit': fit_out,
                    'style': style_out,
                    'sleeve': sleeve_out
                }
        
        return OuterFashionModel(
            num_category=len(encoders['category'].classes_),
            num_color=len(encoders['color'].classes_),
            num_material=len(encoders['material'].classes_),
            num_print=len(encoders['print'].classes_),
            num_fit=len(encoders['fit'].classes_),
            num_style=len(encoders['style'].classes_),
            num_sleeve=len(encoders['sleeve'].classes_)
        )
    
    def _create_dress_model(self, encoders):
        """드레스 모델 생성"""
        from torchvision import models
        import torch.nn as nn
        
        class DressFashionModel(nn.Module):
            def __init__(self, num_category, num_color, num_material, num_print, num_style):
                super(DressFashionModel, self).__init__()
                
                # EfficientNet-B0를 백본으로 사용
                self.backbone = models.efficientnet_b0(pretrained=True)
                
                # 특징 추출기 (마지막 분류층 제거)
                self.features = nn.Sequential(*list(self.backbone.children())[:-1])
                
                # 특징 차원
                feature_dim = 1280
                
                # 공유 특징 변환층
                self.shared_fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(feature_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                )
                
                # 각 태스크별 헤드
                self.category_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_category)
                )
                
                self.color_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_color)
                )
                
                self.material_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_material)
                )
                
                self.print_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_print)
                )
                
                self.style_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_style)
                )
            
            def forward(self, x):
                # 공유 특징 추출
                features = self.features(x)
                shared_features = self.shared_fc(features)
                
                # 각 태스크별 예측
                category_out = self.category_head(shared_features)
                color_out = self.color_head(shared_features)
                material_out = self.material_head(shared_features)
                print_out = self.print_head(shared_features)
                style_out = self.style_head(shared_features)
                
                return {
                    'category': category_out,
                    'color': color_out,
                    'material': material_out,
                    'print': print_out,
                    'style': style_out
                }
        
        return DressFashionModel(
            num_category=len(encoders['category'].classes_),
            num_color=len(encoders['color'].classes_),
            num_material=len(encoders['material'].classes_),
            num_print=len(encoders['print'].classes_),
            num_style=len(encoders['style'].classes_)
        )
    
    def _create_kfashion_model(self, num_classes):
        """KFashionModel 커스텀 모델 생성"""
        from torchvision import models
        import torch.nn as nn
        
        class KFashionModel(nn.Module):
            def __init__(self, num_classes):
                super(KFashionModel, self).__init__()
                
                # EfficientNet-B0 백본
                self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
                
                # 백본의 일부 레이어 고정 (Fine-tuning)
                # EfficientNet의 features 부분에서 앞쪽 레이어들 고정
                for i, param in enumerate(self.backbone.features.parameters()):
                    if i < 100:  # 앞쪽 파라미터들 고정
                        param.requires_grad = False
                
                # Classifier 교체
                in_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Sequential(
                    nn.BatchNorm1d(in_features),
                    nn.Dropout(0.5),
                    nn.Linear(in_features, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                return self.backbone(x)
        
        return KFashionModel(num_classes)
    
    def reconnect_db(self):
        """데이터베이스 연결 재시도"""
        try:
            if hasattr(self, 'db_conn') and self.db_conn:
                self.db_conn.close()
        except:
            pass
        
        try:
            self.db_conn = psycopg2.connect(**self.db_config)
            print("✅ 데이터베이스 재연결 성공")
        except Exception as e:
            print(f"❌ 데이터베이스 재연결 실패: {e}")
            raise
    
    def detect_and_crop_categories(self, image_path: str) -> Dict:
        """YOLO로 카테고리 감지 및 크롭"""
        print("[2/7] YOLO 카테고리 감지 및 크롭 중...")
        
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 열 수 없습니다: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"  📏 이미지 크기: {image_rgb.shape}")
        
        # YOLO Detection 추론
        try:
            results = self.yolo_detection_model(image_path, verbose=False)
            print(f"  🔍 YOLO 결과: {len(results[0].boxes)}개 박스 감지")
            
            # YOLO 결과 상세 정보 출력
            if len(results[0].boxes) > 0:
                print(f"  📊 박스 정보:")
                for i, box in enumerate(results[0].boxes):
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    print(f"    박스 {i}: class_id={class_id}, confidence={confidence:.3f}, bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
            else:
                print("  ❌ YOLO가 아무것도 감지하지 못했습니다!")
                print("  💡 가능한 원인:")
                print("    - 이미지에 의류가 명확하지 않음")
                print("    - YOLO 모델이 해당 이미지를 인식하지 못함")
                print("    - confidence threshold가 너무 높음")
                
        except Exception as e:
            print(f"  ❌ YOLO 추론 실패: {e}")
            return {
                'original': image_rgb,
                'detected_items': {},
                'has_상의': False,
                'has_하의': False,
                'has_아우터': False,
                'has_원피스': False
            }
        
        detected_items = {
            '상의': [],
            '하의': [],
            '아우터': [],
            '원피스': []
        }
        
        # 클래스 이름 매핑 (YOLO 모델의 실제 클래스 순서)
        class_names = ['outer', 'top', 'bottom', 'dress']  # 영어로 수정
        category_mapping = {
            'outer': '아우터',
            'top': '상의',
            'bottom': '하의',
            'dress': '원피스'
        }
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            print(f"  📦 감지된 박스 개수: {len(boxes)}")
            
            for i, box in enumerate(boxes):
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                print(f"    박스 {i}: class_id={class_id}, confidence={confidence:.3f}")
                
                # confidence 0.3으로 낮춤 (더 많은 감지 허용)
                if confidence >= 0.3 and class_id < len(class_names):
                    class_name_en = class_names[class_id]
                    class_name_ko = category_mapping.get(class_name_en)
                    
                    print(f"    → 클래스: {class_name_en} → {class_name_ko}")
                    
                    if class_name_ko:
                        # 바운딩 박스 좌표
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        print(f"    → 바운딩 박스: ({x1},{y1},{x2},{y2})")
                        
                        # 바운딩 박스 유효성 검사
                        if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and x2 <= image_rgb.shape[1] and y2 <= image_rgb.shape[0]:
                            # 이미지 크롭
                            cropped_image = image_rgb[y1:y2, x1:x2]
                            
                            if cropped_image.size > 0:
                                detected_items[class_name_ko].append({
                                    'bbox': (x1, y1, x2, y2),
                                    'confidence': confidence,
                                    'cropped_image': cropped_image
                                })
                                
                                print(f"  ✅ {class_name_ko}: confidence={confidence:.2f}, 크롭 크기={cropped_image.shape}")
                            else:
                                print(f"  ❌ {class_name_ko}: 크롭된 이미지가 비어있음")
                        else:
                            print(f"  ❌ {class_name_ko}: 유효하지 않은 바운딩 박스")
                    else:
                        print(f"  ❌ 클래스 매핑 실패: {class_name_en}")
                else:
                    print(f"  ❌ 박스 {i}: confidence={confidence:.3f} < 0.3 또는 class_id={class_id} >= {len(class_names)}")
        else:
            print("  ❌ 감지된 박스가 없습니다!")
        
        # 각 카테고리별로 가장 높은 confidence 선택
        final_items = {}
        for category, items in detected_items.items():
            if items:
                best_item = max(items, key=lambda x: x['confidence'])
                final_items[category] = best_item
                print(f"  ✅ {category} 선택: confidence={best_item['confidence']:.2f}")
        
        # YOLO 감지 실패 시 임시 해결책: 전체 이미지를 모든 카테고리로 사용
        if not final_items:
            print("  ⚠️ YOLO 감지 실패 - 전체 이미지를 모든 카테고리로 사용")
            final_items = {
                '상의': {
                    'bbox': (0, 0, image_rgb.shape[1], image_rgb.shape[0]),
                    'confidence': 0.5,
                    'cropped_image': image_rgb
                },
                '하의': {
                    'bbox': (0, 0, image_rgb.shape[1], image_rgb.shape[0]),
                    'confidence': 0.5,
                    'cropped_image': image_rgb
                }
            }
            print("  🔧 임시 해결책 적용: 상의, 하의로 분류")
        
        return {
            'original': image_rgb,
            'detected_items': final_items,
            'has_상의': '상의' in final_items,
            'has_하의': '하의' in final_items,
            'has_아우터': '아우터' in final_items,
            'has_원피스': '원피스' in final_items
        }

    
    
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
                
                # 각 카테고리별 속성 저장 (새로운 구조)
                for category, attributes in category_attributes.items():
                    if attributes:
                        # 카테고리별 테이블에 저장 (영어 테이블명 사용)
                        category_mapping = {
                            '상의': 'top_attributes_new',
                            '하의': 'bottom_attributes_new', 
                            '아우터': 'outer_attributes_new',
                            '원피스': 'dress_attributes_new'
                        }
                        table_name = category_mapping.get(category, f"{category.lower()}_attributes")
                        
                        print(f"  💾 {category} 속성 저장 중... (테이블: {table_name})")
                        print(f"    - 속성 개수: {len(attributes)}")
                        for attr_name, attr_data in attributes.items():
                            print(f"    - {attr_name}: {attr_data.get('value', 'Unknown')} ({attr_data.get('confidence', 0.0):.2f})")
                        
                        # 기본 속성 값 추출 (모든 카테고리 공통)
                        category_val = attributes.get('category', {}).get('value', 'Unknown')
                        color_val = attributes.get('color', {}).get('value', 'Unknown')
                        material_val = attributes.get('material', {}).get('value', 'Unknown')
                        print_val = attributes.get('print', {}).get('value', 'Unknown')
                        style_val = attributes.get('style', {}).get('value', 'Unknown')
                        
                        # 카테고리별 특수 속성
                        fit_val = attributes.get('fit', {}).get('value', 'Unknown')
                        sleeve_val = attributes.get('sleeve', {}).get('value', 'Unknown')
                        length_val = attributes.get('length', {}).get('value', 'Unknown')
                        
                        # 신뢰도 추출
                        category_conf = attributes.get('category', {}).get('confidence', 0.0)
                        color_conf = attributes.get('color', {}).get('confidence', 0.0)
                        fit_conf = attributes.get('fit', {}).get('confidence', 0.0)
                        
                        # 카테고리별로 다른 필드 저장
                        if category == '상의':
                            cur.execute(f"""
                                INSERT INTO {table_name} (
                                    item_id, category, color, fit, material, print_pattern, style, sleeve_length,
                                    category_confidence, color_confidence, fit_confidence
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                item_id, category_val, color_val, fit_val, material_val, print_val, style_val, sleeve_val,
                                category_conf, color_conf, fit_conf
                            ))
                        elif category == '하의':
                            cur.execute(f"""
                                INSERT INTO {table_name} (
                                    item_id, category, color, fit, material, print_pattern, style, length,
                                    category_confidence, color_confidence, fit_confidence
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                item_id, category_val, color_val, fit_val, material_val, print_val, style_val, length_val,
                                category_conf, color_conf, fit_conf
                            ))
                        elif category == '아우터':
                            print(f"    📝 아우터 INSERT 실행: {table_name}")
                            print(f"    📊 값들: item_id={item_id}, category={category_val}, color={color_val}, material={material_val}")
                            try:
                                cur.execute(f"""
                                    INSERT INTO {table_name} (
                                        item_id, category, color, fit, material, print_pattern, style, sleeve_length,
                                        category_confidence, color_confidence, fit_confidence
                                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                """, (
                                    item_id, category_val, color_val, fit_val, material_val, print_val, style_val, sleeve_val,
                                    category_conf, color_conf, fit_conf
                                ))
                                print(f"    ✅ 아우터 INSERT 성공!")
                            except Exception as e:
                                print(f"    ❌ 아우터 INSERT 실패: {e}")
                                raise e
                        elif category == '원피스':
                            cur.execute(f"""
                                INSERT INTO {table_name} (
                                    item_id, category, color, material, print_pattern, style,
                                    category_confidence, color_confidence
                                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                item_id, category_val, color_val, material_val, print_val, style_val,
                                category_conf, color_conf
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
        
        for category_en, category_ko in [('top', '상의'), ('bottom', '하의'), ('outer', '아우터'), ('dress', '원피스')]:
            if metadata.get(f'{category_en}_category'):
                doc_parts.append(f"{category_ko}: {metadata[f'{category_en}_category']}")
        
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
    
    def remove_background_with_rembg(self, image: np.ndarray) -> np.ndarray:
        """
        Background Remover를 사용하여 배경 제거
        
        Args:
            image: 입력 이미지 (RGB)
        
        Returns:
            transparent_image: 투명 배경 이미지 (RGBA)
        """
        if not self.rembg_available:
            print("⚠️ Background Remover가 사용 불가능합니다.")
            return None
        
        try:
            # PIL Image로 변환
            pil_image = Image.fromarray(image)
            
            # Background Remover로 배경 제거
            transparent_image = remove(pil_image, session=self.rembg_session)
            
            # numpy array로 변환
            transparent_array = np.array(transparent_image)
            
            return transparent_array
            
        except Exception as e:
            print(f"❌ Background Remover 처리 실패: {e}")
            return None
    
    
    
    def process_image(self, image_path: str, user_id: int, 
                    save_separated_images: bool = False) -> Dict:
        """전체 파이프라인 실행"""
        
        print(f"\n{'='*60}")
        print(f"파이프라인 시작: {image_path}")
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
                    # 카테고리명을 영어로 변환
                    category_en = {'상의': 'top', '하의': 'bottom', '아우터': 'outer', '원피스': 'dress'}[category]
                    metadata[f'{category_en}_{attr_name}'] = attr_data['value']
                    metadata[f'{category_en}_{attr_name}_confidence'] = attr_data['confidence']
            
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
            
            # 10. 분리된 이미지 저장 (기존 호환성 유지)
            if save_separated_images:
                base_dir = Path("./processed_images") / f"user_{user_id}"
                folders = {
                    'full': base_dir / 'full',
                    'top': base_dir / 'top',
                    'bottom': base_dir / 'bottom',
                    'outer': base_dir / 'outer',
                    'dress': base_dir / 'dress'
                }
                
                for folder in folders.values():
                    folder.mkdir(parents=True, exist_ok=True)
                
                # 전체 이미지 저장
                full_path = folders['full'] / f"item_{item_id}_full.jpg"
                Image.fromarray(detection_result['original']).save(full_path)
                print(f"  ✅ 전체 이미지 저장: {full_path}")
                
                # 각 카테고리별 이미지 저장
                category_mapping = {
                    '상의': 'top',
                    '하의': 'bottom', 
                    '아우터': 'outer',
                    '원피스': 'dress'
                }
                
                for category_ko, category_en in category_mapping.items():
                    if category_ko in detection_result['detected_items']:
                        image_path_save = folders[category_en] / f"item_{item_id}_{category_en}.jpg"
                        Image.fromarray(detection_result['detected_items'][category_ko]['cropped_image']).save(image_path_save)
                        print(f"  ✅ {category_ko} 저장: {image_path_save}")
                
            
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
            
            # ✅ 트랜잭션 rollback (추가)
            try:
                if hasattr(self, 'db_conn') and self.db_conn:
                    self.db_conn.rollback()
            except:
                pass
            
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_similar_items(self, image_path: str, n_results: int = 5) -> list:
        """주어진 이미지와 유사한 아이템을 검색"""
        try:
            # 1. 이미지 임베딩 생성
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지를 찾을 수 없습니다: {image_path}")
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            embedding = self.create_embedding(image_rgb)
            
            # 2. ChromaDB에서 유사 아이템 검색
            search_results = self.search_similar(embedding, n_results)
            
            # 3. 결과 정리
            item_ids = [int(id.replace("item_", "")) for id in search_results['ids'][0]]
            distances = search_results['distances'][0]
            
            results = []
            for i, item_id in enumerate(item_ids):
                results.append({
                    'item_id': item_id,
                    'distance': distances[i]
                })
                
            return results
        
        except Exception as e:
            print(f"❌ 유사 아이템 검색 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def close(self):
        """연결 종료"""
        if self.db_conn:
            self.db_conn.close()
            print("PostgreSQL 연결 종료")


# MultiTaskFashionModel 정의
class MultiTaskFashionModel(nn.Module):
    """Multi-task 패션 속성 예측 모델"""
    
    def __init__(self, num_categories, num_colors, num_fits, num_materials):
        super().__init__()
        
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.category_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_categories)
        )
        
        self.color_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_colors)
        )
        
        self.fit_head = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_fits)
        )
        
        self.material_head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_materials)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        
        category_out = self.category_head(features)
        color_out = self.color_head(features)
        fit_out = self.fit_head(features)
        material_out = self.material_head(features)
        
        return category_out, color_out, fit_out, material_out


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
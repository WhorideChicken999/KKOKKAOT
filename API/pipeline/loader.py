"""
모델 로더
- 모든 AI 모델을 로드하고 관리
"""
import torch
from pathlib import Path
from ultralytics import YOLO
from .models import GenderClassifier, StyleClassifier, AttributeClassifier, STYLE_CLASSES, GENDER_CLASSES


class ModelLoader:
    """모든 AI 모델 로딩 및 관리"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ 사용 디바이스: {self.device}")
        
        # 모델 저장소
        self.gender_model = None
        self.style_model = None
        self.yolo_model = None
        self.attribute_models = {}
    
    def load_gender_model(self, model_path: str = None):
        """성별 예측 모델 로드"""
        print("\n1️⃣ 성별 예측 모델 로딩...")
        
        try:
            self.gender_model = GenderClassifier().to(self.device)
            
            if model_path and Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                if 'model_state_dict' in checkpoint:
                    self.gender_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.gender_model.load_state_dict(checkpoint)
                
                print(f"  ✅ 가중치 로드 완료: {model_path}")
            else:
                print(f"  ⚠️ 가중치 파일 없음 (랜덤 초기화 상태)")
                print(f"  💡 나중에 학습된 가중치를 로드하세요")
            
            self.gender_model.eval()
            print(f"  ✅ 성별 모델 준비 완료 (클래스: {GENDER_CLASSES})")
            return True
            
        except Exception as e:
            print(f"  ❌ 성별 모델 로드 실패: {e}")
            return False
    
    def load_style_model(self, model_path: str):
        """스타일 예측 모델 로드"""
        print("\n2️⃣ 스타일 예측 모델 로딩...")
        
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"모델 파일 없음: {model_path}")
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 체크포인트 구조 확인
            if 'model_state_dict' in checkpoint:
                num_classes = len(checkpoint.get('class_to_idx', STYLE_CLASSES))
                self.style_model = StyleClassifier(num_classes=num_classes).to(self.device)
                self.style_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.style_model = StyleClassifier(num_classes=22).to(self.device)
                self.style_model.load_state_dict(checkpoint)
            
            self.style_model.eval()
            print(f"  ✅ 스타일 모델 로드 완료 ({len(STYLE_CLASSES)}개 클래스)")
            return True
            
        except Exception as e:
            print(f"  ❌ 스타일 모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_yolo_model(self, model_path: str):
        """YOLO Detection 모델 로드"""
        print("\n3️⃣ YOLO Detection 모델 로딩...")
        
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"모델 파일 없음: {model_path}")
            
            self.yolo_model = YOLO(model_path)
            print(f"  ✅ YOLO 모델 로드 완료")
            return True
            
        except Exception as e:
            print(f"  ❌ YOLO 모델 로드 실패: {e}")
            return False
    
    def load_attribute_models(self, 
                            top_path: str,
                            bottom_path: str,
                            outer_path: str,
                            dress_path: str):
        """속성 예측 모델 로드 (상의/하의/아우터/원피스)"""
        print("\n4️⃣ 속성 예측 모델 로딩...")
        
        model_configs = {
            'top': {
                'path': top_path,
                'attributes': ['category', 'color', 'material', 'print', 'fit', 'style', 'sleeve', 'gender']
            },
            'bottom': {
                'path': bottom_path,
                'attributes': ['category', 'color', 'material', 'print', 'fit', 'style', 'length', 'gender']
            },
            'outer': {
                'path': outer_path,
                'attributes': ['category', 'color', 'material', 'print', 'fit', 'style', 'sleeve', 'gender']
            },
            'dress': {
                'path': dress_path,
                'attributes': ['category', 'color', 'material', 'print', 'style', 'gender']
            }
        }
        
        success_count = 0
        for category, config in model_configs.items():
            try:
                model_path = config['path']
                
                if not Path(model_path).exists():
                    print(f"  ⚠️ {category} 모델 파일 없음: {model_path}")
                    continue
                
                # 체크포인트 로드
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # 인코더 정보 추출
                encoders = checkpoint.get('encoders', {})
                
                if not encoders:
                    print(f"  ⚠️ {category} 모델에 인코더 정보 없음")
                    continue
                
                # 속성 차원 계산
                attribute_dims = {}
                for attr in config['attributes']:
                    if attr in encoders:
                        attribute_dims[attr] = len(encoders[attr].classes_)
                
                # 모델 생성 및 가중치 로드
                model = AttributeClassifier(attribute_dims).to(self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()
                
                self.attribute_models[category] = {
                    'model': model,
                    'encoders': encoders,
                    'attributes': config['attributes']
                }
                
                print(f"  ✅ {category} 모델 로드 완료 ({len(config['attributes'])}개 속성)")
                success_count += 1
                
            except Exception as e:
                print(f"  ❌ {category} 모델 로드 실패: {e}")
        
        return success_count > 0
    
    def load_all(self, 
                gender_path: str = None,
                style_path: str = "D:/kkokkaot/API/pre_trained_weights/k_fashion_final_model_1019.pth",
                yolo_path: str = "D:/kkokkaot/API/pre_trained_weights/yolo_best.pt",
                top_path: str = "D:/kkokkaot/API/pre_trained_weights/top_best_model.pth",
                bottom_path: str = "D:/kkokkaot/API/pre_trained_weights/bottom_best_model.pth",
                outer_path: str = "D:/kkokkaot/API/pre_trained_weights/outer_best_model.pth",
                dress_path: str = "D:/kkokkaot/API/pre_trained_weights/dress_best_model.pth"):
        """모든 모델 한 번에 로드"""
        
        print("\n" + "="*60)
        print("🚀 AI 모델 로딩 시작")
        print("="*60)
        
        results = {
            'gender': self.load_gender_model(gender_path),
            'style': self.load_style_model(style_path),
            'yolo': self.load_yolo_model(yolo_path),
            'attributes': self.load_attribute_models(top_path, bottom_path, outer_path, dress_path)
        }
        
        print("\n" + "="*60)
        print("📊 모델 로딩 결과")
        print("="*60)
        for model_name, success in results.items():
            status = "✅ 성공" if success else "❌ 실패"
            print(f"  {model_name:12s}: {status}")
        print("="*60 + "\n")
        
        return all(results.values())


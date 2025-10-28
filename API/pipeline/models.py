"""
AI 모델 정의
- 성별 예측 모델
- 스타일 예측 모델
- 속성 예측 모델
"""
import torch
import torch.nn as nn
from torchvision import models


class GenderClassifier(nn.Module):
    """
    성별 분류 모델 (male/female)
    EfficientNet-B0 기반
    """
    def __init__(self):
        super(GenderClassifier, self).__init__()
        
        # EfficientNet-B0 백본
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # 분류기 교체 (2개 클래스: male, female)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)  # male, female
        )
    
    def forward(self, x):
        return self.backbone(x)


class StyleClassifier(nn.Module):
    """
    스타일 분류 모델 (22개 스타일)
    EfficientNet-B0 기반
    """
    def __init__(self, num_classes=22):
        super(StyleClassifier, self).__init__()
        
        # EfficientNet-B0 백본
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
        # 백본의 일부 레이어 고정 (Fine-tuning)
        for i, param in enumerate(self.backbone.features.parameters()):
            if i < 100:  # 앞쪽 파라미터들 고정
                param.requires_grad = False
        
        # 분류기 교체
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


class AttributeClassifier(nn.Module):
    """
    속성 분류 모델 (Multi-task)
    상의/하의/아우터/원피스 속성 예측
    EfficientNet-B0 기반
    """
    def __init__(self, attribute_dims: dict):
        """
        Args:
            attribute_dims: {'category': 10, 'color': 20, 'fit': 5, ...}
        """
        super(AttributeClassifier, self).__init__()
        
        # EfficientNet-B0 백본
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        
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
        
        # 각 속성별 헤드 (기존 가중치 호환을 위해 {attr}_head 형식으로 정의)
        self.attribute_names = list(attribute_dims.keys())
        for attr_name, num_classes in attribute_dims.items():
            head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
            # 기존 가중치는 'category_head', 'color_head' 형식
            setattr(self, f'{attr_name}_head', head)
    
    def forward(self, x):
        # 공유 특징 추출
        features = self.features(x)
        shared_features = self.shared_fc(features)
        
        # 각 속성별 예측
        outputs = {}
        for attr_name in self.attribute_names:
            head = getattr(self, f'{attr_name}_head')
            outputs[attr_name] = head(shared_features)
        
        return outputs


# 스타일 클래스 목록
STYLE_CLASSES = [
    '로맨틱', '페미닌', '섹시', '젠더리스/젠더플루이드', '매스큘린', '톰보이',
    '히피', '오리엔탈', '웨스턴', '컨트리', '리조트', '모던',
    '소피스트케이티드', '아방가르드', '펑크', '키치/키덜트', '레트로',
    '힙합', '클래식', '프레피', '스트리트', '밀리터리', '스포티'
]

# 성별 클래스 목록
GENDER_CLASSES = ['male', 'female']


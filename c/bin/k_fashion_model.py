"""
K-Fashion 이미지 분류 모델
EfficientNet-B0 기반 전이학습 모델
"""

import torch
import torch.nn as nn
from torchvision import models


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


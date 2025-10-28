import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 설정
IMG_SIZE = 224
MODEL_PATH = 'k_fashion_best_model.pth'  # 또는 'k_fashion_final_model.pth'

# 카테고리 (학습시 순서와 동일하게)
TRAIN_DIR = r'D:\K-Fashion_images\Training'
categories = sorted(os.listdir(TRAIN_DIR))

print("=" * 60)
print("K-Fashion 이미지 분류 예측 (PyTorch)")
print("=" * 60)
print(f"\n카테고리 수: {len(categories)}")
print(f"카테고리: {categories}\n")

# 모델 클래스 정의 (학습 코드와 동일)
class KFashionModel(nn.Module):
    def __init__(self, num_classes):
        super(KFashionModel, self).__init__()
        
        self.backbone = models.efficientnet_b0(weights=None)
        
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

# 모델 로드
print(f"모델 로딩 중: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=device)
num_classes = len(categories)

model = KFashionModel(num_classes).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("모델 로드 완료!\n")

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])


def predict_image(image_path, show_top_n=5):
    """
    이미지 예측 함수
    
    Args:
        image_path: 예측할 이미지 경로
        show_top_n: 상위 N개 예측 결과 표시
    """
    # 이미지 로드 및 전처리
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 예측
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predictions = probabilities.cpu().numpy()
    
    # 상위 N개 결과
    top_indices = np.argsort(predictions)[::-1][:show_top_n]
    
    # 결과 출력
    print(f"\n이미지: {os.path.basename(image_path)}")
    print("-" * 50)
    for i, idx in enumerate(top_indices, 1):
        category = categories[idx]
        probability = predictions[idx] * 100
        print(f"{i}. {category:20s} : {probability:6.2f}%")
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 원본 이미지
    ax1.imshow(img)
    ax1.set_title(f'원본 이미지\n예측: {categories[top_indices[0]]}', 
                  fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 예측 확률 막대 그래프
    top_categories = [categories[i] for i in top_indices]
    top_probs = [predictions[i] * 100 for i in top_indices]
    
    colors = ['#FF6B6B' if i == 0 else '#4ECDC4' for i in range(show_top_n)]
    bars = ax2.barh(range(show_top_n), top_probs, color=colors)
    ax2.set_yticks(range(show_top_n))
    ax2.set_yticklabels(top_categories)
    ax2.set_xlabel('확률 (%)', fontsize=12)
    ax2.set_title(f'Top-{show_top_n} 예측 결과', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    
    # 막대에 수치 표시
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        ax2.text(prob + 1, i, f'{prob:.1f}%', 
                va='center', fontsize=10, fontweight='bold')
    
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return categories[top_indices[0]], predictions[top_indices[0]]


def predict_batch(image_folder, show_images=True):
    """
    폴더 내 여러 이미지 일괄 예측
    
    Args:
        image_folder: 이미지가 있는 폴더 경로
        show_images: 이미지 표시 여부
    """
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"폴더에 이미지가 없습니다: {image_folder}")
        return
    
    print(f"\n총 {len(image_files)}개 이미지 예측 중...\n")
    
    results = []
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        
        # 이미지 전처리
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 예측
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            predictions = probabilities.cpu().numpy()
        
        top_idx = np.argmax(predictions)
        
        results.append({
            'filename': img_file,
            'predicted_category': categories[top_idx],
            'confidence': predictions[top_idx] * 100
        })
        
        print(f"{img_file:30s} → {categories[top_idx]:20s} ({predictions[top_idx]*100:5.1f}%)")
    
    # 시각화
    if show_images and len(image_files) <= 12:
        n_images = len(image_files)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        if n_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_images > 1 else [axes]
        
        for idx, (img_file, result) in enumerate(zip(image_files, results)):
            img_path = os.path.join(image_folder, img_file)
            img = Image.open(img_path)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"{result['predicted_category']}\n({result['confidence']:.1f}%)", 
                               fontsize=11, fontweight='bold')
            axes[idx].axis('off')
        
        # 빈 subplot 숨기기
        for idx in range(len(image_files), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return results


# 사용 예시
if __name__ == "__main__":
    print("\n사용 방법:")
    print("1. 단일 이미지 예측:")
    print("   predict_image('이미지경로.jpg')")
    print("\n2. 폴더 내 이미지 일괄 예측:")
    print("   predict_batch('폴더경로')")
    print("\n" + "=" * 60)
    
    # 테스트 이미지 예측 (test_images 폴더가 있다면)
    test_folder = r'D:\test_images'
    if os.path.exists(test_folder):
        print(f"\n테스트 폴더 발견: {test_folder}")
        response = input("테스트 이미지 예측을 실행하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            predict_batch(test_folder)
    else:
        print(f"\n예시: Validation 폴더에서 샘플 이미지 예측")
        print("실제 사용시에는 아래 코드의 주석을 해제하세요:")
        print("# predict_image(r'D:\\K-Fashion_images\\Validation\\스트리트\\이미지파일.jpg')")

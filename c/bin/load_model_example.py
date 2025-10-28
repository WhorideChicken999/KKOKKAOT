"""
저장된 모델을 불러오는 예제
"""

import torch
from k_fashion_model import KFashionModel

# 저장 경로
SAVE_DIR = r'D:\kkokkaot\API\pre_trained_weights'

# ============================================
# 방법 1: 전체 모델 불러오기 (가장 간단)
# ============================================
print("방법 1: 전체 모델 불러오기")
model = torch.load(f'{SAVE_DIR}/k_fashion_best_model_full.pth')
model.eval()
print("✓ 모델 로드 완료\n")


# ============================================
# 방법 2: State dict로 불러오기 (권장)
# ============================================
print("방법 2: State dict로 불러오기")

# 1. 체크포인트 로드
checkpoint = torch.load(f'{SAVE_DIR}/k_fashion_best_model.pth')

# 2. class_to_idx에서 클래스 수 확인
num_classes = len(checkpoint['class_to_idx'])

# 3. 모델 생성
model = KFashionModel(num_classes)

# 4. 가중치 로드
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ 모델 로드 완료")
print(f"  - Epoch: {checkpoint['epoch']}")
print(f"  - Validation Accuracy: {checkpoint['val_acc']:.2f}%")
print(f"  - 클래스 매핑: {checkpoint['class_to_idx']}\n")


# ============================================
# 방법 3: GPU/CPU 선택하여 불러오기
# ============================================
print("방법 3: GPU/CPU 자동 선택")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")

# 전체 모델 로드
model = torch.load(
    f'{SAVE_DIR}/k_fashion_best_model_full.pth',
    map_location=device
)
model.eval()
print("✓ 모델 로드 완료\n")


# ============================================
# 사용 예제
# ============================================
print("=" * 60)
print("추론 예제")
print("=" * 60)

from torchvision import transforms
from PIL import Image

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# 예측 함수
def predict(image_path, model, device, class_to_idx):
    """이미지 예측"""
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # 추론
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        top5_prob, top5_idx = torch.topk(probabilities, 5)
    
    # 클래스 이름 매핑
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    # 결과 출력
    print(f"\n이미지: {image_path}")
    print("-" * 60)
    for i, (prob, idx) in enumerate(zip(top5_prob[0], top5_idx[0])):
        class_name = idx_to_class[idx.item()]
        print(f"{i+1}. {class_name}: {prob.item()*100:.2f}%")

# 사용 예시 (실제 이미지 경로로 변경 필요)
# predict('test_image.jpg', model, device, checkpoint['class_to_idx'])

print("\n✓ 모델 로드 및 사용 방법 완료!")


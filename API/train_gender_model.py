"""
성별 예측 모델 학습
- K-Fashion 데이터셋 사용
- Male/Female 이진 분류
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 파이프라인 모델 import
from pipeline.models import GenderClassifier, GENDER_CLASSES


class GenderDataset(Dataset):
    """성별 예측 데이터셋"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 이미지 로드
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 실패: {img_path} - {e}")
            # 빈 이미지 반환
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


def load_kfashion_data(data_dir: str, csv_path: str = None):
    """
    K-Fashion 데이터셋 로드
    
    Args:
        data_dir: 이미지 디렉토리 경로
        csv_path: 라벨 CSV 파일 경로 (있으면)
    
    Returns:
        image_paths, labels (male=0, female=1)
    """
    print("\n📂 데이터셋 로딩 중...")
    
    data_dir = Path(data_dir)
    
    if csv_path and Path(csv_path).exists():
        # CSV 파일이 있는 경우
        print(f"  📄 CSV 파일 사용: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 필요한 컬럼: image_path, gender
        image_paths = []
        labels = []
        
        for idx, row in df.iterrows():
            img_path = data_dir / row['image_name']  # 또는 적절한 컬럼명
            if img_path.exists():
                image_paths.append(str(img_path))
                # gender 컬럼 값: 'male', 'female', '남성', '여성' 등
                gender = row['gender'].lower()
                if 'male' in gender or '남' in gender:
                    labels.append(0)  # male
                elif 'female' in gender or '여' in gender:
                    labels.append(1)  # female
        
    else:
        # 폴더 구조로 되어 있는 경우
        # data_dir/male/*.jpg
        # data_dir/female/*.jpg
        print(f"  📁 폴더 구조 사용: {data_dir}")
        
        image_paths = []
        labels = []
        
        # Male 이미지
        male_dir = data_dir / 'male'
        if male_dir.exists():
            for img_path in male_dir.glob('*.jpg'):
                image_paths.append(str(img_path))
                labels.append(0)  # male
            for img_path in male_dir.glob('*.png'):
                image_paths.append(str(img_path))
                labels.append(0)
        
        # Female 이미지
        female_dir = data_dir / 'female'
        if female_dir.exists():
            for img_path in female_dir.glob('*.jpg'):
                image_paths.append(str(img_path))
                labels.append(1)  # female
            for img_path in female_dir.glob('*.png'):
                image_paths.append(str(img_path))
                labels.append(1)
    
    print(f"  ✅ 총 {len(image_paths)}개 이미지 로드")
    print(f"  📊 Male: {labels.count(0)}개, Female: {labels.count(1)}개")
    
    return image_paths, labels


def train_gender_model(
    data_dir: str,
    csv_path: str = None,
    batch_size: int = 32,
    epochs: int = 20,
    learning_rate: float = 0.001,
    save_path: str = "D:/kkokkaot/models/gender/best_model.pth"
):
    """
    성별 예측 모델 학습
    
    Args:
        data_dir: 데이터셋 디렉토리
        csv_path: 라벨 CSV 파일 (선택)
        batch_size: 배치 크기
        epochs: 에포크 수
        learning_rate: 학습률
        save_path: 모델 저장 경로
    """
    print("\n" + "="*60)
    print("🎓 Gender 예측 모델 학습 시작")
    print("="*60)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 디바이스: {device}")
    
    # 데이터 로드
    image_paths, labels = load_kfashion_data(data_dir, csv_path)
    
    if len(image_paths) == 0:
        print("❌ 데이터셋이 비어있습니다!")
        print("\n💡 데이터 준비 방법:")
        print("  방법 1: 폴더 구조")
        print("    data_dir/")
        print("      ├── male/")
        print("      │   ├── image1.jpg")
        print("      │   └── image2.jpg")
        print("      └── female/")
        print("          ├── image3.jpg")
        print("          └── image4.jpg")
        print("\n  방법 2: CSV 파일")
        print("    CSV 컬럼: image_name, gender")
        return
    
    # Train/Val 분할
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\n📊 데이터 분할")
    print(f"  Train: {len(train_paths)}개")
    print(f"  Val: {len(val_paths)}개")
    
    # 데이터 증강
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 생성
    train_dataset = GenderDataset(train_paths, train_labels, train_transform)
    val_dataset = GenderDataset(val_paths, val_labels, val_transform)
    
    # 데이터로더
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 모델 생성
    print("\n🤖 모델 생성 중...")
    model = GenderClassifier().to(device)
    
    # 손실 함수 & 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # 학습 기록
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    # 학습
    print(f"\n🚀 학습 시작 (Epochs: {epochs})")
    print("="*60)
    
    for epoch in range(epochs):
        # === Train ===
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels_batch in pbar:
            images = images.to(device)
            labels_batch = labels_batch.to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # 통계
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels_batch.size(0)
            train_correct += predicted.eq(labels_batch).sum().item()
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # === Validation ===
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels_batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = images.to(device)
                labels_batch = labels_batch.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels_batch.size(0)
                val_correct += predicted.eq(labels_batch).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # 기록
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 결과 출력
        print(f"\n📊 Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Best 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 저장 경로 폴더 생성
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': {'male': 0, 'female': 1}
            }, save_path)
            print(f"  ✅ Best 모델 저장! (Val Acc: {val_acc:.2f}%)")
        
        # Learning rate 조정
        scheduler.step(val_loss)
        print("="*60)
    
    print(f"\n🎉 학습 완료!")
    print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"  모델 저장 경로: {save_path}")
    
    # 학습 곡선 시각화
    plot_training_history(history, save_path.replace('.pth', '_history.png'))
    
    return model, history


def plot_training_history(history, save_path):
    """학습 곡선 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  📊 학습 곡선 저장: {save_path}")


if __name__ == "__main__":
    # K-Fashion 데이터셋으로 학습
    train_gender_model(
        data_dir="D:/kkokkaot/API/man_woman",
        batch_size=32,
        epochs=30,
        learning_rate=0.001,
        save_path="D:/kkokkaot/models/gender/best_model.pth"
    )


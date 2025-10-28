import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from k_fashion_model import KFashionModel

# 한글 폰트 설정 (matplotlib)
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 설정
IMG_SIZE = 192  # 속도 향상을 위해 축소 (224 -> 192)
BATCH_SIZE = 256  # 배치 크기 증가로 속도 향상
EPOCHS = 50
LEARNING_RATE = 0.001
USE_AMP = True  # Mixed Precision Training 사용

# 데이터 경로
TRAIN_DIR = r'D:\K-Fashion_images\Training'
VAL_DIR = r'D:\K-Fashion_images\Validation'

# 모델 저장 경로
SAVE_DIR = r'D:\kkokkaot\API\pre_trained_weights'
os.makedirs(SAVE_DIR, exist_ok=True)


# Top-K Accuracy 계산 함수
def top_k_accuracy(output, target, k=3):
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()


# 학습 함수 (AMP 지원)
def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    top3_correct = 0
    
    pbar = tqdm(loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision Training
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        top3_correct += top_k_accuracy(outputs, labels, k=3) * labels.size(0) / 100
        
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    epoch_top3_acc = 100. * top3_correct / total
    
    return epoch_loss, epoch_acc, epoch_top3_acc


# 검증 함수 (AMP 지원)
def validate(model, loader, criterion, device, use_amp=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    top3_correct = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Mixed Precision Training
            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            top3_correct += top_k_accuracy(outputs, labels, k=3) * labels.size(0) / 100
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    epoch_top3_acc = 100. * top3_correct / total
    
    return epoch_loss, epoch_acc, epoch_top3_acc


if __name__ == '__main__':
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    
    print("=" * 60)
    print("K-Fashion 이미지 분류 모델 학습 (PyTorch)")
    print("=" * 60)
    print(f"\n⚡ 속도 최적화 설정:")
    print(f"  - 이미지 크기: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - 배치 크기: {BATCH_SIZE}")
    print(f"  - Mixed Precision (AMP): {'활성화' if USE_AMP else '비활성화'}")
    
    # 카테고리 확인
    categories = sorted(os.listdir(TRAIN_DIR))
    num_classes = len(categories)
    print(f"\n카테고리 수: {num_classes}")
    print(f"카테고리 목록: {categories}\n")
    
    # 데이터 변환 정의 (증강 없음)
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 로드
    print("데이터 로딩 중...")
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=8, pin_memory=True,
                             persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=8, pin_memory=True,
                           persistent_workers=True, prefetch_factor=2)
    
    print(f"\nTraining 이미지 수: {len(train_dataset)}")
    print(f"Validation 이미지 수: {len(val_dataset)}")
    print(f"클래스 인덱스: {train_dataset.class_to_idx}\n")
    
    # 모델 구축
    print("모델 구축 중...")
    model = KFashionModel(num_classes).to(device)
    
    # print("\n모델 구조:")
    # print(model)
    print(f"\n총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"학습 가능한 파라미터 수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
    )
    
    # Mixed Precision Training Scaler
    scaler = torch.amp.GradScaler('cuda') if (USE_AMP and torch.cuda.is_available()) else None
    if scaler:
        print("✓ Mixed Precision Training (AMP) 활성화\n")
    
    # 학습 기록
    history = {
        'train_loss': [], 'train_acc': [], 'train_top3_acc': [],
        'val_loss': [], 'val_acc': [], 'val_top3_acc': []
    }
    
    # 모델 학습
    print("\n" + "=" * 60)
    print("학습 시작!")
    print("=" * 60 + "\n")
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 60)
        
        # 학습
        train_loss, train_acc, train_top3_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # 검증
        val_loss, val_acc, val_top3_acc = validate(
            model, val_loader, criterion, device, USE_AMP and torch.cuda.is_available()
        )
        
        # 학습률 조정
        scheduler.step(val_loss)
        
        # 기록 저장
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_top3_acc'].append(train_top3_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top3_acc'].append(val_top3_acc)
        
        # 결과 출력
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train Top-3: {train_top3_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Top-3: {val_top3_acc:.2f}%")
        
        # 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # State dict 저장 (권장 방식)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_to_idx': train_dataset.class_to_idx
            }, os.path.join(SAVE_DIR, 'k_fashion_best_model.pth'))
            
            # 전체 모델 저장 (재사용 용이)
            torch.save(model, os.path.join(SAVE_DIR, 'k_fashion_best_model_full.pth'))
            
            print(f"✓ 최고 성능 모델 저장! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 조기 종료
        if patience_counter >= patience:
            print(f"\n조기 종료! (Patience: {patience})")
            break
    
    # 최종 모델 저장
    # State dict 저장 (권장 방식)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'class_to_idx': train_dataset.class_to_idx
    }, os.path.join(SAVE_DIR, 'k_fashion_final_model.pth'))
    
    # 전체 모델 저장 (재사용 용이)
    torch.save(model, os.path.join(SAVE_DIR, 'k_fashion_final_model_full.pth'))
    
    print(f"\n최종 모델 저장 완료:")
    print(f"  - State dict: {os.path.join(SAVE_DIR, 'k_fashion_final_model.pth')}")
    print(f"  - 전체 모델: {os.path.join(SAVE_DIR, 'k_fashion_final_model_full.pth')}")
    
    # 학습 결과 시각화
    print("\n학습 결과 시각화 중...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history['train_acc'], label='Train Accuracy')
    axes[0, 0].plot(history['val_acc'], label='Val Accuracy')
    axes[0, 0].set_title('모델 정확도 (Accuracy)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history['train_loss'], label='Train Loss')
    axes[0, 1].plot(history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('모델 손실 (Loss)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Top-3 Accuracy
    axes[1, 0].plot(history['train_top3_acc'], label='Train Top-3 Acc')
    axes[1, 0].plot(history['val_top3_acc'], label='Val Top-3 Acc')
    axes[1, 0].set_title('Top-3 정확도', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Top-3 Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 최종 성능 요약
    axes[1, 1].axis('off')
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    final_top3_acc = history['val_top3_acc'][-1]
    best_val_acc_final = max(history['val_acc'])
    
    summary_text = f"""
최종 학습 결과

▶ Training Accuracy: {final_train_acc:.2f}%
▶ Validation Accuracy: {final_val_acc:.2f}%
▶ Best Validation Accuracy: {best_val_acc_final:.2f}%
▶ Top-3 Accuracy: {final_top3_acc:.2f}%

총 Epoch: {len(history['train_acc'])}
총 카테고리 수: {num_classes}
"""
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                    verticalalignment='center', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('k_fashion_training_result.png', dpi=300, bbox_inches='tight')
    print("학습 결과 그래프 저장 완료: k_fashion_training_result.png")
    plt.show()
    
    # 최종 평가
    print("\n" + "=" * 60)
    print("최종 모델 평가")
    print("=" * 60)
    
    final_val_loss, final_val_acc, final_val_top3 = validate(
        model, val_loader, criterion, device, USE_AMP and torch.cuda.is_available()
    )
    print(f"\n✓ Validation Loss: {final_val_loss:.4f}")
    print(f"✓ Validation Accuracy: {final_val_acc:.2f}%")
    print(f"✓ Top-3 Accuracy: {final_val_top3:.2f}%")
    
    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from collections import defaultdict
import random
from tqdm import tqdm

# 재현성을 위한 시드 설정
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class FashionDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        
        # 이미지 로드
        image = Image.open(item['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'category': item['category_encoded'],
            'color': item['color_encoded'],
            'material': item['material_encoded'],
            'print': item['print_encoded'],
            'fit': item['fit_encoded'],
            'style': item['style_encoded'],
            'sleeve': item['sleeve_encoded']
        }

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

def load_schema(schema_path):
    """스키마 파일에서 속성 정보 로드"""
    df = pd.read_csv(schema_path)
    
    schema = {
        'category': [],
        'color': [],
        'material': [],
        'print': [],
        'fit': [],
        'style': [],
        'sleeve': []
    }
    
    # 카테고리 (상의만)
    schema['category'] = df[(df['category'] == 'Item') & (df['subcategory'] == 'Top')]['label_ko'].tolist()
    
    # 색상 (모든 카테고리)
    schema['color'] = df[df['category'] == 'Color']['label_ko'].tolist()
    
    # 소재 (모든 카테고리)
    schema['material'] = df[df['category'] == 'Material']['label_ko'].tolist()
    
    # 프린트 (모든 카테고리)
    schema['print'] = df[df['category'] == 'Print']['label_ko'].tolist()
    
    # 핏 (상의만)
    schema['fit'] = df[df['subcategory'] == 'Fit']['label_ko'].tolist()
    
    # 스타일 (모든 카테고리)
    schema['style'] = df[df['category'] == 'Style']['label_ko'].tolist()
    
    # 소매기장 (상의만)
    schema['sleeve'] = df[df['subcategory'] == 'Sleeve length']['label_ko'].tolist()
    
    return schema

def match_label_image_pairs(base_path, target_category):
    """라벨과 이미지 파일을 매칭"""
    cnn_labels_path = os.path.join(base_path, 'cnn_labels')
    cropped_images_path = os.path.join(base_path, 'cropped_images')
    
    print(f"\n=== 라벨-이미지 매칭 시작 ({target_category}) ===")
    
    matched_pairs = []
    unmatched_labels = []
    unmatched_images = []
    
    label_dir = os.path.join(cnn_labels_path, target_category)
    image_dir = os.path.join(cropped_images_path, target_category)
    
    if not os.path.exists(label_dir):
        print(f"경고: 라벨 폴더가 없습니다: {label_dir}")
        return []
    if not os.path.exists(image_dir):
        print(f"경고: 이미지 폴더가 없습니다: {image_dir}")
        return []
    
    # 라벨 파일 목록
    label_files = {f: os.path.join(label_dir, f) 
                  for f in os.listdir(label_dir) if f.endswith('.json')}
    
    # 이미지 파일 목록
    image_files = {f: os.path.join(image_dir, f) 
                  for f in os.listdir(image_dir) if f.endswith('.jpg')}
    
    print(f"\n[{target_category}]")
    print(f"  라벨 파일: {len(label_files)}개")
    print(f"  이미지 파일: {len(image_files)}개")
    
    # 매칭 진행
    for label_file, label_path in tqdm(label_files.items(), desc=f"  {target_category} 매칭 중"):
        # 대응하는 이미지 파일명 생성
        image_file = label_file.replace('.json', '.jpg')
        
        if image_file in image_files:
            # 매칭 성공
            matched_pairs.append({
                'category': target_category,
                'label_path': label_path,
                'image_path': image_files[image_file],
                'file_name': label_file.replace('.json', '')
            })
        else:
            # 매칭 실패 - 이미지 없음
            unmatched_labels.append({
                'category': target_category,
                'file_name': label_file
            })
    
    # 매칭되지 않은 이미지 찾기
    matched_image_files = {label_file.replace('.json', '.jpg') for label_file in label_files.keys()}
    for image_file in image_files.keys():
        if image_file not in matched_image_files:
            unmatched_images.append({
                'category': target_category,
                'file_name': image_file
            })
    
    print(f"  매칭 완료: {len(matched_pairs)}개")
    
    # 매칭 통계
    print(f"\n=== 매칭 결과 ===")
    print(f"✓ 매칭 성공: {len(matched_pairs)}개")
    print(f"✗ 이미지 없는 라벨: {len(unmatched_labels)}개")
    print(f"✗ 라벨 없는 이미지: {len(unmatched_images)}개")
    
    if unmatched_labels:
        print(f"\n처음 5개 매칭 실패 라벨 예시:")
        for item in unmatched_labels[:5]:
            print(f"  - [{item['category']}] {item['file_name']}")
    
    if unmatched_images:
        print(f"\n처음 5개 매칭 실패 이미지 예시:")
        for item in unmatched_images[:5]:
            print(f"  - [{item['category']}] {item['file_name']}")
    
    return matched_pairs

def prepare_data(base_path, schema, target_category, sample_size=None):
    """데이터 준비 및 인코딩"""
    # 라벨-이미지 매칭
    matched_pairs = match_label_image_pairs(base_path, target_category)
    
    print(f"\n=== 데이터 준비 시작 ===")
    
    data_list = []
    invalid_data_count = 0
    
    # 매칭된 데이터 처리
    for pair in tqdm(matched_pairs, desc="라벨 데이터 로드 중"):
        try:
            # 라벨 로드
            with open(pair['label_path'], 'r', encoding='utf-8') as f:
                label_data = json.load(f)
            
            item_data = label_data.get('item_data', {})
            style_data = label_data.get('style', {})
            
            # 필요한 속성 추출
            item_category = item_data.get('카테고리', '')
            color = item_data.get('색상', '')
            materials = item_data.get('소재', [])
            prints = item_data.get('프린트', [])
            fit = item_data.get('핏', '')
            sleeve = item_data.get('소매기장', '')
            style = style_data.get('스타일', '')
            
            # 빈 값 체크
            if not item_category or not color:
                invalid_data_count += 1
                continue
            
            if not materials:
                materials = []
            if not prints:
                prints = []
            if not sleeve:
                sleeve = '긴팔'  # 기본값
            if not style:
                style = '모던'  # 기본값
            
            data_list.append({
                'image_path': pair['image_path'],
                'category': item_category,
                'color': color,
                'material': materials[0] if materials else '저지',
                'print': prints[0] if prints else '무지',
                'fit': fit if fit else '노멀',
                'sleeve': sleeve,
                'style': style
            })
        except Exception as e:
            invalid_data_count += 1
            continue
    
    print(f"✓ 유효한 데이터: {len(data_list)}개")
    print(f"✗ 유효하지 않은 데이터: {invalid_data_count}개 (필수 속성 누락 또는 오류)")
    
    # 샘플링
    if sample_size and sample_size < len(data_list):
        print(f"\n샘플링: {len(data_list)}개 → {sample_size}개")
        data_list = random.sample(data_list, sample_size)
    
    print(f"\n=== 라벨 인코딩 시작 ===")
    
    # 라벨 인코딩 준비 (모두 단일 레이블)
    encoders = {}
    for attr in ['category', 'color', 'material', 'print', 'fit', 'style', 'sleeve']:
        encoders[attr] = LabelEncoder()
        try:
            encoders[attr].fit(schema[attr])
            print(f"✓ {attr}: {len(schema[attr])}개 클래스")
        except Exception as e:
            print(f"✗ {attr} 인코더 생성 실패: {e}")
            return None, None
    
    # 데이터 인코딩
    encoded_data_list = []
    skipped_count = 0
    
    for item in tqdm(data_list, desc="데이터 인코딩 중"):
        try:
            encoded_item = {'image_path': item['image_path']}
            
            for attr in ['category', 'color', 'material', 'print', 'fit', 'style', 'sleeve']:
                encoded_item[f'{attr}_encoded'] = encoders[attr].transform([item[attr]])[0]
            
            encoded_data_list.append(encoded_item)
        except ValueError as e:
            # 스키마에 없는 값은 건너뛰기
            skipped_count += 1
            continue
    
    print(f"✓ 인코딩 완료: {len(encoded_data_list)}개")
    if skipped_count > 0:
        print(f"✗ 스키마 미포함으로 스킵: {skipped_count}개")
    
    return encoded_data_list, encoders

def train_model(model, train_loader, val_loader, device, encoders, schema, output_dir, num_epochs=1):
    """모델 학습"""
    # 손실 함수 (모두 단일 레이블)
    criterion = nn.CrossEntropyLoss()
    
    # 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 최고 성능 추적
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(num_epochs):
        print(f"\n에포크 {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # 학습 단계
        model.train()
        train_losses = defaultdict(float)
        train_correct = defaultdict(int)
        train_total = 0
        
        train_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="학습")
        for batch_idx, batch in train_bar:
            images = batch['image'].to(device)
            
            # 라벨
            labels = {attr: batch[attr].to(device) for attr in ['category', 'color', 'material', 'print', 'fit', 'style', 'sleeve']}
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # 손실 계산
            losses = {}
            total_loss = 0
            for attr in ['category', 'color', 'material', 'print', 'fit', 'style', 'sleeve']:
                losses[attr] = criterion(outputs[attr], labels[attr])
                total_loss += losses[attr]
                train_losses[attr] += losses[attr].item()
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # 통계
            train_losses['total'] += total_loss.item()
            
            # 정확도 계산
            for attr in ['category', 'color', 'material', 'print', 'fit', 'style', 'sleeve']:
                _, pred = outputs[attr].max(1)
                train_correct[attr] += (pred == labels[attr]).sum().item()
            train_total += images.size(0)
            
            # tqdm 상태바 업데이트
            train_bar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'cat_acc': f'{100*train_correct["category"]/train_total:.1f}%',
                'color_acc': f'{100*train_correct["color"]/train_total:.1f}%'
            })
        
        # 평균 손실 및 정확도
        num_batches = len(train_loader)
        print(f"\n[학습] 평균 손실: {train_losses['total']/num_batches:.4f}")
        for attr in ['category', 'color', 'material', 'print', 'fit', 'style', 'sleeve']:
            print(f"  - {attr} 손실: {train_losses[attr]/num_batches:.4f}, "
                  f"정확도: {100*train_correct[attr]/train_total:.2f}%")
        
        # 검증 단계
        model.eval()
        val_losses = defaultdict(float)
        val_correct = defaultdict(int)
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="검증")
            for batch in val_bar:
                images = batch['image'].to(device)
                
                labels = {attr: batch[attr].to(device) for attr in ['category', 'color', 'material', 'print', 'fit', 'style', 'sleeve']}
                
                outputs = model(images)
                
                losses = {}
                total_loss = 0
                for attr in ['category', 'color', 'material', 'print', 'fit', 'style', 'sleeve']:
                    losses[attr] = criterion(outputs[attr], labels[attr])
                    total_loss += losses[attr]
                    val_losses[attr] += losses[attr].item()
                
                val_losses['total'] += total_loss.item()
                
                # 정확도 계산
                for attr in ['category', 'color', 'material', 'print', 'fit', 'style', 'sleeve']:
                    _, pred = outputs[attr].max(1)
                    val_correct[attr] += (pred == labels[attr]).sum().item()
                val_total += images.size(0)
                
                # tqdm 상태바 업데이트
                val_bar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'cat_acc': f'{100*val_correct["category"]/val_total:.1f}%',
                    'color_acc': f'{100*val_correct["color"]/val_total:.1f}%'
                })
        
        num_val_batches = len(val_loader)
        avg_val_loss = val_losses['total']/num_val_batches
        
        print(f"\n[검증] 평균 손실: {avg_val_loss:.4f}")
        for attr in ['category', 'color', 'material', 'print', 'fit', 'style', 'sleeve']:
            print(f"  - {attr} 손실: {val_losses[attr]/num_val_batches:.4f}, "
                  f"정확도: {100*val_correct[attr]/val_total:.2f}%")
        
        scheduler.step(avg_val_loss)
        
        # 체크포인트 저장
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_losses['total']/len(train_loader),
            'val_loss': avg_val_loss,
            'encoders': encoders,
            'schema': schema
        }
        
        # 마지막 체크포인트 저장
        checkpoint_path = os.path.join(output_dir, 'last_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"\n체크포인트 저장: {checkpoint_path}")
        
        # 최고 성능 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            print(f"★ 최고 성능 모델 저장! (에포크 {best_epoch}, 검증 손실: {best_val_loss:.4f})")
    
    print(f"\n학습 완료! 최고 성능: 에포크 {best_epoch}, 검증 손실: {best_val_loss:.4f}")
    return best_val_loss

def main():
    # 설정
    base_path = r'D:\prepared_data'
    schema_path = r'D:\kkokkaot\API\kfashion_attributes_schema.csv'
    output_dir = r'D:\kkokkaot\models\top'
    target_category = '상의'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")
    print(f"대상 카테고리: {target_category}")
    
    # 스키마 로드
    print("\n스키마 로드 중...")
    schema = load_schema(schema_path)
    for attr, classes in schema.items():
        print(f"{attr}: {len(classes)}개 클래스")
    
    # 데이터 준비 (샘플링)
    print("\n데이터 준비 중...")
    sample_size = 66686 #66???  # 1 epoch 테스트를 위해 1000개 샘플링
    data_list, encoders = prepare_data(base_path, schema, target_category, sample_size=sample_size)
    
    if data_list is None:
        print("데이터 준비 실패!")
        return
    
    # Train/Val 분할
    train_data, val_data = train_test_split(data_list, test_size=0.2, random_state=42)
    print(f"학습 데이터: {len(train_data)}개, 검증 데이터: {len(val_data)}개")
    
    # 데이터 변환
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = FashionDataset(train_data, transform=transform)
    val_dataset = FashionDataset(val_data, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # 모델 생성
    print("\n모델 생성 중...")
    model = TopFashionModel(
        num_category=len(schema['category']),
        num_color=len(schema['color']),
        num_material=len(schema['material']),
        num_print=len(schema['print']),
        num_fit=len(schema['fit']),
        num_style=len(schema['style']),
        num_sleeve=len(schema['sleeve'])
    )
    model = model.to(device)
    
    # 모델 학습
    print("\n학습 시작...")
    train_model(model, train_loader, val_loader, device, encoders, schema, output_dir, num_epochs=50)
    
    print(f"\n✓ 모든 가중치가 {output_dir} 폴더에 저장되었습니다.")
    print(f"  - best_model.pth: 최고 성능 모델")
    print(f"  - last_checkpoint.pth: 마지막 체크포인트")

if __name__ == '__main__':
    main()

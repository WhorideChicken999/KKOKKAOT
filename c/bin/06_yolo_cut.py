import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from tqdm import tqdm

# GPU 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {DEVICE}")

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class FashionDataset(Dataset):
    """패션 속성 예측 데이터셋"""
    
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 이미지 로드
        try:
            image = Image.open(row['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except:
            # 에러 시 빈 이미지
            image = torch.zeros(3, 224, 224)
        
        # 라벨 추출
        labels = {}
        
        for cat_type in ['상의', '원피스', '하의', '아우터']:
            cat = row.get(f'{cat_type}_카테고리', '')
            color = row.get(f'{cat_type}_색상', '')
            fit = row.get(f'{cat_type}_핏', '')
            mat = row.get(f'{cat_type}_소재', '[]')
            
            if pd.notna(cat) and cat:
                labels['category'] = str(cat)
                labels['color'] = str(color) if pd.notna(color) and color else 'none'
                labels['fit'] = str(fit) if pd.notna(fit) and fit else 'none'
                
                # 소재 파싱
                if isinstance(mat, str):
                    try:
                        mat_list = eval(mat) if mat.startswith('[') else [mat]
                    except:
                        mat_list = []
                else:
                    mat_list = mat if isinstance(mat, list) else []
                
                labels['materials'] = mat_list if mat_list else []
                break
        
        if 'category' not in labels:
            labels = {'category': 'none', 'color': 'none', 'fit': 'none', 'materials': []}
        
        return image, labels


class MultiTaskFashionModel(nn.Module):
    """Multi-task 패션 속성 예측 모델 (소재 포함)"""
    
    def __init__(self, num_categories, num_colors, num_fits, num_materials):
        super().__init__()
        
        # EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Task-specific heads
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
        
        # 소재 head (multi-label)
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
        material_out = self.material_head(features)  # sigmoid는 loss에서 처리
        
        return category_out, color_out, fit_out, material_out


def prepare_data(csv_path='dataset_with_gender.csv'):
    """데이터 준비 및 인코딩"""
    
    print("데이터 로드 중...")
    df = pd.read_csv(csv_path)
    
    # 카테고리, 색상, 핏, 소재 정보 추출
    categories = []
    colors = []
    fits = []
    materials = []
    
    for idx, row in df.iterrows():
        # 우선순위: 상의 > 원피스 > 하의 > 아우터
        cat_found = False
        for cat_type in ['상의', '원피스', '하의', '아우터']:
            cat = row.get(f'{cat_type}_카테고리', '')
            if pd.notna(cat) and cat:
                categories.append(cat)
                colors.append(row.get(f'{cat_type}_색상', ''))
                fits.append(row.get(f'{cat_type}_핏', ''))
                
                # 소재 (리스트 형태)
                mat = row.get(f'{cat_type}_소재', '[]')
                if isinstance(mat, str):
                    try:
                        mat_list = eval(mat) if mat.startswith('[') else [mat]
                    except:
                        mat_list = []
                else:
                    mat_list = mat if isinstance(mat, list) else []
                materials.append(mat_list)
                
                cat_found = True
                break
        
        if not cat_found:
            categories.append('')
            colors.append('')
            fits.append('')
            materials.append([])
    
    df['target_category'] = categories
    df['target_color'] = colors
    df['target_fit'] = fits
    df['target_materials'] = materials
    
    # 빈 값 필터링 - unknown 제외
    df = df[
        (df['target_category'] != '') & 
        (df['target_category'] != 'unknown')
    ].reset_index(drop=True)
    
    print(f"유효 데이터: {len(df)}개")
    
    # 라벨 인코딩
    category_encoder = LabelEncoder()
    color_encoder = LabelEncoder()
    fit_encoder = LabelEncoder()
    
    df['category_encoded'] = category_encoder.fit_transform(df['target_category'])
    
    # 색상과 핏은 빈 값을 'none'으로 대체
    df['target_color'] = df['target_color'].fillna('none').replace('', 'none')
    df['target_fit'] = df['target_fit'].fillna('none').replace('', 'none')
    
    df['color_encoded'] = color_encoder.fit_transform(df['target_color'])
    df['fit_encoded'] = fit_encoder.fit_transform(df['target_fit'])
    
    # 소재: 모든 unique 소재 찾기
    all_materials = set()
    for mat_list in df['target_materials']:
        if isinstance(mat_list, list):
            all_materials.update(mat_list)
    all_materials = sorted(list(all_materials))
    
    # 소재 인덱스 매핑
    material_to_idx = {mat: idx for idx, mat in enumerate(all_materials)}
    
    print(f"\n클래스 수:")
    print(f"  카테고리: {len(category_encoder.classes_)}")
    print(f"  색상: {len(color_encoder.classes_)}")
    print(f"  핏: {len(fit_encoder.classes_)}")
    print(f"  소재: {len(all_materials)}")
    
    # Train/Test 분할
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category_encoded'])
    
    print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")
    
    # 인코더 저장
    encoders = {
        'category': category_encoder,
        'color': color_encoder,
        'fit': fit_encoder,
        'material_to_idx': material_to_idx,
        'material_classes': all_materials
    }
    
    return train_df, test_df, encoders


def collate_fn(batch):
    """커스텀 collate function"""
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    return images, labels

def train_model(train_df, test_df, encoders, epochs=10, batch_size=32):
    """모델 학습"""
    
    # 데이터셋 생성
    train_dataset = FashionDataset(train_df, transform=transform)
    test_dataset = FashionDataset(test_df, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, collate_fn=collate_fn)
    
    # 모델 생성
    num_categories = len(encoders['category'].classes_)
    num_colors = len(encoders['color'].classes_)
    num_fits = len(encoders['fit'].classes_)
    num_materials = len(encoders['material_classes'])
    
    model = MultiTaskFashionModel(num_categories, num_colors, num_fits, num_materials).to(DEVICE)
    
    # 손실 함수 & 옵티마이저
    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    material_to_idx = encoders['material_to_idx']
    
    # 학습
    print("\n모델 학습 시작...")
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(DEVICE)
            
            # 라벨 인코딩
            cat_labels = torch.tensor([encoders['category'].transform([l['category']])[0] 
                                      for l in labels]).to(DEVICE)
            color_labels = torch.tensor([encoders['color'].transform([l['color']])[0] 
                                        for l in labels]).to(DEVICE)
            fit_labels = torch.tensor([encoders['fit'].transform([l['fit']])[0] 
                                      for l in labels]).to(DEVICE)
            
            # 소재 multi-hot 인코딩
            mat_labels = torch.zeros(len(labels), num_materials).to(DEVICE)
            for i, label in enumerate(labels):
                for mat in label['materials']:
                    if mat in material_to_idx:
                        mat_labels[i, material_to_idx[mat]] = 1
            
            optimizer.zero_grad()
            
            cat_out, color_out, fit_out, mat_out = model(images)
            
            loss = (criterion_ce(cat_out, cat_labels) + 
                   criterion_ce(color_out, color_labels) + 
                   criterion_ce(fit_out, fit_labels) +
                   criterion_bce(mat_out, mat_labels)) / 4
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        scheduler.step()
        
        # 검증
        model.eval()
        correct_cat = 0
        correct_color = 0
        correct_fit = 0
        mat_precision = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(DEVICE)
                
                cat_labels = torch.tensor([encoders['category'].transform([l['category']])[0] 
                                          for l in labels]).to(DEVICE)
                color_labels = torch.tensor([encoders['color'].transform([l['color']])[0] 
                                            for l in labels]).to(DEVICE)
                fit_labels = torch.tensor([encoders['fit'].transform([l['fit']])[0] 
                                          for l in labels]).to(DEVICE)
                
                mat_labels = torch.zeros(len(labels), num_materials).to(DEVICE)
                for i, label in enumerate(labels):
                    for mat in label['materials']:
                        if mat in material_to_idx:
                            mat_labels[i, material_to_idx[mat]] = 1
                
                cat_out, color_out, fit_out, mat_out = model(images)
                
                _, cat_pred = torch.max(cat_out, 1)
                _, color_pred = torch.max(color_out, 1)
                _, fit_pred = torch.max(fit_out, 1)
                mat_pred = (torch.sigmoid(mat_out) > 0.5).float()
                
                total += cat_labels.size(0)
                correct_cat += (cat_pred == cat_labels).sum().item()
                correct_color += (color_pred == color_labels).sum().item()
                correct_fit += (fit_pred == fit_labels).sum().item()
                
                # 소재 정확도
                mat_correct = (mat_pred * mat_labels).sum()
                mat_total = mat_pred.sum()
                if mat_total > 0:
                    mat_precision += (mat_correct / mat_total).item()
        
        cat_acc = 100 * correct_cat / total
        color_acc = 100 * correct_color / total
        fit_acc = 100 * correct_fit / total
        mat_prec = 100 * mat_precision / len(test_loader) if len(test_loader) > 0 else 0
        avg_acc = (cat_acc + color_acc + fit_acc + mat_prec) / 4
        
        print(f"\nEpoch {epoch+1}")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Test - 카테고리: {cat_acc:.2f}%, 색상: {color_acc:.2f}%, 핏: {fit_acc:.2f}%, 소재: {mat_prec:.2f}%")
        
        # 베스트 모델 저장
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'encoders': encoders
            }, 'fashion_attribute_model.pth')
            print(f"  ★ 베스트 모델 저장 (평균: {avg_acc:.2f}%)")
    
    print("\n학습 완료!")
    return model, encoders


if __name__ == '__main__':
    # 데이터 준비
    train_df, test_df, encoders = prepare_data('dataset_with_gender.csv')
    
    # 모델 학습
    model, encoders = train_model(train_df, test_df, encoders, epochs=20, batch_size=128)
    
    print("\n모델 저장 완료: fashion_attribute_model.pth")
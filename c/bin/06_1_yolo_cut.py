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
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 디바이스: {DEVICE}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class FashionDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        try:
            image = Image.open(row['image_path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except:
            image = torch.zeros(3, 224, 224)
        
        labels = {
            'category': row['target_category'],
            'color': row['target_color'],
            'fit': row['target_fit'],
            'materials': row['target_materials']
        }
        
        return image, labels


class FashionAttributeModel(nn.Module):
    """범용 패션 속성 예측 모델"""
    
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


def prepare_data_by_type(csv_path, item_type='top'):
    """상의 또는 하의 데이터만 추출하여 준비"""
    
    print(f"\n{'='*60}")
    print(f"{item_type.upper()} 데이터 준비 중...")
    print(f"{'='*60}")
    
    df = pd.read_csv(csv_path)
    
    # 카테고리 정의
    top_categories = [
        '티셔츠', '블라우스', '니트', '가디건', '재킷', '셔츠', '후드티',
        '맨투맨', '조끼', '베스트', '코트', '점퍼', '패딩'
    ]
    
    bottom_categories = [
        '팬츠', '스커트', '청바지', '레깅스', '쇼츠', '치마', 
        '슬랙스', '조거팬츠', '와이드팬츠'
    ]
    
    if item_type == 'top':
        prefix = '상의'
        valid_categories = top_categories
    else:
        prefix = '하의'
        valid_categories = bottom_categories
    
    # 데이터 추출
    categories = []
    colors = []
    fits = []
    materials = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        cat = row.get(f'{prefix}_카테고리', '')
        
        if pd.notna(cat) and cat and cat in valid_categories:
            categories.append(cat)
            colors.append(row.get(f'{prefix}_색상', 'none'))
            fits.append(row.get(f'{prefix}_핏', 'none'))
            
            # 소재 처리
            mat = row.get(f'{prefix}_소재', '[]')
            if isinstance(mat, str):
                try:
                    mat_list = eval(mat) if mat.startswith('[') else [mat]
                except:
                    mat_list = []
            else:
                mat_list = mat if isinstance(mat, list) else []
            materials.append(mat_list if mat_list else [])
            
            valid_indices.append(idx)
    
    # 필터링된 데이터프레임 생성
    filtered_df = df.iloc[valid_indices].copy()
    filtered_df['target_category'] = categories
    filtered_df['target_color'] = colors
    filtered_df['target_fit'] = fits
    filtered_df['target_materials'] = materials
    
    # none 처리
    filtered_df['target_color'] = filtered_df['target_color'].fillna('none').replace('', 'none')
    filtered_df['target_fit'] = filtered_df['target_fit'].fillna('none').replace('', 'none')
    
    print(f"유효 데이터: {len(filtered_df)}개")
    print(f"카테고리 분포:")
    print(filtered_df['target_category'].value_counts().head(10))
    
    # 레이블 인코딩
    category_encoder = LabelEncoder()
    color_encoder = LabelEncoder()
    fit_encoder = LabelEncoder()
    
    filtered_df['category_encoded'] = category_encoder.fit_transform(filtered_df['target_category'])
    filtered_df['color_encoded'] = color_encoder.fit_transform(filtered_df['target_color'])
    filtered_df['fit_encoded'] = fit_encoder.fit_transform(filtered_df['target_fit'])
    
    # 소재 인덱스
    all_materials = set()
    for mat_list in filtered_df['target_materials']:
        if isinstance(mat_list, list):
            all_materials.update(mat_list)
    all_materials = sorted(list(all_materials))
    material_to_idx = {mat: idx for idx, mat in enumerate(all_materials)}
    
    print(f"\n클래스 수:")
    print(f"  카테고리: {len(category_encoder.classes_)}")
    print(f"  색상: {len(color_encoder.classes_)}")
    print(f"  핏: {len(fit_encoder.classes_)}")
    print(f"  소재: {len(all_materials)}")
    
    # Train/Test 분할
    train_df, test_df = train_test_split(
        filtered_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=filtered_df['category_encoded']
    )
    
    print(f"\nTrain: {len(train_df)}, Test: {len(test_df)}")
    
    encoders = {
        'category': category_encoder,
        'color': color_encoder,
        'fit': fit_encoder,
        'material_to_idx': material_to_idx,
        'material_classes': all_materials
    }
    
    return train_df, test_df, encoders


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    return images, labels


def train_model(train_df, test_df, encoders, item_type='top', epochs=20, batch_size=128):
    """모델 학습"""
    
    print(f"\n{'='*60}")
    print(f"{item_type.upper()} 모델 학습 시작")
    print(f"{'='*60}\n")
    
    train_dataset = FashionDataset(train_df, transform=transform)
    test_dataset = FashionDataset(test_df, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, collate_fn=collate_fn)
    
    num_categories = len(encoders['category'].classes_)
    num_colors = len(encoders['color'].classes_)
    num_fits = len(encoders['fit'].classes_)
    num_materials = len(encoders['material_classes'])
    
    model = FashionAttributeModel(num_categories, num_colors, num_fits, num_materials).to(DEVICE)
    
    criterion_ce = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    material_to_idx = encoders['material_to_idx']
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
        correct_cat = correct_color = correct_fit = mat_precision = total = 0
        
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
        
        if avg_acc > best_acc:
            best_acc = avg_acc
            model_path = f'fashion_{item_type}_model.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'encoders': encoders
            }, model_path)
            print(f"  ★ 베스트 모델 저장: {model_path} (평균: {avg_acc:.2f}%)")
    
    print(f"\n{item_type.upper()} 모델 학습 완료!")
    return model, encoders


if __name__ == '__main__':
    CSV_PATH = 'dataset_with_gender.csv'
    
    # 상의 모델 학습
    print("\n" + "="*60)
    print("1. 상의 모델 학습")
    print("="*60)
    
    top_train_df, top_test_df, top_encoders = prepare_data_by_type(CSV_PATH, item_type='top')
    top_model, top_encoders = train_model(top_train_df, top_test_df, top_encoders, 
                                          item_type='top', epochs=20, batch_size=128)
    
    # 하의 모델 학습
    print("\n" + "="*60)
    print("2. 하의 모델 학습")
    print("="*60)
    
    bottom_train_df, bottom_test_df, bottom_encoders = prepare_data_by_type(CSV_PATH, item_type='bottom')
    bottom_model, bottom_encoders = train_model(bottom_train_df, bottom_test_df, bottom_encoders,
                                                item_type='bottom', epochs=20, batch_size=128)
    
    print("\n" + "="*60)
    print("✓ 모든 모델 학습 완료")
    print("  - fashion_top_model.pth")
    print("  - fashion_bottom_model.pth")
    print("="*60)
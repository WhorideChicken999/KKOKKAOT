import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# GPU 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {DEVICE}")

# CLIP 모델 로드
print("\nCLIP 모델 로드 중...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 성별 분류 텍스트 프롬프트 (더 명확하게)
GENDER_PROMPTS = {
    'female': [
        "women's fashion clothing",
        "feminine dress or skirt",
        "ladies clothing with ruffles or frills",
        "female fashion style"
    ],
    'male': [
        "men's fashion clothing",
        "masculine casual wear",
        "male street style clothing"
    ],
    'unisex': [
        "unisex casual clothing",
        "gender neutral fashion"
    ]
}


def predict_gender_clip(image_path: str, threshold: float = 0.3) -> dict:
    """CLIP을 사용한 성별 예측 (개선 버전)"""
    try:
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        # 모든 프롬프트 준비
        all_prompts = []
        prompt_labels = []
        for gender, prompts in GENDER_PROMPTS.items():
            all_prompts.extend(prompts)
            prompt_labels.extend([gender] * len(prompts))
        
        # CLIP 인코딩
        inputs = processor(
            text=all_prompts,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)
        
        # 유사도 계산
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0].cpu().numpy()
        
        # 성별별 평균 확률 계산
        gender_scores = {}
        for gender in GENDER_PROMPTS.keys():
            indices = [i for i, label in enumerate(prompt_labels) if label == gender]
            gender_scores[gender] = probs[indices].mean()
        
        # 최종 예측 (female vs male만 비교, unisex 제외)
        female_score = gender_scores['female']
        male_score = gender_scores['male']
        
        # female과 male 중 더 높은 쪽 선택
        if female_score > male_score:
            predicted_gender = 'female'
            confidence = female_score
        else:
            predicted_gender = 'male'
            confidence = male_score
        
        # 둘 다 낮으면 unisex
        if max(female_score, male_score) < threshold:
            predicted_gender = 'unisex'
            confidence = gender_scores['unisex']
        
        return {
            'gender': predicted_gender,
            'confidence': float(confidence),
            'scores': {k: float(v) for k, v in gender_scores.items()}
        }
        
    except Exception as e:
        print(f"에러 ({image_path}): {e}")
        return {
            'gender': 'unknown',
            'confidence': 0.0,
            'scores': {}
        }


def predict_gender_metadata(row: pd.Series) -> tuple:
    """메타데이터 기반 성별 예측 (색상 위주)"""
    
    # 여성 색상
    female_colors = ['핑크', '라벤더', '퍼플', '로즈', '코랄']
    
    # 중립 색상
    neutral_colors = ['베이지', '화이트', '그레이', '블랙', '네이비', '브라운', '카키']
    
    # 밝은 색상 (여성 가능성)
    bright_colors = ['스카이블루', '민트', '레몬', '옐로우', '라임']
    
    female_score = 0
    unisex_score = 0
    
    # 1. 색상 체크 (모든 카테고리)
    all_colors = []
    for col in row.index:
        if '색상' in col and pd.notna(row[col]) and row[col]:
            all_colors.append(str(row[col]))
    
    for color in all_colors:
        if color in female_colors:
            female_score += 3
        elif color in bright_colors:
            female_score += 1
        elif color in neutral_colors:
            unisex_score += 1
    
    # 2. 폴더 스타일 체크
    style = row.get('style', '')
    if style in ['페미닌', '로맨틱', '섹시', '리조트']:
        female_score += 5
    elif style in ['젠더리스', '톰보이', '스트리트', '스포티', '밀리터리', '힙합']:
        unisex_score += 3
    
    # 3. JSON 스타일 체크
    main_style = row.get('main_style', '')
    if pd.notna(main_style) and main_style:
        if main_style in ['페미닌', '로맨틱', '섹시', '리조트']:
            female_score += 3
        elif main_style in ['젠더리스', '톰보이', '스트리트', '스포티']:
            unisex_score += 2
    
    # 최종 판단
    if female_score >= 5:
        return 'female', 0.7 + min(female_score / 50, 0.3)
    elif female_score >= 3:
        return 'female', 0.6
    elif unisex_score > 0 or female_score > 0:
        return 'unisex', 0.6
    else:
        return 'unisex', 0.5


def add_gender_predictions(csv_path: str, 
                          output_path: str = 'dataset_with_gender.csv',
                          use_heuristic_validation: bool = True,
                          batch_size: int = 32):
    """데이터셋에 성별 예측 추가 (메타데이터 기반)"""
    
    # CSV 로드
    print(f"\nCSV 로드 중: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"총 {len(df)}개 데이터")
    
    # 성별 예측 결과 저장
    gender_results = []
    
    print("\n성별 예측 중 (메타데이터 기반)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 메타데이터 기반 예측
        gender, confidence = predict_gender_metadata(row)
        
        gender_results.append({
            'gender': gender,
            'gender_confidence': confidence,
            'gender_female_score': confidence if gender == 'female' else 1 - confidence,
            'gender_male_score': confidence if gender == 'male' else 0.1,
            'gender_unisex_score': confidence if gender == 'unisex' else 0.1
        })
    
    # 결과 병합
    gender_df = pd.DataFrame(gender_results)
    df_with_gender = pd.concat([df, gender_df], axis=1)
    
    # 저장
    df_with_gender.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n성별 예측 완료: {output_path}")
    
    # 통계
    print("\n=== 성별 분포 ===")
    print(df_with_gender['gender'].value_counts())
    print(f"\n평균 확신도: {df_with_gender['gender_confidence'].mean():.3f}")
    
    # 성별별 주요 카테고리
    print("\n=== 성별별 주요 카테고리 ===")
    for gender in ['female', 'unisex']:
        if gender in df_with_gender['gender'].values:
            print(f"\n{gender}:")
            gender_df = df_with_gender[df_with_gender['gender'] == gender]
            
            # 존재하는 카테고리 컬럼만 확인
            category_cols = [col for col in gender_df.columns if '카테고리' in col]
            
            for col in category_cols[:4]:  # 처음 4개만
                cat_counts = gender_df[col].value_counts().head(3)
                if not cat_counts.empty and cat_counts.iloc[0] > 0:
                    print(f"  {col}: {dict(list(cat_counts.items())[:2])}")
    
    return df_with_gender


if __name__ == '__main__':
    # 샘플링된 데이터에 성별 추가
    df_with_gender = add_gender_predictions(
        csv_path='sampled_dataset.csv',
        output_path='dataset_with_gender.csv',
        use_heuristic_validation=True
    )
    
    print("\n완료!")
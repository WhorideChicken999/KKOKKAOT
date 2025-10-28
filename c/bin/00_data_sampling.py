import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import random
from typing import Dict, List, Tuple

# 경로 설정
BASE_PATH = Path('D:/D_Study/kkokkaot/k_fashion_data')
LABEL_PATH = BASE_PATH / '라벨링데이터'
IMAGE_PATH = BASE_PATH / '원천데이터'

# 23개 스타일 (기타 제외)
STYLES = [
    '레트로', '로맨틱', '리조트', '매니시', '모던', '밀리터리', 
    '섹시', '소피스트케이티드', '스트리트', '스포티', '아방가르드', 
    '오리엔탈', '웨스턴', '젠더리스', '컨트리', '클래식', '키치', 
    '톰보이', '펑크', '페미닌', '프레피', '히피', '힙합'
]

TARGET_SAMPLE_SIZE = 20000


def is_valid_data(data: Dict) -> bool:
    """라벨링이 완전한 데이터인지 확인 (완화된 기준)"""
    dataset_info = data.get('데이터셋 정보', {})
    detail = dataset_info.get('데이터셋 상세설명', {})  # ← 수정
    labeling = detail.get('라벨링', {})  # ← 수정
    
    # 최소 하나의 의류 카테고리에 색상이라도 있으면 유효
    for category in ['상의', '하의', '원피스', '아우터']:
        cat_data = labeling.get(category, [])
        
        if isinstance(cat_data, list) and len(cat_data) > 0:
            item = cat_data[0]
            # 비어있지 않은 객체이고, 색상이나 카테고리 중 하나라도 있으면 OK
            if item and (item.get('색상') or item.get('카테고리')):
                return True
        elif isinstance(cat_data, dict) and cat_data:
            if cat_data.get('색상') or cat_data.get('카테고리'):
                return True
    
    return False


def load_all_labels() -> Dict[str, List[Dict]]:
    """모든 JSON 라벨 파일을 스타일별로 로드 (유효한 데이터만)"""
    style_data = defaultdict(list)
    
    # 경로 존재 확인
    print(f"라벨 경로: {LABEL_PATH}")
    print(f"이미지 경로: {IMAGE_PATH}")
    
    total_files = 0
    valid_files = 0
    
    for style in STYLES:
        style_dir = LABEL_PATH / style
        img_style_dir = IMAGE_PATH / style
        
        if not style_dir.exists() or not img_style_dir.exists():
            continue
            
        json_files = list(style_dir.glob('*.json'))
        total_files += len(json_files)
        
        style_valid = 0
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 유효성 검사
                if not is_valid_data(data):
                    continue
                
                # 이미지 파일 존재 확인
                img_id = json_file.stem
                img_path = img_style_dir / f"{img_id}.jpg"
                
                if img_path.exists():
                    data['style'] = style
                    data['json_path'] = str(json_file)
                    data['image_path'] = str(img_path)
                    data['image_id'] = img_id
                    style_data[style].append(data)
                    style_valid += 1
                    valid_files += 1
                    
            except Exception as e:
                continue
        
        if style_valid > 0:
            print(f"{style}: {style_valid}개 유효 데이터 ({len(json_files)}개 중)")
    
    print(f"\n총 파일: {total_files}개")
    print(f"유효 데이터: {valid_files}개 ({valid_files/total_files*100:.1f}%)")
    
    return style_data


def stratified_sampling(style_data: Dict[str, List[Dict]], 
                       target_size: int = TARGET_SAMPLE_SIZE) -> List[Dict]:
    """스타일별 계층적 샘플링 (방식 B)"""
    
    # 각 스타일별 데이터 개수 확인
    style_counts = {style: len(data) for style, data in style_data.items()}
    total_available = sum(style_counts.values())
    
    print("\n=== 스타일별 데이터 분포 ===")
    for style, count in sorted(style_counts.items(), key=lambda x: x[1]):
        print(f"{style:15s}: {count:6,}개")
    print(f"\n총 가용 데이터: {total_available:,}개")
    
    # 샘플링 전략
    target_per_style = target_size // len(STYLES)
    print(f"\n목표: 스타일당 약 {target_per_style}개씩")
    
    sampled_data = []
    
    for style, data_list in style_data.items():
        available = len(data_list)
        
        if available <= target_per_style:
            # 데이터 부족: 전체 사용
            sample_count = available
            sampled = data_list
            print(f"{style:15s}: 전체 {available}개 사용 (부족)")
        else:
            # 데이터 충분: 랜덤 샘플링
            sample_count = target_per_style
            sampled = random.sample(data_list, sample_count)
            print(f"{style:15s}: {sample_count}개 샘플링")
        
        sampled_data.extend(sampled)
    
    print(f"\n총 샘플링된 데이터: {len(sampled_data):,}개")
    return sampled_data


def extract_metadata(data: Dict) -> Dict:
    """JSON에서 필요한 메타데이터 추출"""
    dataset_info = data.get('데이터셋 정보', {})
    detail = dataset_info.get('데이터셋 상세설명', {})  # ← 수정
    labeling = detail.get('라벨링', {})  # ← 수정
    
    # 스타일 정보 - 폴더명을 메인으로 사용
    style_data = labeling.get('스타일', [])
    if isinstance(style_data, list) and len(style_data) > 0 and style_data[0]:
        style_info = style_data[0]
    elif isinstance(style_data, dict):
        style_info = style_data
    else:
        style_info = {}
    
    metadata = {
        'image_id': data.get('image_id'),
        'image_path': data.get('image_path'),
        'json_path': data.get('json_path'),
        'style': data.get('style'),  # 폴더명 (이게 메인 스타일)
        'main_style': style_info.get('스타일', ''),  # JSON 내부 스타일 (비어있을 수 있음)
        'sub_style': style_info.get('서브스타일', ''),
    }
    
    # 카테고리별 속성 추출
    for category in ['상의', '하의', '원피스', '아우터']:
        cat_data = labeling.get(category, [])
        
        # 배열인지 확인
        if isinstance(cat_data, list) and len(cat_data) > 0 and cat_data[0]:
            item = cat_data[0]
        elif isinstance(cat_data, dict):
            item = cat_data
        else:
            continue
            
        prefix = category
        
        # 각 필드를 안전하게 추출 (없으면 빈 값)
        metadata[f'{prefix}_카테고리'] = item.get('카테고리', '')
        metadata[f'{prefix}_색상'] = item.get('색상', '')
        metadata[f'{prefix}_서브색상'] = item.get('서브색상', '')
        metadata[f'{prefix}_소재'] = item.get('소재', [])
        metadata[f'{prefix}_프린트'] = item.get('프린트', [])
        metadata[f'{prefix}_디테일'] = item.get('디테일', [])
        metadata[f'{prefix}_핏'] = item.get('핏', '')
        metadata[f'{prefix}_기장'] = item.get('기장', '')
        metadata[f'{prefix}_소매기장'] = item.get('소매기장', '')
        metadata[f'{prefix}_넥라인'] = item.get('넥라인', '')
        metadata[f'{prefix}_옷깃'] = item.get('옷깃', '')
    
    return metadata


def create_dataset_csv(sampled_data: List[Dict], output_path: str = 'sampled_dataset.csv'):
    """샘플링된 데이터를 CSV로 저장"""
    records = []
    
    for data in sampled_data:
        metadata = extract_metadata(data)
        records.append(metadata)
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ CSV 저장 완료: {output_path}")
    print(f"  - 총 {len(df)}개 행")
    print(f"  - {len(df.columns)}개 컬럼")
    
    return df


if __name__ == '__main__':
    # 재현성을 위한 시드 설정
    random.seed(42)
    
    print("=== K-Fashion 데이터 샘플링 시작 ===\n")
    
    # 1. 모든 라벨 로드
    print("1. 라벨 파일 로드 중...")
    style_data = load_all_labels()
    
    # 2. 계층적 샘플링
    print("\n2. 계층적 샘플링 수행 중...")
    sampled_data = stratified_sampling(style_data, TARGET_SAMPLE_SIZE)
    
    # 3. CSV 생성
    print("\n3. CSV 파일 생성 중...")
    df = create_dataset_csv(sampled_data)
    
    # 4. 통계 출력
    print("\n=== 샘플링 결과 통계 ===")
    
    if len(df) > 0:
        print(f"\n스타일별 분포:")
        print(df['style'].value_counts().sort_index())
    else:
        print("\n경고: 샘플링된 데이터가 없습니다!")
        print("경로와 폴더명을 확인해주세요.")
    
    print("\n\n✓ 완료!")
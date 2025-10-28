import pandas as pd
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
import warnings
warnings.filterwarnings('ignore')

# Fashion-CLIP 모델 (Hugging Face transformers 사용)
from transformers import CLIPProcessor, CLIPModel

# GPU 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {DEVICE}")

# Fashion-CLIP 모델 로드
# Fashion-focused CLIP: patrickjohncyh/fashion-clip 또는 일반 CLIP 사용
print("\nFashion-CLIP 모델 로드 중...")
MODEL_NAME = "patrickjohncyh/fashion-clip"  # Fashion 특화 모델

try:
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    print(f"Fashion-CLIP 모델 로드 완료: {MODEL_NAME}")
except:
    print(f"Fashion-CLIP 로드 실패, 기본 CLIP 사용")
    MODEL_NAME = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()


def encode_image(image_path: str) -> np.ndarray:
    """이미지를 임베딩 벡터로 변환"""
    try:
        image = Image.open(image_path).convert('RGB')
        
        inputs = processor(
            images=image,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # L2 정규화
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    except Exception as e:
        print(f"임베딩 에러 ({image_path}): {e}")
        # 에러 시 제로 벡터 반환
        embedding_dim = 768 if 'large' in MODEL_NAME else 512
        return np.zeros(embedding_dim)


def create_metadata(row: pd.Series) -> dict:
    """ChromaDB용 메타데이터 생성"""
    metadata = {
        'image_id': str(row.get('image_id', '')),
        'style': str(row.get('style', '')),
        'main_style': str(row.get('main_style', '')),
        'sub_style': str(row.get('sub_style', '')),
        'gender': str(row.get('gender', 'unknown')),
        'gender_confidence': float(row.get('gender_confidence', 0.0)),
    }
    
    # 카테고리별 주요 속성만 포함
    for category in ['상의', '하의', '원피스', '아우터']:
        prefix = f'{category}_'
        
        # 카테고리
        cat = row.get(f'{prefix}카테고리', '')
        if pd.notna(cat) and cat:
            metadata[f'{prefix}카테고리'] = str(cat)
        
        # 색상
        color = row.get(f'{prefix}색상', '')
        if pd.notna(color) and color:
            metadata[f'{prefix}색상'] = str(color)
        
        # 핏
        fit = row.get(f'{prefix}핏', '')
        if pd.notna(fit) and fit:
            metadata[f'{prefix}핏'] = str(fit)
    
    return metadata


def build_chromadb(csv_path: str,
                   collection_name: str = "fashion_collection",
                   persist_directory: str = "./chroma_db",
                   batch_size: int = 32):
    """ChromaDB 구축"""
    
    # DataFrame 로드
    print(f"\nCSV 로드: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"총 {len(df)}개 데이터")
    
    # ChromaDB 클라이언트 생성
    print(f"\nChromaDB 초기화: {persist_directory}")
    client = chromadb.PersistentClient(path=persist_directory)
    
    # 기존 컬렉션 삭제 (있으면)
    try:
        client.delete_collection(name=collection_name)
        print(f"기존 컬렉션 '{collection_name}' 삭제")
    except:
        pass
    
    # 새 컬렉션 생성
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "K-Fashion 의류 이미지 임베딩"}
    )
    print(f"컬렉션 '{collection_name}' 생성 완료")
    
    # 배치 처리
    print("\n임베딩 생성 및 ChromaDB 저장 중...")
    
    for start_idx in tqdm(range(0, len(df), batch_size)):
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        batch_ids = []
        batch_embeddings = []
        batch_metadatas = []
        batch_documents = []
        
        for idx, row in batch_df.iterrows():
            # 임베딩 생성
            embedding = encode_image(row['image_path'])
            
            # 메타데이터 생성
            metadata = create_metadata(row)
            
            # 검색용 텍스트 문서 생성
            doc_parts = [
                f"스타일: {row.get('main_style', '')}",
                f"성별: {row.get('gender', '')}",
            ]
            
            # 카테고리 정보 추가
            for cat in ['상의', '하의', '원피스', '아우터']:
                cat_name = row.get(f'{cat}_카테고리', '')
                if pd.notna(cat_name) and cat_name:
                    doc_parts.append(f"{cat}: {cat_name}")
            
            document = " | ".join(doc_parts)
            
            batch_ids.append(f"img_{row['image_id']}")
            batch_embeddings.append(embedding.tolist())
            batch_metadatas.append(metadata)
            batch_documents.append(document)
        
        # ChromaDB에 배치 추가
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_documents
        )
    
    print(f"\n✓ ChromaDB 구축 완료")
    print(f"  - 컬렉션: {collection_name}")
    print(f"  - 총 아이템: {collection.count()}")
    print(f"  - 저장 경로: {persist_directory}")
    
    return collection


def search_similar_items(collection,
                        query_image_path: str,
                        n_results: int = 10,
                        filter_metadata: dict = None):
    """유사 이미지 검색"""
    
    print(f"\n검색 쿼리: {query_image_path}")
    
    # 쿼리 이미지 임베딩
    query_embedding = encode_image(query_image_path)
    
    # 검색
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        where=filter_metadata  # 필터링 (예: {"gender": "female"})
    )
    
    print(f"\n검색 결과 (상위 {n_results}개):")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n{i+1}. 유사도: {1-distance:.4f}")
        print(f"   {doc}")
        print(f"   성별: {metadata.get('gender', 'unknown')}")
    
    return results


if __name__ == '__main__':
    # ChromaDB 구축
    collection = build_chromadb(
        csv_path='dataset_with_gender.csv',
        collection_name='fashion_collection',
        persist_directory='./chroma_db',
        batch_size=32
    )
    
    print("\n✓ 모든 작업 완료")
    
    # 테스트 검색 (첫 번째 이미지로)
    df = pd.read_csv('dataset_with_gender.csv')
    if len(df) > 0:
        test_image = df.iloc[0]['image_path']
        print("\n=== 테스트 검색 ===")
        search_similar_items(
            collection=collection,
            query_image_path=test_image,
            n_results=5
        )
import chromadb
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# GPU 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fashion-CLIP 모델 로드 (검색용)
print("Fashion-CLIP 모델 로드 중...")
MODEL_NAME = "patrickjohncyh/fashion-clip"

try:
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
except:
    MODEL_NAME = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()


def encode_image(image_path: str) -> np.ndarray:
    """이미지를 임베딩 벡터로 변환"""
    image = Image.open(image_path).convert('RGB')
    
    inputs = processor(
        images=image,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().numpy()[0]


# 기존 ChromaDB 로드
print("\n기존 ChromaDB 로드 중...")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="fashion_collection")

print(f"컬렉션 로드 완료")
print(f"  - 총 아이템: {collection.count()}")


def search_similar_fashion(
    query_image_path: str,
    n_results: int = 10,
    gender_filter: str = None  # 'female', 'unisex', None
):
    """유사 의류 검색"""
    
    print(f"\n검색 쿼리: {query_image_path}")
    
    # 쿼리 이미지 임베딩
    query_embedding = encode_image(query_image_path)
    
    # 필터 설정
    where_filter = None
    if gender_filter:
        where_filter = {"gender": gender_filter}
    
    # 검색
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        where=where_filter
    )
    
    print(f"\n검색 결과 (상위 {n_results}개):")
    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n{i+1}. 유사도: {1-distance:.4f}")
        print(f"   {doc}")
        print(f"   성별: {metadata.get('gender')}")
        print(f"   이미지 ID: {metadata.get('image_id')}")
    
    return results


# 테스트 검색
if __name__ == '__main__':
    # 예시 1: 일반 검색
    test_image = "D:/D_Study/kkokkaot/k_fashion_data/원천데이터/레트로/100317.jpg"
    search_similar_fashion(test_image, n_results=5)
    
    # 예시 2: 여성복만 검색
    print("\n\n=== 여성복만 검색 ===")
    search_similar_fashion(test_image, n_results=5, gender_filter='female')
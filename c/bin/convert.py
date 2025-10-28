import json
import os
from pathlib import Path
def convert_kfashion_to_training_format(json_path: str, output_dir: str = "./converted_data"):
    """
    K-Fashion JSON을 YOLO, SAM2, CNN 학습용 형식으로 변환
    
    Args:
        json_path: 원본 K-Fashion JSON 파일 경로
        output_dir: 변환된 파일들을 저장할 디렉토리
    """
    
    # 출력 디렉토리 생성
    yolo_dir = Path(output_dir) / "yolo"
    sam_dir = Path(output_dir) / "sam"
    cnn_dir = Path(output_dir) / "cnn"
    
    for dir_path in [yolo_dir, sam_dir, cnn_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # JSON 파일 읽기
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 파일 번호 및 이미지 파일명 생성
    item_no = data['데이터셋 정보']['파일 번호']
    image_info = data['이미지 정보']
    json_filename = Path(json_path).stem  # ✅ 확장자 제외한 파일명
    image_filename = f"{json_filename}.jpg"  # ✅ .jpg로 변환
    img_width = image_info['이미지 너비']
    img_height = image_info['이미지 높이']
    dataset_detail = data['데이터셋 정보']['데이터셋 상세설명']
    
    # 카테고리 매핑 (COCO 형식용)
    category_mapping = {
        '아우터': 0,
        '상의': 1,
        '하의': 2,
        '원피스': 3
    }
    
    # ========================================
    # 1. YOLO용 COCO 형식 (렉트좌표)
    # ========================================
    coco_data = {
        "images": [{
            "id": item_no,
            "file_name": image_filename,  # ✅ 22614.jpg
            "width": img_width,
            "height": img_height
        }],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "outer"},
            {"id": 1, "name": "top"},
            {"id": 2, "name": "bottom"},
            {"id": 3, "name": "dress"}
        ]
    }
    
    annotation_id = 1
    rect_coords = dataset_detail['렉트좌표']
    
    for category_name, boxes in rect_coords.items():
        if boxes and boxes[0]:
            for box in boxes:
                if box:
                    x = box['X좌표']
                    y = box['Y좌표']
                    width = box['가로']
                    height = box['세로']
                    
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": item_no,
                        "category_id": category_mapping[category_name],
                        "bbox": [x, y, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })
                    annotation_id += 1
    
    # YOLO JSON 저장
    yolo_output_path = yolo_dir / f"yolo_{item_no}.json"
    with open(yolo_output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)
    
    # ========================================
    # 2. SAM2용 형식 (폴리곤좌표)
    # ========================================
    sam_data = {
        "image_id": item_no,
        "file_name": image_filename,  # ✅ 22614.jpg
        "width": img_width,
        "height": img_height,
        "annotations": []
    }
    
    polygon_coords = dataset_detail['폴리곤좌표']
    
    for category_name, polygons in polygon_coords.items():
        if polygons and polygons[0]:
            for polygon in polygons:
                if polygon:
                    points = []
                    idx = 1
                    while f'X좌표{idx}' in polygon:
                        x = polygon[f'X좌표{idx}']
                        y = polygon[f'Y좌표{idx}']
                        points.append([x, y])
                        idx += 1
                    
                    sam_data["annotations"].append({
                        "category": category_name,
                        "category_id": category_mapping[category_name],
                        "segmentation": points,
                        "num_points": len(points)
                    })
    
    # SAM JSON 저장
    sam_output_path = sam_dir / f"sam_{item_no}.json"
    with open(sam_output_path, 'w', encoding='utf-8') as f:
        json.dump(sam_data, f, ensure_ascii=False, indent=2)
    
    # ========================================
    # 3. CNN용 속성 정보 (라벨링)
    # ========================================
    cnn_data = {
        "image_id": item_no,
        "file_name": image_filename,  # ✅ 22614.jpg
        "items": {}
    }
    
    labeling = dataset_detail['라벨링']
    
    # 스타일 정보
    if labeling['스타일'] and labeling['스타일'][0]:
        cnn_data['style'] = labeling['스타일'][0]
    else:
        cnn_data['style'] = {}
    
    # 각 카테고리별 속성
    for category_name in ['아우터', '상의', '하의', '원피스']:
        category_data = labeling[category_name]
        
        if category_data and category_data[0]:
            item_info = category_data[0]
            
            cnn_data['items'][category_name] = {
                "카테고리": item_info.get('카테고리', ''),
                "색상": item_info.get('색상', ''),
                "서브색상": item_info.get('서브색상', ''),
                "기장": item_info.get('기장', ''),
                "소매기장": item_info.get('소매기장', ''),
                "넥라인": item_info.get('넥라인', ''),
                "칼라": item_info.get('칼라', ''),
                "핏": item_info.get('핏', ''),
                "소재": item_info.get('소재', []),
                "프린트": item_info.get('프린트', []),
                "디테일": item_info.get('디테일', [])
            }
        else:
            cnn_data['items'][category_name] = {}
    
    # CNN JSON 저장
    cnn_output_path = cnn_dir / f"cnn_{item_no}.json"
    with open(cnn_output_path, 'w', encoding='utf-8') as f:
        json.dump(cnn_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 변환 완료: {item_no}")
    print(f"   - YOLO: {yolo_output_path}")
    print(f"   - SAM:  {sam_output_path}")
    print(f"   - CNN:  {cnn_output_path}")
    
    return {
        'yolo': str(yolo_output_path),
        'sam': str(sam_output_path),
        'cnn': str(cnn_output_path)
    }

def batch_convert(input_dir: str, output_dir: str = "./converted_data"):
    """
    디렉토리 내 모든 K-Fashion JSON 파일을 일괄 변환
    
    Args:
        input_dir: 원본 JSON 파일들이 있는 디렉토리
        output_dir: 변환된 파일들을 저장할 디렉토리
    """
    
    json_files = list(Path(input_dir).glob("*.json"))
    
    print(f"\n{'='*60}")
    print(f"📦 총 {len(json_files)}개 파일 변환 시작")
    print(f"{'='*60}\n")
    
    success_count = 0
    fail_count = 0
    
    for json_file in json_files:
        try:
            convert_kfashion_to_training_format(str(json_file), output_dir)
            success_count += 1
        except Exception as e:
            print(f"❌ 실패: {json_file.name} - {e}")
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ 변환 완료: {success_count}개")
    print(f"❌ 실패: {fail_count}개")
    print(f"{'='*60}\n")


# ========================================
# 사용 예시
# ========================================
if __name__ == "__main__":
    
    # # 1. 단일 파일 변환
    # convert_kfashion_to_training_format(
    #     json_path="./k_fashion_data/kf_labels/101.json",
    #     output_dir="./converted_data"
    # )
    
    # 2. 디렉토리 전체 일괄 변환
    batch_convert(
        input_dir="./k_fashion_data/kf_labels",
        output_dir="./converted_data"
    )
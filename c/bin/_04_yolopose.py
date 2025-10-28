import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt

# YOLOv11-pose 모델 로드
print("YOLOv11-pose 모델 로드 중...")
# model = YOLO('yolo11n-pose.pt')  # nano 버전 (빠름)
model = YOLO('yolo11m-pose.pt')  # medium 버전 (정확)
print("모델 로드 완료")

# COCO 키포인트 인덱스
KEYPOINT_NAMES = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle'     # 16
]


def detect_person_and_keypoints(image_path: str):
    """이미지에서 사람 검출 및 키포인트 추출"""
    
    # 이미지 로드
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 열 수 없습니다: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # YOLO 추론
    results = model(image_path, verbose=False)
    
    if len(results[0].boxes) == 0:
        print("사람이 감지되지 않았습니다")
        return None, image_rgb
    
    # 첫 번째 사람 선택 (신뢰도 가장 높은)
    result = results[0]
    
    # 키포인트 추출
    if result.keypoints is None or len(result.keypoints.data) == 0:
        print("키포인트를 감지하지 못했습니다")
        return None, image_rgb
    
    keypoints = result.keypoints.data[0].cpu().numpy()  # [17, 3] (x, y, confidence)
    
    return keypoints, image_rgb


def calculate_waist_position(keypoints: np.ndarray) -> int:
    """허리 위치 계산 (상하의 분리선)"""
    
    # 어깨 (5, 6)와 골반 (11, 12) 중간 지점을 허리로 간주
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    
    # 신뢰도 체크
    valid_points = []
    if left_shoulder[2] > 0.5:  # confidence > 0.5
        valid_points.append(left_shoulder)
    if right_shoulder[2] > 0.5:
        valid_points.append(right_shoulder)
    if left_hip[2] > 0.5:
        valid_points.append(left_hip)
    if right_hip[2] > 0.5:
        valid_points.append(right_hip)
    
    if len(valid_points) < 2:
        print("충분한 키포인트가 감지되지 않았습니다")
        return None
    
    # 어깨와 골반의 평균 Y 좌표
    shoulder_y = np.mean([p[1] for p in [left_shoulder, right_shoulder] if p[2] > 0.5])
    hip_y = np.mean([p[1] for p in [left_hip, right_hip] if p[2] > 0.5])
    
    # 허리는 어깨와 골반 사이 (약 60% 지점)
    waist_y = int(shoulder_y + (hip_y - shoulder_y) * 0.6)
    
    return waist_y


def separate_top_bottom(image: np.ndarray, waist_y: int):
    """이미지를 상의/하의로 분리"""
    
    height, width = image.shape[:2]
    
    # 상의: 0 ~ waist_y
    top_image = image[0:waist_y, :]
    
    # 하의: waist_y ~ 끝
    bottom_image = image[waist_y:height, :]
    
    return top_image, bottom_image


def visualize_separation(image: np.ndarray, keypoints: np.ndarray, waist_y: int):
    """분리 결과 시각화"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 원본 + 키포인트
    axes[0].imshow(image)
    axes[0].set_title('Origin + Keypoint')
    
    # 키포인트 그리기
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5:
            axes[0].plot(x, y, 'ro', markersize=8)
            axes[0].text(x, y, str(i), color='white', fontsize=8)
    
    # 허리선 표시
    if waist_y:
        axes[0].axhline(y=waist_y, color='yellow', linewidth=2, linestyle='--')
        axes[0].text(10, waist_y, 'WAIST', color='yellow', fontsize=12, 
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    axes[0].axis('off')
    
    # 상의
    top_image, bottom_image = separate_top_bottom(image, waist_y)
    axes[1].imshow(top_image)
    axes[1].set_title('Top')
    axes[1].axis('off')
    
    # 하의
    axes[2].imshow(bottom_image)
    axes[2].set_title('Bottom')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


def process_fashion_image(image_path: str, visualize: bool = True):
    """전체 파이프라인"""
    
    print(f"\n처리 중: {image_path}")
    
    # 1. 키포인트 검출
    keypoints, image = detect_person_and_keypoints(image_path)
    
    if keypoints is None:
        print("처리 실패")
        return None
    
    # 2. 허리 위치 계산
    waist_y = calculate_waist_position(keypoints)
    
    if waist_y is None:
        print("허리 위치를 찾을 수 없습니다")
        return None
    
    print(f"허리 위치: Y = {waist_y}")
    
    # 3. 상하의 분리
    top_image, bottom_image = separate_top_bottom(image, waist_y)
    
    print(f"상의 크기: {top_image.shape}")
    print(f"하의 크기: {bottom_image.shape}")
    
    # 4. 시각화
    if visualize:
        visualize_separation(image, keypoints, waist_y)
    
    return {
        'original': image,
        'keypoints': keypoints,
        'waist_y': waist_y,
        'top': top_image,
        'bottom': bottom_image
    }


if __name__ == '__main__':
    # 테스트 이미지 경로
    test_image = "D:/D_Study/kkokkaot/k_fashion_data/원천데이터/레트로/100317.jpg"
    
    # 처리
    result = process_fashion_image(test_image, visualize=True)
    
    if result:
        print("\n처리 완료")
        
        # 결과 저장 (선택)
        cv2.imwrite('./samples/user/top.jpg', cv2.cvtColor(result['top'], cv2.COLOR_RGB2BGR))
        cv2.imwrite('./samples/user/bottom.jpg', cv2.cvtColor(result['bottom'], cv2.COLOR_RGB2BGR))
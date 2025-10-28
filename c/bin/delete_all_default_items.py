import requests

# 모든 기본 아이템 삭제
try:
    response = requests.delete("http://127.0.0.1:4000/api/default-items")
    print(f"상태 코드: {response.status_code}")
    print(f"응답: {response.json()}")
except Exception as e:
    print(f"오류: {e}")

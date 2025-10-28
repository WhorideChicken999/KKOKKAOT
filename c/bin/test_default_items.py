import requests

# 기본 아이템 AI 분석 API 호출
try:
    response = requests.post("http://127.0.0.1:4000/api/process-default-items")
    print(f"상태 코드: {response.status_code}")
    print(f"응답: {response.json()}")
except Exception as e:
    print(f"오류: {e}")

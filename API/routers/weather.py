"""
날씨 API
- OpenWeatherMap 연동
"""
from fastapi import APIRouter
import requests
import os
from datetime import datetime
from config import settings

router = APIRouter(prefix="/api", tags=["날씨"])


@router.get("/weather")
def get_weather(city: str = "Seoul", lat: float = None, lon: float = None):
    """실시간 날씨 정보 조회 (OpenWeatherMap API 사용)"""
    
    # OpenWeatherMap API 키 (.env 파일에서 로드)
    API_KEY = settings.WEATHER_API_KEY
    
    # API 키가 없거나 유효하지 않으면 기본값 반환
    if not API_KEY or API_KEY == "your_api_key_here" or API_KEY == "":
        return {
            "success": True,
            "temperature": 22,
            "feels_like": 22,
            "weather": "Clouds",
            "description": "흐림",
            "icon": "☁️",
            "style_tip": "흐림 · 가벼운 레이어드 스타일링 추천",
            "city": city,
            "date": datetime.now().strftime("%m월 %d일")
        }
    
    try:
        # OpenWeatherMap API 호출 (위도/경도 또는 도시 이름)
        if lat is not None and lon is not None:
            # 위도/경도로 조회 (더 정확함)
            url = f"{settings.WEATHER_API_URL}?lat={lat}&lon={lon}&appid={API_KEY}&units=metric&lang=kr"
        else:
            # 도시 이름으로 조회
            url = f"{settings.WEATHER_API_URL}?q={city}&appid={API_KEY}&units=metric&lang=kr"
        
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            # 날씨 정보 추출
            temp = round(data['main']['temp'])
            feels_like = round(data['main']['feels_like'])
            weather_main = data['weather'][0]['main']
            weather_desc = data['weather'][0]['description']
            
            # 날씨에 따른 아이콘 및 스타일 추천
            weather_icon_map = {
                'Clear': '☀️',
                'Clouds': '☁️',
                'Rain': '🌧️',
                'Drizzle': '🌦️',
                'Thunderstorm': '⛈️',
                'Snow': '❄️',
                'Mist': '🌫️',
                'Fog': '🌫️',
            }
            
            # 온도에 따른 스타일 추천
            if temp >= 28:
                style_tip = "더위 주의 · 통풍이 잘 되는 가벼운 옷 추천"
            elif temp >= 23:
                style_tip = "쾌적한 날씨 · 가벼운 여름 스타일 추천"
            elif temp >= 20:
                style_tip = "선선한 날씨 · 가벼운 레이어드 스타일링 추천"
            elif temp >= 17:
                style_tip = "약간 쌀쌀 · 얇은 아우터 추천"
            elif temp >= 12:
                style_tip = "쌀쌀한 날씨 · 가디건이나 자켓 추천"
            elif temp >= 9:
                style_tip = "추운 날씨 · 따뜻한 아우터 필수"
            elif temp >= 5:
                style_tip = "매우 추움 · 두꺼운 코트와 목도리 추천"
            else:
                style_tip = "한파 · 패딩과 방한 용품 필수"
            
            return {
                "success": True,
                "temperature": temp,
                "feels_like": feels_like,
                "weather": weather_main,
                "description": weather_desc,
                "icon": weather_icon_map.get(weather_main, '☁️'),
                "style_tip": f"{weather_desc} · {style_tip}",
                "city": city,
                "date": datetime.now().strftime("%m월 %d일")
            }
        else:
            # API 호출 실패 시 기본값 반환
            return {
                "success": False,
                "message": "날씨 정보를 가져올 수 없습니다.",
                "temperature": 22,
                "weather": "Clouds",
                "description": "흐림",
                "icon": "☁️",
                "style_tip": "맑음 · 가벼운 레이어드 스타일링 추천",
                "date": datetime.now().strftime("%m월 %d일")
            }
            
    except Exception as e:
        print(f"❌ 날씨 API 오류: {e}")
        # 오류 시 기본값 반환
        return {
            "success": False,
            "message": f"오류 발생: {str(e)}",
            "temperature": 22,
            "weather": "Clouds",
            "description": "흐림",
            "icon": "☁️",
            "style_tip": "맑음 · 가벼운 레이어드 스타일링 추천",
            "date": datetime.now().strftime("%m월 %d일")
        }


"""
Pydantic 모델 정의
API 요청/응답 데이터 구조를 정의합니다.
"""
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime


# 회원가입 요청
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    ageGroup: int = 20
    stylePreferences: List[str] = []


# 로그인 요청
class LoginRequest(BaseModel):
    email: str
    password: str


# 사용자 응답
class UserResponse(BaseModel):
    user_id: int
    name: str
    email: str


# API 응답 (성공)
class SuccessResponse(BaseModel):
    success: bool = True
    message: str
    data: Optional[dict] = None


# API 응답 (실패)
class ErrorResponse(BaseModel):
    success: bool = False
    message: str
    error: Optional[str] = None


# 옷장 아이템
class WardrobeItem(BaseModel):
    item_id: int
    user_id: int
    image_path: str
    style: Optional[str] = None
    created_at: datetime


# 추천 결과
class RecommendationItem(BaseModel):
    item_id: int
    image_path: str
    similarity_score: float
    style: Optional[str] = None


# 날씨 정보
class WeatherInfo(BaseModel):
    temperature: float
    description: str
    humidity: int
    wind_speed: float


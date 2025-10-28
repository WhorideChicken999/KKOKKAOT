"""
패션 분석 파이프라인 패키지

사용 예시:
```python
from pipeline import FashionPipeline

pipeline = FashionPipeline()
result = pipeline.process("photo.jpg", user_id=1)
print(f"Gender: {result['gender']}")
print(f"Style: {result['style']}")
```
"""

from .main import FashionPipeline, analyze_fashion_item
from .loader import ModelLoader
from .predictor import FashionPredictor
from .database import DatabaseManager

__all__ = [
    'FashionPipeline',
    'analyze_fashion_item',
    'ModelLoader',
    'FashionPredictor',
    'DatabaseManager'
]



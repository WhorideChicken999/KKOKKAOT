#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM 세그멘테이션 실행 스크립트
YOLO + SAM을 사용한 이미지 세그멘테이션 실행
"""

import sys
import os
from pathlib import Path

# 현재 디렉토리를 프로젝트 루트로 설정
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# SAM 세그멘테이션 모듈 임포트
from c.sam_segmentation import main

if __name__ == "__main__":
    print("🚀 SAM 세그멘테이션 실행!")
    print(f"📁 프로젝트 루트: {project_root}")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

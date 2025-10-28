#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN 라벨 학습 실행 스크립트
간단한 실행을 위한 래퍼 스크립트
"""

import sys
import os
from pathlib import Path

# 현재 디렉토리를 프로젝트 루트로 설정
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# CNN 학습 모듈 임포트
from cnn_label_training import main

if __name__ == "__main__":
    print("🚀 CNN 라벨 학습 실행!")
    print(f"📁 프로젝트 루트: {project_root}")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

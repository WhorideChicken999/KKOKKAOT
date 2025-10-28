# _warmup_models.py
# 목적: 임베딩 모델 + LLM이 현재 환경에서 정상 로드되는지 사전 확인
# 이 버전은:
#  - 캐시 경로를 D:로 강제 (C: 안 씀)
#  - Qwen2.5-0.5B-Instruct만 로드 (가벼운 쪽)
#  - offload도 D:로
#
# 실행 방법 (CMD든 PowerShell이든 둘 다 똑같이 됨):
#   D:\kkokkaot_HJ\kkokkaot_venv\Scripts\python.exe D:\kkokkaot_HJ\API\_warmup_models.py

import os

########################################
# 0) 먼저 캐시/오프로딩 위치를 D:로 강제로 설정
########################################
os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\hf_cache\transformers"

OFFLOAD_DIR = r"D:\hf_offload"

# 폴더 없으면 만들어두기
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
os.makedirs(OFFLOAD_DIR, exist_ok=True)

########################################
# 1) 이제부터 라이브러리 import (중요!)
########################################
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def main():
    # -------------------------
    # 1. 임베딩 모델 (RAG용)
    # -------------------------
    print("[1/2] 임베딩 모델 로드 시작: intfloat/multilingual-e5-small")
    emb = SentenceTransformer("intfloat/multilingual-e5-small")
    v = emb.encode(["passage: 테스트 문장입니다."], normalize_embeddings=True)
    print(f"임베딩 OK, shape={v.shape}")

    # -------------------------
    # 2. LLM (추천 멘트용)
    # -------------------------
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"[2/2] LLM 로드 시작: {model_id}")
    print(f"- HF_HOME={os.environ['HF_HOME']}")
    print(f"- TRANSFORMERS_CACHE={os.environ['TRANSFORMERS_CACHE']}")
    print(f"- offload_folder={OFFLOAD_DIR}")

    # 토크나이저
    tok = AutoTokenizer.from_pretrained(model_id)

    # 모델 본체
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",         # accelerate가 알아서 CPU 등에 배치
        dtype="auto",              # dtype 자동
        low_cpu_mem_usage=True,    # 로딩 시 RAM 절약
        offload_folder=OFFLOAD_DIR # 부족하면 D:\hf_offload 로 스왑
    )

    # 간단 테스트 (진짜 생성 되는지)
    inputs = tok("테스트용 한 줄입니다.", return_tensors="pt").to(mdl.device)
    out = mdl.generate(**inputs, max_new_tokens=8)
    decoded = tok.decode(out[0], skip_special_tokens=True)

    print("LLM OK, sample:", decoded[:200])


if __name__ == "__main__":
    main()

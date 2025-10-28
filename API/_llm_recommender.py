# _llm_recommender.py
#
# 역할:
# - LLM + RAG + (가능하면) DB 기반 코디 추천 어시스턴트
# - DB 없어도 최소한 대화랑 컨텍스트 추출은 돌아감
#
# 전체를 D: 드라이브에서만 쓰도록 강제로 설정:
#   - 모델 캐시        -> D:\hf_cache
#   - 모델 오프로딩    -> D:\hf_offload
#   - RAG 스토어 파일  -> D:\rag_store\rag_store.pkl
#
# 필요한 pip:
#   pip install psycopg2-binary transformers sentence-transformers scikit-learn accelerate


import os
import json
import re
import pickle
from typing import Dict, List, Optional

import psycopg2
from psycopg2 import OperationalError

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer


# ======================================
# 0. 전역 경로 설정 (전부 D 드라이브)
# ======================================

D_CACHE_DIR = r"D:\hf_cache"                 # 허깅페이스 모델 캐시
D_OFFLOAD_DIR = r"D:\hf_offload"             # 모델 오프로딩 폴더
D_RAG_STORE = r"D:\rag_store\rag_store.pkl"  # RAG 저장 위치

# 폴더가 없으면 만들어둔다
for _p in [D_CACHE_DIR, D_OFFLOAD_DIR, os.path.dirname(D_RAG_STORE)]:
    os.makedirs(_p, exist_ok=True)

# Hugging Face 캐시를 D:로 강제 (C: 용량 안 쓰게)
os.environ["HF_HOME"] = D_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = os.path.join(D_CACHE_DIR, "transformers")


# ======================================
# 1. RAG Store
# ======================================

class LocalRAGStore:
    """
    - sentence-transformers로 문장 임베딩
    - NearestNeighbors(cosine)로 유사도 검색
    - 로컬 파일에 pickle로 저장
    """

    DEFAULT_MODEL = "intfloat/multilingual-e5-small"
    DEFAULT_STORE_PATH = D_RAG_STORE  # D드라이브에 고정

    def __init__(self, model_name: str = None, store_path: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self.store_path = store_path or self.DEFAULT_STORE_PATH

        # 가벼운 임베딩 모델
        self.model = SentenceTransformer(self.model_name)

        # 메모리 구조
        self.texts: List[str] = []
        self.metas: List[Dict] = []
        self.embeds: Optional[np.ndarray] = None
        self.nn: Optional[NearestNeighbors] = None

        # 기존 rag_store.pkl 있으면 로드
        if os.path.exists(self.store_path):
            self._load()

    def is_empty(self) -> bool:
        return (self.embeds is None) or (len(self.texts) == 0)

    def add(self, texts: List[str], metas: Optional[List[Dict]] = None):
        """
        새 문장들만 추가해서 인덱스 업데이트
        """
        if metas is None:
            metas = [{} for _ in texts]
        assert len(texts) == len(metas)

        new_embeds = self._encode(texts)

        if self.embeds is None:
            self.embeds = new_embeds
            self.texts = list(texts)
            self.metas = list(metas)
        else:
            self.embeds = np.vstack([self.embeds, new_embeds])
            self.texts.extend(texts)
            self.metas.extend(metas)

        self._rebuild_index()
        self._save()

    def rebuild(self, items: List[tuple]):
        """
        전체 갈아끼우기.
        items = [(text, meta_dict), ...]
        """
        if not items:
            self.texts, self.metas, self.embeds, self.nn = [], [], None, None
            self._save()
            return

        texts = [t for t, _ in items]
        metas = [m for _, m in items]

        self.texts = texts
        self.metas = metas
        self.embeds = self._encode(texts)

        self._rebuild_index()
        self._save()

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        query랑 비슷한 조각 k개 리턴
        """
        if self.embeds is None or self.nn is None or len(self.texts) == 0:
            return []

        q = self._encode([query])
        dists, idxs = self.nn.kneighbors(q, n_neighbors=min(k, len(self.texts)))

        out = []
        for dist, idx in zip(dists[0], idxs[0]):
            out.append({
                "text": self.texts[idx],
                "score": float(1.0 - dist),  # cosine distance -> 유사도
                "meta": self.metas[idx],
            })
        return out

    # ---------- 내부 유틸 ----------

    def _encode(self, texts: List[str]) -> np.ndarray:
        """
        e5 모델은 "passage: ..." prefix 주는 게 베스트 프랙티스
        """
        pre = [f"passage: {t}" for t in texts]
        v = self.model.encode(
            pre,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return v.astype(np.float32)

    def _rebuild_index(self):
        self.nn = NearestNeighbors(metric="cosine")
        self.nn.fit(self.embeds)

    def _save(self):
        os.makedirs(os.path.dirname(self.store_path), exist_ok=True)
        with open(self.store_path, "wb") as f:
            pickle.dump({
                "model_name": self.model_name,
                "texts": self.texts,
                "metas": self.metas,
                "embeds": self.embeds,
            }, f)

    def _load(self):
        with open(self.store_path, "rb") as f:
            data = pickle.load(f)
        self.model_name = data.get("model_name", self.DEFAULT_MODEL)
        self.texts = data.get("texts", [])
        self.metas = data.get("metas", [])
        self.embeds = data.get("embeds", None)

        # 인덱스 복구
        if self.embeds is not None and len(self.texts) == len(self.metas):
            self._rebuild_index()


# ======================================
# 2. LLMRecommender
# ======================================

class LLMRecommender:
    """
    LLM + RAG + (가능하면) DB 추천
    DB 연결 실패해도 최소 대화는 동작하도록 설계
    """

    # 경로
    CACHE_DIR = D_CACHE_DIR
    OFFLOAD_DIR = D_OFFLOAD_DIR

    # 사용할 LLM
    MODEL_NAME_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"

    # PostgreSQL 접속 정보
    DB_CONFIG_HARD_DEFAULT = {
        "host": "localhost",
        "port": 5432,
        "dbname": "kkokkaot",
        "user": "postgres",
        "password": "1234",  # 필요하면 여기서 바꾸면 됨
    }

    def __init__(
        self,
        db_config: dict | None = None,
        model_name: str | None = None
    ):
        print("\n🤖 LLM 추천 시스템 초기화 중...")

        # ------------------
        # (1) DB 연결 시도
        # ------------------
        self.db_conn = None
        cfg = db_config if db_config is not None else self.DB_CONFIG_HARD_DEFAULT
        try:
            self.db_conn = psycopg2.connect(**cfg)
            print("✅ PostgreSQL 연결 완료")
        except OperationalError as e:
            print("⚠️ DB 연결 실패 (일단 LLM만 쓸게):", e)
            self.db_conn = None

        # ------------------
        # (2) LLM 로드
        # ------------------
        model_id = model_name or self.MODEL_NAME_DEFAULT

        # D: 경로 재확인
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.OFFLOAD_DIR, exist_ok=True)

        # 혹시 모를 상황 대비해서 한 번 더 환경변수 고정
        os.environ["HF_HOME"] = self.CACHE_DIR
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.CACHE_DIR, "transformers")

        print(f"📥 모델 로딩 중: {model_id}")
        print(f"   cache_dir   = {self.CACHE_DIR}")
        print(f"   offload_dir = {self.OFFLOAD_DIR}")

        # 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=self.CACHE_DIR,
        )

        # 모델 본체 (auto device map + offload)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=self.CACHE_DIR,
            device_map="auto",          # accelerate 가 램/디스크/VRAM 자동 분배
            dtype="auto",
            low_cpu_mem_usage=True,
            offload_folder=self.OFFLOAD_DIR,
        )

        print("✅ LLM 모델 로드 완료")

        # ------------------
        # (3) 상태들
        # ------------------
        self.conversation_history = {}

        # RAG 준비
        self.rag = LocalRAGStore(store_path=D_RAG_STORE)
        self._maybe_seed_rag()
        print("✅ RAG 스토어 준비 완료\n")

    # ----------------------------------
    # RAG 시드
    # ----------------------------------
    def _maybe_seed_rag(self):
        """
        RAG가 비어 있으면 기본 지식 채워준다.
        """
        if not self.rag.is_empty():
            return

        seed = [
            ("비 오는 날엔 어두운 하의와 방수 아우터가 얼룩에 안전하다.", {"tag": "weather"}),
            ("데이트엔 상의는 깔끔, 하의는 과하지 않게 균형을 잡아라.", {"tag": "occasion"}),
            ("회사 회의엔 셔츠/블라우스 + 재킷 조합이 가장 안정적이다.", {"tag": "work"}),
            ("더운 날엔 코튼/린넨, 여유 핏이 쾌적하다.", {"tag": "temp"}),
            ("추운 날엔 니트/울/패딩 계열로 보온을 확보하라.", {"tag": "temp"}),
            ("러블리 무드는 리본·프릴은 1~2포인트만, 실루엣은 단순하게.", {"tag": "style"}),
            ("워크웨어는 치어 코트·카펜터 팬츠·러기드 슈즈가 핵심이다.", {"tag": "style"}),
            ("프레피는 셔츠·니트베스트·플리츠, 로퍼가 상징적이다.", {"tag": "style"}),
            ("스포티룩은 기능성 원단과 조거/트랙팬츠가 기본이다.", {"tag": "style"}),
            ("와이드 하의엔 상의를 크롭/슬림으로 맞추어 밸런스를 잡아라.", {"tag": "silhouette"}),
        ]
        self.rag.rebuild(seed)
        print(f"✅ RAG 시드 문서 주입 완료: {len(seed)}개")

    # ----------------------------------
    # 메인 chat() 로직
    # ----------------------------------
    def chat(self, user_id: int, user_message: str, selected_item_ids: list = None) -> Dict:
        print("\n" + "=" * 60)
        print(f"💬 사용자 메시지: {user_message}")
        if selected_item_ids:
            print(f"👕 선택된 아이템: {selected_item_ids}")
        print("=" * 60)

        # 1) 히스토리 꺼내기
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        history = self.conversation_history[user_id]

        # 2) 옷장 요약
        wardrobe_info = self._get_wardrobe_summary(user_id)
        print(f"📋 옷장 정보: {wardrobe_info}")

        # 3) RAG 검색
        retrieved = self.rag.search(user_message, k=5)
        retrieved_texts = [r["text"] for r in retrieved]
        print(f"🔎 RAG 검색 상위: {retrieved_texts}")

        # 4) 시스템 프롬프트 만들기
        system_prompt = self._build_system_prompt(wardrobe_info, retrieved_texts)
        print(f"📝 시스템 프롬프트 (첫 200자): {system_prompt[:200]}...")

        # 5) 모델 입력 메시지
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_message},
        ]

        # 6) LLM 응답
        llm_response = self._generate_response(messages, wardrobe_info)

        # 7) 히스토리 업데이트
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": llm_response})
        if len(history) > 20:
            history[:] = history[-20:]
        self.conversation_history[user_id] = history

        # 8) 컨텍스트 추출
        context = self._extract_context(history)
        print(f"\n📋 추출된 컨텍스트: {json.dumps(context, ensure_ascii=False, indent=2)}")

        # 9) 추천 아이템 뽑기
        is_rec = self._is_recommendation_request(user_message)
        print(f"\n🔍 추천 요청 여부: {is_rec}")
        print(f"🔍 선택된 아이템 존재: {bool(selected_item_ids)}")

        recommendations = []
        show_all_wardrobe = False

        if self.db_conn is None:
            print("⚠️ DB 없음 → 추천/옷장 기능은 빈 리스트로 반환")
        else:
            if selected_item_ids and is_rec:
                print("\n🎯 선택된 아이템 기반 추천 시작...")
                recommendations = self._recommend_based_on_selected(user_id, selected_item_ids, context)
                if len(recommendations) == 0:
                    print("⚠️ 추천 결과 없음 → 대체 추천 호출")
                    recommendations = self._get_fallback_recommendations(user_id, selected_item_ids)
            else:
                show_all_wardrobe = self._is_show_wardrobe_request(user_message)
                if show_all_wardrobe:
                    recommendations = self._get_all_wardrobe_items(user_id)
                else:
                    if self._should_recommend(context):
                        recommendations = self._recommend_items(user_id, context)

        return {
            "response": llm_response,
            "context": context,
            "recommendations": recommendations,
            "need_more_info": (
                self.db_conn is not None
                and not (selected_item_ids and is_rec)
                and not show_all_wardrobe
                and not self._should_recommend(context)
            ),
        }

    # ----------------------------------
    # 헬퍼: 플래그 체크
    # ----------------------------------

    def _is_recommendation_request(self, user_message: str) -> bool:
        msg = user_message.lower().strip()
        keywords = [
            '추천', '추천해', '추천해줘', '스타일링', '패션', '코디',
            '어울리', '매칭', '입', '입을', '조합', '믹스매치', '골라', '찾아'
        ]
        return any(keyword in msg for keyword in keywords)

    def _is_show_wardrobe_request(self, user_message: str) -> bool:
        msg = user_message.lower().strip()
        keywords = [
            '옷장 보여', '옷장 보기', '옷장 전체', '내 옷 보여',
            '내 옷장', '가진 옷', '뭐 있', '몇 개'
        ]
        return any(keyword in msg for keyword in keywords)

    # ----------------------------------
    # DB 의존 함수들
    # ----------------------------------

    def _recommend_based_on_selected(self, user_id: int, selected_item_ids: list, context: dict) -> list:
        """
        사용자가 특정 아이템(예: 상의)을 골랐을 때
        그거랑 잘 맞는 하의/아우터 같은 조합 찾아주는 모드
        """
        if self.db_conn is None:
            return []

        try:
            with self.db_conn.cursor() as cur:
                placeholders = ','.join(['%s'] * len(selected_item_ids))
                query = f"""
                    SELECT 
                        w.item_id,
                        w.has_top, w.has_bottom, w.has_outer, w.has_dress,
                        t.category as top_cat, t.color as top_color,
                        b.category as bottom_cat, b.color as bottom_color,
                        o.category as outer_cat, o.color as outer_color,
                        d.category as dress_cat, d.color as dress_color
                    FROM wardrobe_items w
                    LEFT JOIN top_attributes_new    t ON w.item_id = t.item_id
                    LEFT JOIN bottom_attributes_new b ON w.item_id = b.item_id
                    LEFT JOIN outer_attributes_new  o ON w.item_id = o.item_id
                    LEFT JOIN dress_attributes_new  d ON w.item_id = d.item_id
                    WHERE w.item_id IN ({placeholders})
                """
                cur.execute(query, selected_item_ids)
                selected_items = cur.fetchall()

                if not selected_items:
                    return []

                has_top    = any(row[1] for row in selected_items)
                has_bottom = any(row[2] for row in selected_items)
                has_outer  = any(row[3] for row in selected_items)
                has_dress  = any(row[4] for row in selected_items)

                filters = []
                if has_dress:
                    filters.append("w.has_outer = TRUE")
                elif has_top and has_bottom:
                    filters.append("w.has_outer = TRUE")
                elif has_top:
                    filters.append("w.has_bottom = TRUE")
                elif has_bottom:
                    filters.append("w.has_top = TRUE")
                else:
                    filters.append("(w.has_top = TRUE OR w.has_dress = TRUE)")

                excluded_ids = ','.join(map(str, selected_item_ids))
                filters.append(f"w.item_id NOT IN ({excluded_ids})")
                where_clause = " AND ".join(filters)

                recommendation_query = f"""
                    SELECT w.item_id, w.user_id, w.is_default,
                           CASE WHEN w.user_id = %s THEN 0 ELSE 1 END AS user_priority
                    FROM wardrobe_items w
                    LEFT JOIN top_attributes_new    t ON w.item_id = t.item_id
                    LEFT JOIN bottom_attributes_new b ON w.item_id = b.item_id
                    LEFT JOIN outer_attributes_new  o ON w.item_id = o.item_id
                    LEFT JOIN dress_attributes_new  d ON w.item_id = d.item_id
                    WHERE (w.user_id = %s OR w.is_default = TRUE)
                      AND ({where_clause})
                    GROUP BY w.item_id, w.user_id, w.is_default
                    ORDER BY user_priority, w.item_id DESC
                    LIMIT 10
                """
                cur.execute(recommendation_query, (user_id, user_id))
                results = cur.fetchall()
                item_ids = [row[0] for row in results]

                if len(item_ids) == 0:
                    # 없으면 fallback
                    cur.execute("""
                        SELECT DISTINCT w.item_id
                        FROM wardrobe_items w
                        WHERE w.user_id = %s OR w.is_default = TRUE
                        ORDER BY 
                            CASE WHEN w.user_id = %s THEN 0 ELSE 1 END,
                            w.item_id DESC
                        LIMIT 5
                    """, (user_id, user_id))
                    results = cur.fetchall()
                    item_ids = [row[0] for row in results]

                return item_ids

        except Exception as e:
            print("❌ 선택 기반 추천 오류:", e)
            try:
                self.db_conn.rollback()
            except Exception:
                pass
            return []

    def _get_fallback_recommendations(self, user_id: int, exclude_ids: list = None) -> list:
        """
        그래도 아무 것도 못 뽑았을 때 그냥 몇 개라도 추천
        """
        if self.db_conn is None:
            return []

        try:
            with self.db_conn.cursor() as cur:
                if exclude_ids:
                    placeholders = ','.join(['%s'] * len(exclude_ids))
                    query = f"""
                        SELECT DISTINCT w.item_id
                        FROM wardrobe_items w
                        WHERE (w.user_id = %s OR w.is_default = TRUE)
                          AND w.item_id NOT IN ({placeholders})
                        ORDER BY 
                            CASE WHEN w.user_id = %s THEN 0 ELSE 1 END,
                            w.item_id DESC
                        LIMIT 8
                    """
                    params = [user_id] + list(exclude_ids) + [user_id]
                else:
                    query = """
                        SELECT DISTINCT w.item_id
                        FROM wardrobe_items w
                        WHERE w.user_id = %s OR w.is_default = TRUE
                        ORDER BY 
                            CASE WHEN w.user_id = %s THEN 0 ELSE 1 END,
                            w.item_id DESC
                        LIMIT 8
                    """
                    params = [user_id, user_id]

                cur.execute(query, params)
                results = cur.fetchall()
                return [row[0] for row in results]

        except Exception as e:
            print("❌ 대체 추천 오류:", e)
            try:
                self.db_conn.rollback()
            except Exception:
                pass
            return []

    def _get_all_wardrobe_items(self, user_id: int) -> list:
        """
        "내 옷장 보여줘" 같은 요청일 때 전체 리스트
        """
        if self.db_conn is None:
            return []

        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT item_id
                    FROM wardrobe_items
                    WHERE user_id = %s
                    ORDER BY item_id DESC
                """, (user_id,))
                results = cur.fetchall()
                return [row[0] for row in results]

        except Exception as e:
            print("❌ 옷장 아이템 조회 오류:", e)
            try:
                self.db_conn.rollback()
            except Exception:
                pass
            return []

    def _get_wardrobe_summary(self, user_id: int) -> str:
        """
        "옷장에 뭐 있어?" 같은 질문에 쓸 요약 문자열
        """
        print(f"\n👕 옷장 정보 조회 시작... user_id: {user_id}")

        if self.db_conn is None:
            return "옷장 정보 없음 (DB 연결 안 됨)"

        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(CASE WHEN has_top THEN 1 END) as tops,
                        COUNT(CASE WHEN has_bottom THEN 1 END) as bottoms,
                        COUNT(CASE WHEN has_outer THEN 1 END) as outers,
                        COUNT(CASE WHEN has_dress THEN 1 END) as dresses
                    FROM wardrobe_items
                    WHERE user_id = %s
                """, (user_id,))
                result = cur.fetchone()

                if result and result[0] > 0:
                    total, tops, bottoms, outers, dresses = result
                    return (
                        f"옷장: 총 {total}개 "
                        f"(상의 {tops}개, 하의 {bottoms}개, 아우터 {outers}개, 원피스 {dresses}개)"
                    )
                else:
                    return "옷장이 비어있음"

        except Exception as e:
            print("❌ 옷장 정보 조회 실패:", e)
            try:
                self.db_conn.rollback()
            except Exception:
                pass
            return "옷장 정보 없음"

    # ----------------------------------
    # 프롬프트 / LLM 응답
    # ----------------------------------

    def _build_system_prompt(self, wardrobe_info: str = "", retrieved_texts: Optional[List[str]] = None) -> str:
        wardrobe_context = f"\n\n[사용자 옷장] {wardrobe_info}" if wardrobe_info else ""
        rag_block = ""
        if retrieved_texts:
            joined = "\n- " + "\n- ".join(retrieved_texts[:5])
            rag_block = (
                f"\n\n[참고 지식]\n{joined}\n"
                "(위 내용은 참고만. 실제 사용자 상황이 우선임.)"
            )

        return f"""너는 친근한 한국 패션 스타일리스트 AI야.{wardrobe_context}{rag_block}

[목표]
사용자 상황에 맞는 옷을 골라주고, 필요한 질문을 짧게 한다.

[수집해야 할 정보]
1) 날씨/온도 (춥다, 덥다, 비 온다 등)
2) 어디 가는지 (출근, 데이트, 운동, 파티 등)
3) 원하는 무드/스타일 (편하게, 단정하게, 꾸안꾸 등)

[대화 규칙]
- 반말
- 한 번에 1~2문장
- 너무 길게 설명하지 말기
- 정보 2가지 이상 알면 "좋아, 추천 들어갈게!" 라고 말하고 질문 멈추기
- "내 옷장 뭐 있어?" 이런 말 오면 옷장 정보 말해주기

[예시]
사용자: 오늘 뭐 입지?
AI: 어디 가? 그리고 날씨는 어때? ☁

사용자: 회사 가는데 좀 추워
AI: 오케이. 추운 날 출근룩으로 따뜻하고 단정한 걸로 골라볼게 🔥

사용자: 데이트하는데 시원하고 예쁜 거
AI: 좋아. 데이트용으로 시원하고 깔끔한 거 골라볼게 💗

항상 한국어로만 답하고, 너무 딱딱하지 않게.
"""

    def _get_rule_based_response(self, user_message: str, history: List[Dict], wardrobe_info: str = "") -> Optional[str]:
        """
        빠른 반응용 규칙 답변 (LLM 호출 전에 한 번 시도)
        """
        msg = user_message.lower().strip()

        if any(word in msg for word in ['안녕', '하이', '헬로', '안뇽', 'hi', 'hello']):
            if len(history) == 0:
                return "안녕! 오늘 뭐 입을지 같이 보자 😌"
            return "응 들려~ 어떤 상황이야 지금?"

        if any(word in msg for word in ['옷장 보여', '옷장 보기', '옷장 전체', '내 옷 보여']):
            return "옷장 보여줄게. 마음에 드는 거 고르고 '추천해줘'라고 말해봐 ✨"

        if any(word in msg for word in ['몇 개', '몇개', '개수', '얼마나']):
            if wardrobe_info and '옷장:' in wardrobe_info:
                return f"{wardrobe_info.replace('옷장:', '').strip()} 있어. 그중에서 골라볼까?"
            return "몇 개 있는지 확인해볼게!"

        if any(word in msg for word in ['옷장', '뭐 있', '가진 옷', '내 옷', '내꺼']):
            if wardrobe_info and '옷장:' in wardrobe_info:
                return f"{wardrobe_info.replace('옷장:', '').strip()} 있어. 볼래?"
            return "지금 네 옷장 정보 불러올게."

        if any(word in msg for word in ['추천', '추천해', '뭐 입', '코디', '입을까', '매칭', '골라', '스타일링', '패션', '어울리', '조합', '찾아']):
            return "오케이. 잘 어울리는 조합 뽑아볼게 ✨"

        if any(word in msg for word in ['춥', '추워', '추운', '겨울', '쌀쌀']):
            if any(word in msg for word in ['회사', '출근', '일', '미팅']):
                return "추운 날 회사룩이면 따뜻하면서 단정한 걸로 가야지. 그렇게 골라볼게 🔥"
            elif any(word in msg for word in ['데이트', '만남', '약속']):
                return "추운데 데이트면 따뜻+예쁨 균형 가야지 😏 그쪽으로 볼게"
            return "좀 추운 날이구나. 어디 가는 길이야? (회사/데이트/운동 등)"

        if any(word in msg for word in ['더워', '덥', '더운', '여름', '뜨거']):
            if any(word in msg for word in ['회사', '출근', '일', '미팅']):
                return "덥고 회사 가는 날이면 시원+단정 조합으로 가야 돼. 그거로 볼게 😎"
            return "덥구나. 어디 가는 거야?"

        if any(word in msg for word in ['비', '우산', '장마', '소나기', '빗']):
            return "비 오는 날이면 너무 밝은 하의만 아니면 좋아. 어디 가?"

        if any(word in msg for word in ['운동', '헬스', '조깅', '요가', '필라테스']):
            return "운동복 찾는 거지? 편하고 땀 잘 마르는 쪽으로 볼게 💪"

        if any(word in msg for word in ['회사', '출근', '일', '미팅', '업무']):
            return "출근룩이구나. 날씨는 추워? 아니면 덥긴 해?"

        if any(word in msg for word in ['데이트', '소개팅', '만남', '약속']):
            return "오 데이트네 😏 날씨는 어때? 춥거나 덥거나?"

        if any(word in msg for word in ['도움', '사용법', '어떻게', '방법']):
            return "날씨랑 어디 가는지만 말해주면 내가 알아서 조합 뽑아줘. 예: '추운데 회사 가' 이런 식으로만 말해줘도 돼."

        if len(msg) <= 2:
            return "조금만 더 알려줄래? 예: '추운데 회사 가야 돼' 이런 느낌으로 ✨"

        return "오케이. 지금 어디 가는 상황이야? 그리고 날씨 어때? 😊"

    def _generate_response(self, messages: List[Dict], wardrobe_info: str = "") -> str:
        """
        1) 규칙 답변 먼저 시도
        2) 규칙으로 안 되면 실제 LLM 호출
        """
        if len(messages) > 1 and messages[-1]['role'] == 'user':
            user_msg = messages[-1]['content']
            history = [m for m in messages[1:-1]]
            rule_response = self._get_rule_based_response(user_msg, history, wardrobe_info)
            if rule_response:
                return rule_response

        # 규칙에서 답 못 만들면 프롬프트 만들어서 LLM 호출
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"{content}\n\n"
            elif role == "user":
                prompt += f"사용자: {content}\n"
            elif role == "assistant":
                prompt += f"AI: {content}\n"
        prompt += "AI: "

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.5,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
                if self.tokenizer.pad_token_id is None
                else self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # 한 줄만
        response = response.split('\n')[0].strip()
        # 반복된 문장 줄이기
        response = re.sub(r'(.+?)\1{2,}', r'\1', response)
        # 이상한 문자 제거
        response = re.sub(r'[^\w\s가-힣.,!?~\-]', '', response)

        # 너무 길면 자연스러운 문장부호까지 잘라주기
        if len(response) > 100:
            for delimiter in ['!', '?', '~', '.']:
                if delimiter in response[:100]:
                    parts = response[:100].split(delimiter)
                    if len(parts) > 1:
                        response = delimiter.join(parts[:-1]) + delimiter
                        break

        # 응답이 너무 빈약하면 안전 문구
        if not response or len(response) < 2:
            response = "지금 어디 가는지랑 날씨만 알려줘도 내가 골라줄 수 있어 😊"

        # 혹시 영어만 나오면 폴백
        korean_chars = sum(1 for c in response if '가' <= c <= '힣')
        if korean_chars < len(response) * 0.3:
            response = "상황(출근/데이트 등) + 날씨(추움/더움 등)만 말해주면 코디 뽑아줄게 ✨"

        return response

    # ----------------------------------
    # 컨텍스트 추출 (날씨/상황 등)
    # ----------------------------------
    def _extract_context(self, history: List[Dict]) -> Dict:
        context = {
            'weather': None,
            'temperature': None,
            'occasion': None,
            'health': None,
            'style': None,
            'color_pref': None,
            'formality': None,
        }

        full_text = ' '.join(
            [msg['content'] for msg in history if msg['role'] == 'user']
        ).lower()

        # 온도 / 날씨
        if any(word in full_text for word in ['엄청 춥', '너무 춥', '정말 춥', '겨울', '영하', '꽁꽁', '얼어']):
            context['weather'] = 'cold'
            context['temperature'] = 'very_cold'
        elif any(word in full_text for word in ['춥', '추워', '쌀쌀', '쌀쌀하', '시원', '차가']):
            context['weather'] = 'cold'
            context['temperature'] = 'cold'
        elif any(word in full_text for word in ['서늘', '선선', '가을', '봄']):
            context['temperature'] = 'cool'
        elif any(word in full_text for word in ['따뜻', '포근']):
            context['temperature'] = 'warm'
        elif any(word in full_text for word in ['더워', '덥', '뜨거', '무더', '여름', '폭염']):
            context['weather'] = 'hot'
            context['temperature'] = 'hot'

        if any(word in full_text for word in ['비', '우산', '장마', '소나기', '빗', '비옴']):
            context['weather'] = 'rainy'
        if any(word in full_text for word in ['눈', '함박눈', '폭설']):
            context['weather'] = 'snowy'

        # 목적/상황
        if any(word in full_text for word in ['회사', '출근', '업무', '미팅', '회의', '프레젠테이션', '발표', '직장']):
            context['occasion'] = 'work'
            context['formality'] = 'formal'
        if any(word in full_text for word in ['면접', '인터뷰', '취업', '입사']):
            context['occasion'] = 'interview'
            context['formality'] = 'very_formal'
        if any(word in full_text for word in ['데이트', '소개팅', '만남', '약속', '썸', '연애']):
            context['occasion'] = 'date'
            context['formality'] = 'semi_formal'
        if any(word in full_text for word in ['운동', '헬스', '조깅', '러닝', '요가', '필라테스', '체육관']):
            context['occasion'] = 'exercise'
            context['style'] = 'comfortable'
        if any(word in full_text for word in ['파티', '결혼식', '행사', '모임', '클럽', '술자리']):
            context['occasion'] = 'party'
            context['formality'] = 'formal'
        if any(word in full_text for word in ['집', '집에', '편하게', '쉬는', '휴식']):
            context['occasion'] = 'home'
            context['style'] = 'comfortable'
        if any(word in full_text for word in ['여행', '휴가', '놀러', '나들이', '외출']):
            context['occasion'] = 'travel'
            context['style'] = 'casual'

        # 건강 상태
        if any(word in full_text for word in ['감기', '아프', '목', '기침', '열', '콧물', '몸살', '독감']):
            context['health'] = 'sick'
        if any(word in full_text for word in ['목', '목이', '목아', '목감기']):
            context['health'] = 'sore_throat'

        # 스타일 취향
        if any(word in full_text for word in ['편안', '편하게', '캐주얼', '편한', '느슨', '루즈']):
            context['style'] = 'casual'
        if any(word in full_text for word in ['격식', '정장', '깔끔', '단정', '정돈']):
            context['style'] = 'formal'
        if any(word in full_text for word in ['멋', '세련', '스타일리시', '트렌디', '패셔너블']):
            context['style'] = 'stylish'

        # 색 취향
        if any(word in full_text for word in ['밝은', '화사', '파스텔', '연한', '하얀', '흰색']):
            context['color_pref'] = 'bright'
        if any(word in full_text for word in ['어두운', '다크', '검은', '블랙', '진한']):
            context['color_pref'] = 'dark'
        if any(word in full_text for word in ['베이지', '그레이', '아이보리', '무채색']):
            context['color_pref'] = 'neutral'

        return context

    def _should_recommend(self, context: Dict) -> bool:
        """
        정보가 어느 정도 확보되면 추천해도 된다고 판단
        """
        filled_contexts = sum(1 for v in context.values() if v is not None)
        return filled_contexts >= 2

    def _recommend_items(self, user_id: int, context: Dict) -> List[int]:
        """
        대화만으로도 (날씨+상황 등) 충분히 정보가 모였을 때,
        그 조건에 맞는 후보 아이템을 뽑는다.
        """
        if self.db_conn is None:
            return []

        filters = []

        # --- 온도/날씨 기반 필터 ---
        if context['temperature'] == 'very_cold':
            filters.append("""
                (t.category IN ('coat', 'puffer', 'padding', 'long coat')
                 OR t.materials::text ILIKE '%padding%'
                 OR t.materials::text ILIKE '%down%'
                 OR t.materials::text ILIKE '%wool%')
            """)
        elif context['temperature'] == 'cold' or context['weather'] == 'cold':
            filters.append("""
                (t.category IN ('coat', 'jacket', 'cardigan', 'jumper', 'hoodie', 'blazer')
                 OR t.materials::text ILIKE '%wool%'
                 OR t.materials::text ILIKE '%fleece%'
                 OR t.materials::text ILIKE '%knit%')
            """)
        elif context['temperature'] == 'cool':
            filters.append("""
                (t.category IN ('cardigan', 'shirt', 'blouse', 'light jacket')
                 OR t.fit = 'regular')
            """)
        elif context['temperature'] == 'hot' or context['weather'] == 'hot':
            filters.append("""
                (t.category IN ('t-shirt', 'tank', 'blouse', 'short sleeve')
                 OR t.materials::text ILIKE '%cotton%'
                 OR t.materials::text ILIKE '%linen%'
                 OR t.fit = 'loose')
            """)

        # 비/눈 등
        if context['weather'] == 'rainy':
            filters.append("""
                (t.color IN ('black', 'navy', 'gray', 'dark', 'charcoal')
                 OR b.color IN ('black', 'navy', 'gray', 'dark'))
            """)
        if context['weather'] == 'snowy':
            filters.append("""
                (b.category IN ('pants', 'jeans', 'long skirt')
                 AND t.category IN ('coat', 'puffer', 'jacket'))
            """)

        # 장소/격식
        if context['occasion'] == 'interview' or context['formality'] == 'very_formal':
            filters.append("""
                (t.category IN ('shirt', 'blouse', 'blazer', 'suit jacket')
                 AND t.color IN ('white', 'black', 'navy', 'gray', 'beige')
                 AND b.category IN ('pants', 'skirt', 'suit pants'))
            """)
        elif context['occasion'] == 'work' or context['formality'] == 'formal':
            filters.append("""
                (t.category IN ('shirt', 'blouse', 'blazer', 'cardigan')
                 AND t.color NOT IN ('neon', 'bright', 'hot pink')
                 AND (b.category IN ('pants', 'skirt') OR b.fit = 'regular'))
            """)
        elif context['occasion'] == 'date':
            filters.append("""
                (t.category IN ('shirt', 'blouse', 'knit', 'dress')
                 OR t.fit IN ('slim', 'regular'))
            """)
        elif context['occasion'] == 'party':
            filters.append("""
                (t.category IN ('dress', 'blouse', 'fancy top')
                 OR t.color IN ('red', 'gold', 'silver', 'bright', 'wine'))
            """)
        elif context['occasion'] == 'exercise':
            filters.append("""
                ((t.category IN ('t-shirt', 'hoodie', 'tank', 'sports wear')
                  AND t.fit IN ('loose', 'relaxed'))
                 OR t.materials::text ILIKE '%polyester%'
                 OR t.materials::text ILIKE '%spandex%')
            """)
        elif context['occasion'] == 'home':
            filters.append("""
                (t.fit IN ('loose', 'oversized', 'relaxed')
                 AND t.materials::text ILIKE '%cotton%')
            """)
        elif context['occasion'] == 'travel':
            filters.append("""
                ((t.category IN ('t-shirt', 'shirt', 'hoodie')
                  AND t.fit IN ('regular', 'loose'))
                 AND b.category IN ('jeans', 'pants', 'shorts'))
            """)

        # 건강상태
        if context['health'] == 'sick' or context['health'] == 'sore_throat':
            filters.append("""
                (t.fit IN ('loose', 'relaxed', 'oversized')
                 AND (t.materials::text ILIKE '%cotton%' 
                      OR t.materials::text ILIKE '%wool%'
                      OR t.category IN ('hoodie', 'knit', 'cardigan')))
            """)

        # 스타일 취향
        if context['style'] == 'casual':
            filters.append("(t.fit IN ('loose', 'relaxed') OR b.fit IN ('loose', 'relaxed'))")
        if context['style'] == 'formal':
            filters.append("(t.category IN ('shirt', 'blouse', 'blazer') AND t.fit IN ('slim', 'regular'))")
        if context['style'] == 'stylish':
            filters.append("(t.fit IN ('slim', 'regular') OR t.category IN ('blazer', 'dress', 'knit'))")

        # 색 취향
        if context['color_pref'] == 'bright':
            filters.append("(t.color IN ('white', 'beige', 'ivory', 'light', 'pastel'))")
        if context['color_pref'] == 'dark':
            filters.append("(t.color IN ('black', 'navy', 'gray', 'charcoal', 'dark') OR b.color IN ('black', 'navy'))")
        if context['color_pref'] == 'neutral':
            filters.append("(t.color IN ('beige', 'gray', 'ivory', 'brown', 'camel'))")

        where_clause = " AND ".join(filters) if filters else "TRUE"

        query = f"""
            SELECT w.item_id, w.user_id,
                   CASE WHEN w.user_id = %s THEN 0 ELSE 1 END AS user_priority
            FROM wardrobe_items w
            LEFT JOIN top_attributes_new    t ON w.item_id = t.item_id
            LEFT JOIN bottom_attributes_new b ON w.item_id = b.item_id
            LEFT JOIN outer_attributes_new  o ON w.item_id = o.item_id
            LEFT JOIN dress_attributes_new  d ON w.item_id = d.item_id
            WHERE (w.user_id = %s OR w.is_default = TRUE)
              AND ({where_clause})
            GROUP BY w.item_id, w.user_id
            ORDER BY user_priority, w.item_id DESC
            LIMIT 6
        """

        # 💥 FIXED: 예전엔 params가 [user_id, user_id, user_id] 이런 식으로 3개 들어가서 에러났음
        params = [user_id, user_id]

        try:
            with self.db_conn.cursor() as cur:
                cur.execute(query, params)
                results = cur.fetchall()
                item_ids = [row[0] for row in results]

                # 후보가 0개면 fallback으로 몇 개라도 보내주기
                if len(item_ids) == 0:
                    cur.execute("""
                        SELECT DISTINCT w.item_id
                        FROM wardrobe_items w
                        WHERE w.user_id = %s OR w.is_default = TRUE
                        ORDER BY 
                            CASE WHEN w.user_id = %s THEN 0 ELSE 1 END,
                            w.item_id DESC
                        LIMIT 5
                    """, (user_id, user_id))
                    results = cur.fetchall()
                    item_ids = [row[0] for row in results]

                return item_ids

        except Exception as e:
            print("❌ 추천 생성 오류:", e)
            try:
                self.db_conn.rollback()
            except Exception:
                pass
            return []

    # ----------------------------------
    # 기타 유틸
    # ----------------------------------

    def reset_conversation(self, user_id: int):
        if user_id in self.conversation_history:
            del self.conversation_history[user_id]
            print(f"✅ 사용자 {user_id}의 대화 히스토리 초기화 완료")

    def close(self):
        if self.db_conn:
            self.db_conn.close()
            print("PostgreSQL 연결 종료")

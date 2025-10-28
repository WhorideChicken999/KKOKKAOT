# _llm_recommender.py
# 
# 🆕 새로운 파일 생성
# LLM 기반 대화형 옷 추천 시스템
#
# 주요 기능:
# 1. Llama 모델을 사용한 자연어 대화
# 2. 대화에서 컨텍스트 추출 (날씨, 상황, 건강 상태 등)
# 3. 추출된 정보를 바탕으로 옷장 아이템 필터링
# 4. 사용자 맞춤형 추천 생성

import os
import json
import re
from typing import Dict, List, Optional
import psycopg2
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMRecommender:
    """LLM 기반 대화형 옷 추천 시스템"""
    
    def __init__(self, db_config: dict, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):        
        """
        초기화
        
        Args:
            db_config: PostgreSQL 연결 설정
            model_name: 사용할 한국어 특화 모델 이름
        """
        print("\n🤖 LLM 추천 시스템 초기화 중...")
        
        # DB 연결
        self.db_conn = psycopg2.connect(**db_config)
        print("✅ PostgreSQL 연결 완료")
        
        # Llama 모델 로드
        print(f"📥 모델 로딩 중: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        print("✅ LLM 모델 로드 완료\n")
        
        # 대화 히스토리 저장
        self.conversation_history = {}
    
    def chat(self, user_id: int, user_message: str, selected_item_ids: list = None) -> Dict:
        """
        사용자와 대화하고 추천 생성
        
        Args:
            user_id: 사용자 ID
            user_message: 사용자 메시지
            selected_item_ids: 선택된 아이템 ID 리스트 (옵션)
            
        Returns:
            dict: {
                'response': str (LLM 응답),
                'context': dict (추출된 컨텍스트),
                'recommendations': list (추천 아이템 ID 리스트),
                'need_more_info': bool (추가 정보 필요 여부)
            }
        """
        print(f"\n{'='*60}")
        print(f"💬 사용자 메시지: {user_message}")
        if selected_item_ids:
            print(f"👕 선택된 아이템: {selected_item_ids}")
        print(f"{'='*60}")
        
        # 1. 대화 히스토리 가져오기
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        history = self.conversation_history[user_id]
        
        # 2. 옷장 정보 가져오기 (새로 추가)
        wardrobe_info = self._get_wardrobe_summary(user_id)
        print(f"📋 옷장 정보: {wardrobe_info}")
        
        # 3. 시스템 프롬프트 구성
        system_prompt = self._build_system_prompt(wardrobe_info)
        print(f"📝 시스템 프롬프트 (첫 200자): {system_prompt[:200]}...")
        
        # 3. 대화 히스토리를 포함한 프롬프트 생성
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": user_message}
        ]
        
        # 4. LLM 응답 생성
        llm_response = self._generate_response(messages, wardrobe_info)
        
        # 5. 히스토리에 추가
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": llm_response})
        
        # 최근 10개 대화만 유지
        if len(history) > 20:
            history = history[-20:]
        self.conversation_history[user_id] = history
        
        # 6. 컨텍스트 추출
        context = self._extract_context(history)
        print(f"\n📋 추출된 컨텍스트: {json.dumps(context, ensure_ascii=False, indent=2)}")
        
        # 7. 선택된 아이템 기반 추천 확인
        is_recommendation_request = self._is_recommendation_request(user_message)
        print(f"\n🔍 추천 요청 여부: {is_recommendation_request}")
        print(f"🔍 선택된 아이템 존재: {bool(selected_item_ids)}")
        
        recommendations = []
        
        if selected_item_ids and is_recommendation_request:
            # 선택된 아이템 기반 추천
            print(f"\n🎯 선택된 아이템 기반 추천 시작...")
            recommendations = self._recommend_based_on_selected(user_id, selected_item_ids, context)
            print(f"\n✨ 추천 아이템: {len(recommendations)}개")
            
            if len(recommendations) == 0:
                print(f"⚠️ 추천 결과가 없어서 대체 추천 실행...")
                # 대체 추천: 선택된 아이템과 무관하게 추천
                recommendations = self._get_fallback_recommendations(user_id, selected_item_ids)
                print(f"✅ 대체 추천 아이템: {len(recommendations)}개")
        else:
            # 8. 옷장 전체 보기 요청 확인
            show_all_wardrobe = self._is_show_wardrobe_request(user_message)
            
            if show_all_wardrobe:
                # 옷장 전체 보기
                recommendations = self._get_all_wardrobe_items(user_id)
                print(f"\n👕 옷장 전체 보기: {len(recommendations)}개 아이템")
            else:
                # 9. 추천 필요 여부 판단
                need_recommendations = self._should_recommend(context)
                
                if need_recommendations:
                    # 10. 옷장 아이템 필터링 및 추천
                    recommendations = self._recommend_items(user_id, context)
                    print(f"\n✨ 추천 아이템: {len(recommendations)}개")
        
        return {
            'response': llm_response,
            'context': context,
            'recommendations': recommendations,
            'need_more_info': not (selected_item_ids and is_recommendation_request) and not show_all_wardrobe and not self._should_recommend(context)
        }
    
    def _is_recommendation_request(self, user_message: str) -> bool:
        """추천 요청인지 확인"""
        msg = user_message.lower().strip()
        keywords = ['추천', '추천해', '추천해줘', '스타일링', '패션', '코디', '어울리', '매칭', '입', '입을', '조합', '믹스매치', '골라', '찾아']
        result = any(keyword in msg for keyword in keywords)
        print(f"🔍 추천 요청 감지: {result} (메시지: '{msg}')")
        return result
    
    def _is_show_wardrobe_request(self, user_message: str) -> bool:
        """옷장 전체 보기 요청인지 확인"""
        msg = user_message.lower().strip()
        keywords = ['옷장 보여', '옷장 보기', '옷장 전체', '내 옷 보여', '내 옷장', '가진 옷', '뭐 있', '몇 개']
        return any(keyword in msg for keyword in keywords)
    
    def _recommend_based_on_selected(self, user_id: int, selected_item_ids: list, context: dict) -> list:
        """선택된 아이템 기반 추천"""
        try:
            with self.db_conn.cursor() as cur:
                # 선택된 아이템의 속성 가져오기
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
                    LEFT JOIN top_attributes_new t ON w.item_id = t.item_id
                    LEFT JOIN bottom_attributes_new b ON w.item_id = b.item_id
                    LEFT JOIN outer_attributes_new o ON w.item_id = o.item_id
                    LEFT JOIN dress_attributes_new d ON w.item_id = d.item_id
                    WHERE w.item_id IN ({placeholders})
                """
                
                cur.execute(query, selected_item_ids)
                selected_items = cur.fetchall()
                
                if not selected_items:
                    print("⚠️ 선택된 아이템 정보를 찾을 수 없음")
                    return []
                
                # 선택된 아이템 분석
                has_top = any(row[1] for row in selected_items)
                has_bottom = any(row[2] for row in selected_items)
                has_outer = any(row[3] for row in selected_items)
                has_dress = any(row[4] for row in selected_items)
                
                print(f"📊 선택된 아이템 분석: 상의={has_top}, 하의={has_bottom}, 아우터={has_outer}, 드레스={has_dress}")
                
                # 어울리는 아이템 추천 (부족한 카테고리 추천)
                filters = []
                
                if has_dress:
                    # 드레스가 있으면 아우터 추천
                    filters.append("w.has_outer = TRUE")
                elif has_top and has_bottom:
                    # 상의 + 하의가 있으면 아우터 추천
                    filters.append("w.has_outer = TRUE")
                elif has_top:
                    # 상의만 있으면 하의 추천
                    filters.append("w.has_bottom = TRUE")
                elif has_bottom:
                    # 하의만 있으면 상의 추천
                    filters.append("w.has_top = TRUE")
                else:
                    # 아우터만 있으면 상의+하의 또는 드레스 추천
                    filters.append("(w.has_top = TRUE OR w.has_dress = TRUE)")
                
                # 선택된 아이템 제외
                excluded_ids = ','.join(map(str, selected_item_ids))
                filters.append(f"w.item_id NOT IN ({excluded_ids})")
                
                where_clause = " AND ".join(filters)
                
                # 👇 테이블 이름 수정: top_attributes → top_attributes_new
                recommendation_query = f"""
                    SELECT w.item_id, w.user_id, w.is_default,
                           CASE WHEN w.user_id = %s THEN 0 ELSE 1 END as user_priority
                    FROM wardrobe_items w
                    LEFT JOIN top_attributes_new t ON w.item_id = t.item_id
                    LEFT JOIN bottom_attributes_new b ON w.item_id = b.item_id
                    LEFT JOIN outer_attributes_new o ON w.item_id = o.item_id
                    LEFT JOIN dress_attributes_new d ON w.item_id = d.item_id
                    WHERE (w.user_id = %s OR w.is_default = TRUE)
                      AND ({where_clause})
                    GROUP BY w.item_id, w.user_id, w.is_default
                    ORDER BY user_priority, w.item_id DESC
                    LIMIT 10
                """
                
                cur.execute(recommendation_query, (user_id, user_id))
                results = cur.fetchall()
                
                item_ids = [row[0] for row in results]
                print(f"✅ 추천 아이템 ID: {item_ids}")
                print(f"  📊 사용자 아이템: {sum(1 for row in results if not row[2])}")
                print(f"  📦 기본 아이템: {sum(1 for row in results if row[2])}")
                
                if len(item_ids) == 0:
                    print(f"⚠️ 조건에 맞는 아이템이 없어서 기본 추천 실행...")
                    # 👇 필터 없이 모든 아이템에서 추천
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
                    print(f"✅ 기본 추천 아이템 ID: {item_ids}")
                
                return item_ids
        
        except Exception as e:
            print(f"❌ 선택 기반 추천 오류: {e}")
            import traceback
            traceback.print_exc()
            
            # 트랜잭션 롤백
            try:
                self.db_conn.rollback()
                print("🔄 트랜잭션 롤백 완료")
            except Exception as rollback_error:
                print(f"⚠️ 롤백 실패: {rollback_error}")
            
            return []
    
    def _get_fallback_recommendations(self, user_id: int, exclude_ids: list = None) -> list:
        """대체 추천: 조건 없이 모든 아이템에서 추천 (기본 아이템 포함)"""
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
                item_ids = [row[0] for row in results]
                
                print(f"  ✅ 대체 추천 아이템 ID: {item_ids}")
                return item_ids
        
        except Exception as e:
            print(f"  ❌ 대체 추천 오류: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                self.db_conn.rollback()
                print("🔄 트랜잭션 롤백 완료")
            except Exception as rollback_error:
                print(f"⚠️ 롤백 실패: {rollback_error}")
            
            return []
    
    def _get_all_wardrobe_items(self, user_id: int) -> list:
        """사용자의 전체 옷장 아이템 가져오기"""
        try:
            with self.db_conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT item_id
                    FROM wardrobe_items
                    WHERE user_id = %s
                    ORDER BY item_id DESC
                """, (user_id,))
                
                results = cur.fetchall()
                item_ids = [row[0] for row in results]
                
                print(f"  ✅ 옷장 전체 아이템 ID: {item_ids}")
                return item_ids
        
        except Exception as e:
            print(f"  ❌ 옷장 아이템 조회 오류: {e}")
            
            # 트랜잭션 롤백
            try:
                self.db_conn.rollback()
                print("🔄 트랜잭션 롤백 완료")
            except Exception as rollback_error:
                print(f"⚠️ 롤백 실패: {rollback_error}")
            
            return []
    
    def _get_wardrobe_summary(self, user_id: int) -> str:
        """사용자 옷장 요약 정보 가져오기"""
        print(f"\n👕 옷장 정보 조회 시작... user_id: {user_id}")
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
                print(f"📦 쿼리 결과: {result}")
                
                if result and result[0] > 0:
                    total, tops, bottoms, outers, dresses = result
                    summary = f"옷장: 총 {total}개 (상의 {tops}개, 하의 {bottoms}개, 아우터 {outers}개, 원피스 {dresses}개)"
                    print(f"✅ 옷장 요약: {summary}")
                    return summary
                else:
                    print(f"⚠️ 옷장이 비어있음")
                    return "옷장이 비어있음"
        except Exception as e:
            print(f"❌ 옷장 정보 조회 실패: {e}")
            import traceback
            traceback.print_exc()
            
            # 트랜잭션 롤백
            try:
                self.db_conn.rollback()
                print("🔄 트랜잭션 롤백 완료")
            except Exception as rollback_error:
                print(f"⚠️ 롤백 실패: {rollback_error}")
            
            return "옷장 정보 없음"
    
    # ✅ _build_system_prompt 함수 전체 교체
    def _build_system_prompt(self, wardrobe_info: str = "") -> str:
        """시스템 프롬프트 구성"""
        wardrobe_context = f"\n\n**사용자 {wardrobe_info}**" if wardrobe_info else ""
        
        return f"""당신은 친근한 한국 패션 스타일리스트 AI입니다.{wardrobe_context}

**역할:**
오늘 사용자에게 필요한 옷을 파악하기 위해 자연스럽게 대화하세요.

**수집할 정보:**
1. 날씨/온도 (춥다, 덥다, 비온다 등)
2. 상황/장소 (출근, 데이트, 운동, 파티 등)
3. 스타일 선호 (캐주얼, 격식, 편안함 등)

**대화 규칙:**
- 응답은 1-2문장으로 짧게 유지
- 한 번에 1-2개 질문만
- 친근하게 반말로 대화
- 이모지 적절히 사용
- 2가지 이상 정보를 얻으면 "좋아! 추천 시작할게!" 하고 질문 중단
- 사용자가 "내 옷장"이나 "뭐 있어?"라고 물으면 옷장 정보를 알려줌

**대화 예시:**
사용자: "오늘 뭐 입을까?"
AI: "오늘 어디 가? 날씨는 어때? 😊"

사용자: "회사 가는데 좀 추워"
AI: "알겠어! 추운 날 출근 스타일 찾아줄게! 깔끔하고 따뜻한 옷 추천할게 ✨"

사용자: "데이트하는데 시원한 옷"
AI: "좋아! 데이트에 시원하면서 멋진 스타일 찾아줄게! 💕"

**중요:**
- 자연스럽고 짧게
- 긴 문장 금지
- 사용자 상황에 집중
- 한국어로만 대화"""
    
    def _get_rule_based_response(self, user_message: str, history: List[Dict], wardrobe_info: str = "") -> Optional[str]:
        """규칙 기반 응답 (우선 사용)"""
        msg = user_message.lower().strip()
        
        # 인사
        if any(word in msg for word in ['안녕', '하이', '헬로', '안뇽', 'hi', 'hello']):
            if len(history) == 0:
                return "안녕하세요! 오늘 뭐 입을지 고민이신가요? 😊"
            return "네! 무엇을 도와드릴까요?"
        
        # 옷장 관련 (전체 보기)
        if any(word in msg for word in ['옷장 보여', '옷장 보기', '옷장 전체', '내 옷 보여']):
            if wardrobe_info and '옷장:' in wardrobe_info:
                return f"옷장을 보여드릴게요! {wardrobe_info.replace('옷장:', '')} 있네요 👗✨\n\n💡 옷을 클릭하면 선택할 수 있어요! 선택 후 '추천해줘'라고 말씀해주세요."
            return "옷장을 보여드릴게요! 👗✨\n\n💡 옷을 클릭하면 선택할 수 있어요!"
        
        # 옷장 관련 (개수 질문)
        if any(word in msg for word in ['몇 개', '몇개', '개수', '얼마나']):
            if wardrobe_info and '옷장:' in wardrobe_info:
                return f"{wardrobe_info.replace('옷장:', '')} 있어요! 옷장을 보여드릴까요? 😊"
            return "옷장을 확인해볼게요!"
        
        # 옷장 관련 (일반)
        if any(word in msg for word in ['옷장', '뭐 있', '가진 옷', '내 옷', '내꺼']):
            if wardrobe_info and '옷장:' in wardrobe_info:
                return f"{wardrobe_info.replace('옷장:', '')} 있어요! 보여드릴까요?"
            return "옷장을 확인해볼게요! 어떤 스타일을 찾고 계세요?"
        
        # 추천 요청
        if any(word in msg for word in ['추천', '추천해', '뭐 입', '코디', '입을까', '매칭', '골라', '스타일링', '패션', '어울리', '조합', '찾아']):
            return "좋아요! 어울리는 옷을 찾아드릴게요! ✨"
        
        # 날씨 - 춥다
        if any(word in msg for word in ['춥', '추워', '추운', '겨울', '쌀쌀']):
            if any(word in msg for word in ['회사', '출근', '일', '미팅']):
                return "추운 날 출근 스타일이네요! 따뜻하고 단정한 옷 찾아드릴게요 ✨"
            elif any(word in msg for word in ['데이트', '만남', '약속']):
                return "추운 날 데이트 룩이네요! 따뜻하면서 멋진 스타일 찾아드릴게요 💕"
            return "추운 날씨네요! 어디 가시나요? (출근/데이트/운동 등)"
        
        # 날씨 - 덥다
        if any(word in msg for word in ['더워', '덥', '더운', '여름', '뜨거']):
            if any(word in msg for word in ['회사', '출근', '일', '미팅']):
                return "더운 날 출근 스타일이네요! 시원하고 단정한 옷 찾아드릴게요 ✨"
            return "더운 날씨네요! 어디 가시나요?"
        
        # 날씨 - 따뜻하다
        if any(word in msg for word in ['따뜻', '포근', '선선']):
            return "따뜻한 옷 찾으시는군요! 어디 가시나요? (출근/데이트/운동 등)"
        
        # 상황 - 출근
        if any(word in msg for word in ['회사', '출근', '일', '미팅', '업무']):
            return "출근 룩이시군요! 날씨는 어떠세요? (춥다/덥다/적당)"
        
        # 상황 - 데이트
        if any(word in msg for word in ['데이트', '소개팅', '만남', '약속']):
            return "데이트 룩이시군요! 날씨는 어떠세요?"
        
        # 상황 - 운동
        if any(word in msg for word in ['운동', '헬스', '조깅', '요가', '필라테스']):
            return "운동 복장이시군요! 편하고 활동적인 옷 찾아드릴게요! 💪"
        
        # 비
        if any(word in msg for word in ['비', '우산', '장마', '빗물']):
            return "비 오는 날이네요! 어두운 색 옷이 좋을 것 같아요. 어디 가세요?"
        
        # 불만/재시도
        if any(word in msg for word in ['뭐라', '이상', '다시', '못', '제대로', '영어', '이해']):
            return "죄송해요! 다시 말씀해주시겠어요? 간단하게 말씀해주시면 도와드릴게요! 😊"
        
        # 감사
        if any(word in msg for word in ['고마', '감사', '땡큐', 'thanks']):
            return "천만에요! 또 필요하시면 언제든지 말씀해주세요! 😊"
        
        # 도움말
        if any(word in msg for word in ['도움', '사용법', '어떻게', '방법']):
            return "날씨와 상황을 알려주시면 옷을 추천해드려요! 예: '추운데 회사 가', '더운데 데이트' 😊"
        
        # 짧은 응답들 (2글자 이하)
        if len(msg) <= 2:
            return "좀 더 자세히 말씀해주시겠어요? 😊"
        
        # 기본 폴백 (패턴 매칭 안됨)
        return "무엇을 도와드릴까요? 날씨와 상황을 알려주시면 옷을 추천해드려요! 😊"
    
    def _generate_response(self, messages: List[Dict], wardrobe_info: str = "") -> str:
        """LLM 응답 생성"""
        
        # 규칙 기반 응답 먼저 시도
        if len(messages) > 1 and messages[-1]['role'] == 'user':
            user_msg = messages[-1]['content']
            history = [m for m in messages[1:-1]]  # 시스템 프롬프트 제외
            rule_response = self._get_rule_based_response(user_msg, history, wardrobe_info)
            if rule_response:
                print(f"\n🎯 규칙 기반 응답 사용: {rule_response}")
                return rule_response
        
        # LLM 응답 생성
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
        
        print(f"\n📝 프롬프트 (마지막 150자): ...{prompt[-150:]}")
        
        # 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,  # 짧게
                temperature=0.8,     # 자연스러운 응답
                top_p=0.95,
                top_k=50,           # 다양성 제어
                repetition_penalty=1.5,  # 반복 강력 방지
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 디코딩
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # 👇 응답 정리
        response = response.strip()
        
        # 첫 번째 줄만 사용
        lines = response.split('\n')
        response = lines[0].strip()
        
        # 반복되는 단어/문자 제거 (중국어 반복 같은 것)
        # 같은 단어가 3번 이상 반복되면 1번만 남김
        response = re.sub(r'(.+?)\1{2,}', r'\1', response)
        
        # 이상한 문자 제거 (한글, 영문, 숫자, 기본 문장부호, 이모지만 허용)
        response = re.sub(r'[^\w\s가-힣.,!?~😀-🙏💕✨🎉👍\-]', '', response)
        
        # 너무 길면 자르기
        if len(response) > 100:
            for delimiter in ['!', '?', '~', '.']:
                if delimiter in response[:100]:
                    parts = response[:100].split(delimiter)
                    if len(parts) > 1:
                        response = delimiter.join(parts[:-1]) + delimiter
                        break
        
        # 응답 검증
        if not response or len(response) < 2:
            response = "죄송해요, 다시 말씀해주시겠어요?"
        
        # 한글이 거의 없으면 (50% 미만) 기본 응답 사용
        korean_chars = len([c for c in response if '가' <= c <= '힣'])
        if korean_chars < len(response) * 0.3:
            print(f"⚠️ 한글 비율 낮음 ({korean_chars}/{len(response)}), 폴백 응답 사용")
            response = "네! 어떤 옷을 찾으시나요? 상황이나 날씨를 알려주세요! 😊"
        
        print(f"\n🤖 LLM 응답: {response}")
        return response
    
    def _extract_context(self, history: List[Dict]) -> Dict:
        """대화에서 컨텍스트 추출 - 개선된 버전"""
        
        context = {
            'weather': None,      # 'cold', 'hot', 'rainy', 'normal'
            'temperature': None,  # 'very_cold', 'cold', 'cool', 'warm', 'hot'
            'occasion': None,     # 'work', 'date', 'casual', 'party', 'exercise'
            'health': None,       # 'sick', 'healthy'
            'style': None,        # 'formal', 'casual', 'comfortable', 'stylish'
            'color_pref': None,   # 'bright', 'dark', 'neutral'
            'formality': None,    # 'very_formal', 'formal', 'semi_formal', 'casual'
        }
        
        # 대화 전체 텍스트 결합
        full_text = ' '.join([msg['content'] for msg in history if msg['role'] == 'user']).lower()
        
        # 👇 날씨/온도 키워드 (더 세분화)
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
        
        # 👇 상황 키워드 (더 다양하게)
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
        
        if any(word in full_text for word in ['파티', '결혼식', '행사', '모임', '파티', '클럽', '술자리']):
            context['occasion'] = 'party'
            context['formality'] = 'formal'
        
        if any(word in full_text for word in ['집', '집에', '편하게', '쉬는', '휴식']):
            context['occasion'] = 'home'
            context['style'] = 'comfortable'
        
        if any(word in full_text for word in ['여행', '휴가', '놀러', '나들이', '외출']):
            context['occasion'] = 'travel'
            context['style'] = 'casual'
        
        # 👇 건강 상태 (더 구체적으로)
        if any(word in full_text for word in ['감기', '아프', '목', '기침', '열', '콧물', '몸살', '독감']):
            context['health'] = 'sick'
        
        if any(word in full_text for word in ['목', '목이', '목아', '목감기']):
            context['health'] = 'sore_throat'
        
        # 👇 스타일 선호
        if any(word in full_text for word in ['편안', '편하게', '캐주얼', '편한', '느슨', '루즈']):
            context['style'] = 'casual'
        
        if any(word in full_text for word in ['격식', '정장', '깔끔', '단정', '정돈']):
            context['style'] = 'formal'
        
        if any(word in full_text for word in ['멋', '세련', '스타일리시', '트렌디', '패셔너블']):
            context['style'] = 'stylish'
        
        # 👇 색상 선호
        if any(word in full_text for word in ['밝은', '화사', '파스텔', '연한', '하얀', '흰색']):
            context['color_pref'] = 'bright'
        
        if any(word in full_text for word in ['어두운', '다크', '검은', '블랙', '진한']):
            context['color_pref'] = 'dark'
        
        if any(word in full_text for word in ['베이지', '그레이', '아이보리', '무채색']):
            context['color_pref'] = 'neutral'
        
        return context


    # ✅ _should_recommend 함수 개선 (line ~190 근처)
    def _should_recommend(self, context: Dict) -> bool:
        """추천 시작 여부 판단 - 개선된 버전"""
        
        # 최소 2개 이상의 컨텍스트 정보가 있으면 추천 가능
        filled_contexts = sum(1 for v in context.values() if v is not None)
        
        print(f"\n🔍 컨텍스트 개수: {filled_contexts}개")
        
        # 날씨 + 상황 또는 상황 + 스타일 등 2개 이상이면 추천
        return filled_contexts >= 2


    # ✅ _recommend_items 함수 개선 (line ~200 근처)
    def _recommend_items(self, user_id: int, context: Dict) -> List[int]:
        """컨텍스트 기반 아이템 추천 - 개선된 버전"""
        
        print(f"\n🔍 컨텍스트 기반 추천 시작...")
        print(f"📋 적용된 컨텍스트: {json.dumps(context, ensure_ascii=False, indent=2)}")
        
        filters = []
        params = [user_id]
        
        # 1. 온도/날씨에 따른 필터 (더 세밀하게)
        if context['temperature'] == 'very_cold':
            # 매우 추움 → 패딩, 두꺼운 코트, 목도리 필요
            filters.append("""
                (t.category IN ('coat', 'puffer', 'padding', 'long coat')
                 OR t.materials::text ILIKE '%padding%'
                 OR t.materials::text ILIKE '%down%'
                 OR t.materials::text ILIKE '%wool%')
            """)
        
        elif context['temperature'] == 'cold' or context['weather'] == 'cold':
            # 추움 → 아우터, 긴팔, 따뜻한 소재
            filters.append("""
                (t.category IN ('coat', 'jacket', 'cardigan', 'jumper', 'hoodie', 'blazer')
                 OR t.materials::text ILIKE '%wool%'
                 OR t.materials::text ILIKE '%fleece%'
                 OR t.materials::text ILIKE '%knit%')
            """)
        
        elif context['temperature'] == 'cool':
            # 선선함 → 가디건, 얇은 재킷
            filters.append("""
                (t.category IN ('cardigan', 'shirt', 'blouse', 'light jacket')
                 OR t.fit = 'regular')
            """)
        
        elif context['temperature'] == 'hot' or context['weather'] == 'hot':
            # 더움 → 반팔, 얇은 소재
            filters.append("""
                (t.category IN ('t-shirt', 'tank', 'blouse', 'short sleeve')
                 OR t.materials::text ILIKE '%cotton%'
                 OR t.materials::text ILIKE '%linen%'
                 OR t.fit = 'loose')
            """)
        
        if context['weather'] == 'rainy':
            # 비 → 어두운 색상, 방수 소재
            filters.append("""
                (t.color IN ('black', 'navy', 'gray', 'dark', 'charcoal')
                 OR b.color IN ('black', 'navy', 'gray', 'dark'))
            """)
        
        if context['weather'] == 'snowy':
            # 눈 → 부츠 어울리는 스타일
            filters.append("""
                (b.category IN ('pants', 'jeans', 'long skirt')
                 AND t.category IN ('coat', 'puffer', 'jacket'))
            """)
        
        # 2. 상황/격식에 따른 필터
        if context['occasion'] == 'interview' or context['formality'] == 'very_formal':
            # 면접/매우 격식 → 정장 스타일
            filters.append("""
                (t.category IN ('shirt', 'blouse', 'blazer', 'suit jacket')
                 AND t.color IN ('white', 'black', 'navy', 'gray', 'beige')
                 AND b.category IN ('pants', 'skirt', 'suit pants'))
            """)
        
        elif context['occasion'] == 'work' or context['formality'] == 'formal':
            # 출근/격식 → 단정한 스타일
            filters.append("""
                (t.category IN ('shirt', 'blouse', 'blazer', 'cardigan')
                 AND t.color NOT IN ('neon', 'bright', 'hot pink')
                 AND (b.category IN ('pants', 'skirt') OR b.fit = 'regular'))
            """)
        
        elif context['occasion'] == 'date':
            # 데이트 → 세련되고 깔끔한 스타일
            filters.append("""
                (t.category IN ('shirt', 'blouse', 'knit', 'dress')
                 OR t.fit IN ('slim', 'regular'))
            """)
        
        elif context['occasion'] == 'party':
            # 파티 → 화려한 색상, 드레스
            filters.append("""
                (t.category IN ('dress', 'blouse', 'fancy top')
                 OR t.color IN ('red', 'gold', 'silver', 'bright', 'wine'))
            """)
        
        elif context['occasion'] == 'exercise':
            # 운동 → 편안하고 신축성 있는 소재
            filters.append("""
                ((t.category IN ('t-shirt', 'hoodie', 'tank', 'sports wear')
                  AND t.fit IN ('loose', 'relaxed'))
                 OR t.materials::text ILIKE '%polyester%'
                 OR t.materials::text ILIKE '%spandex%')
            """)
        
        elif context['occasion'] == 'home':
            # 집 → 가장 편안한 옷
            filters.append("""
                (t.fit IN ('loose', 'oversized', 'relaxed')
                 AND t.materials::text ILIKE '%cotton%')
            """)
        
        elif context['occasion'] == 'travel':
            # 여행 → 활동적이고 편안한 스타일
            filters.append("""
                ((t.category IN ('t-shirt', 'shirt', 'hoodie')
                  AND t.fit IN ('regular', 'loose'))
                 AND b.category IN ('jeans', 'pants', 'shorts'))
            """)
        
        # 3. 건강 상태
        if context['health'] == 'sick' or context['health'] == 'sore_throat':
            # 아픔 → 따뜻하고 편안한 옷
            filters.append("""
                (t.fit IN ('loose', 'relaxed', 'oversized')
                 AND (t.materials::text ILIKE '%cotton%' 
                      OR t.materials::text ILIKE '%wool%'
                      OR t.category IN ('hoodie', 'knit', 'cardigan')))
            """)
        
        # 4. 스타일 선호
        if context['style'] == 'casual':
            filters.append("(t.fit IN ('loose', 'relaxed') OR b.fit IN ('loose', 'relaxed'))")
        
        if context['style'] == 'formal':
            filters.append("(t.category IN ('shirt', 'blouse', 'blazer') AND t.fit IN ('slim', 'regular'))")
        
        if context['style'] == 'stylish':
            filters.append("(t.fit IN ('slim', 'regular') OR t.category IN ('blazer', 'dress', 'knit'))")
        
        # 5. 색상 선호
        if context['color_pref'] == 'bright':
            filters.append("(t.color IN ('white', 'beige', 'ivory', 'light', 'pastel'))")
        
        if context['color_pref'] == 'dark':
            filters.append("(t.color IN ('black', 'navy', 'gray', 'charcoal', 'dark') OR b.color IN ('black', 'navy'))")
        
        if context['color_pref'] == 'neutral':
            filters.append("(t.color IN ('beige', 'gray', 'ivory', 'brown', 'camel'))")
        
        # SQL 쿼리 구성
        where_clause = " AND ".join(filters) if filters else "TRUE"
        
        query = f"""
            SELECT w.item_id, w.user_id,
                   CASE WHEN w.user_id = %s THEN 0 ELSE 1 END as user_priority
            FROM wardrobe_items w
            LEFT JOIN top_attributes_new t ON w.item_id = t.item_id
            LEFT JOIN bottom_attributes_new b ON w.item_id = b.item_id
            LEFT JOIN outer_attributes_new o ON w.item_id = o.item_id
            LEFT JOIN dress_attributes_new d ON w.item_id = d.item_id
            WHERE (w.user_id = %s OR w.is_default = TRUE)
              AND ({where_clause})
            GROUP BY w.item_id, w.user_id
            ORDER BY user_priority, w.item_id DESC
            LIMIT 6
        """
        
        params.extend([user_id, user_id])  # 2개의 %s를 위해 user_id를 2번 추가
        
        try:
            with self.db_conn.cursor() as cur:
                # 👇 쿼리 디버깅 추가
                print(f"\n🔍 실행할 SQL 쿼리:")
                print(f"WHERE 절: {where_clause}")
                print(f"파라미터: {params}")
                
                cur.execute(query, params)
                results = cur.fetchall()
                
                item_ids = [row[0] for row in results]
                print(f"  ✅ 추천된 아이템 ID: {item_ids}")
                
                if len(item_ids) == 0:
                    print(f"  ⚠️ 조건에 맞는 아이템이 없어서 기본 추천 실행...")
                    
                    # 👇 모든 아이템 수 확인
                    cur.execute("SELECT COUNT(*) FROM wardrobe_items WHERE user_id = %s OR is_default = TRUE", (user_id,))
                    total_items = cur.fetchone()[0]
                    print(f"  📊 전체 아이템 수: {total_items}")
                    
                    # 필터 없이 사용자 아이템 우선 추천
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
                    print(f"  ✅ 기본 추천 아이템 ID: {item_ids}")
                
                return item_ids
        
        except Exception as e:
            print(f"  ❌ 추천 생성 오류: {e}")
            import traceback
            traceback.print_exc()
            
            # 트랜잭션 롤백
            try:
                self.db_conn.rollback()
                print("🔄 트랜잭션 롤백 완료")
            except Exception as rollback_error:
                print(f"⚠️ 롤백 실패: {rollback_error}")
            
            return []
    
    def reset_conversation(self, user_id: int):
        """대화 히스토리 초기화"""
        if user_id in self.conversation_history:
            del self.conversation_history[user_id]
            print(f"✅ 사용자 {user_id}의 대화 히스토리 초기화 완료")
    
    def close(self):
        """연결 종료"""
        if self.db_conn:
            self.db_conn.close()
            print("PostgreSQL 연결 종료")
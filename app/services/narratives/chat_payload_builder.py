"""
Chat Payload Builder

채팅 요청을 위한 페이로드 구성 로직을 관리하는 클래스
"""
import logging

from app.schemas.chat.basic_chat_request import BasicChatRequest

logger = logging.getLogger(__name__)

class ChatPayloadBuilder:
    """채팅 요청 페이로드 구성기"""
    
    @staticmethod
    def build_user_message(request: BasicChatRequest) -> str:
        """사용자 메시지 페이로드 구성"""
        # 대화 맥락 구성
        conversation_context = ""
        if request.summary:
            conversation_context = f"이전 대화 요약: {request.summary}\n\n"
        
        # 사용자 정보와 메시지 구성
        user_message = f"""
            {conversation_context}현재 사용자 메시지: {request.input_message}

            사용자 정보:
            - 고객ID: {request.customer_info.customer_id}
            - 이름: {request.customer_info.name}
            - 신용한도: {request.customer_info.credit_limit:,}원
            - 현재잔액: {request.balance:,}원
            - 기분: {request.mood}
            - 취미: {request.hobby}
            """
        
        return user_message.strip()

    @staticmethod
    def build_keyword_user_message(hobby: str, mood: str, credit_limit: int, balance: int) -> str:
        """키워드 생성용 사용자 데이터 블록 (지시문은 프롬프트 템플릿에 있다)"""
        return (
            "사용자 정보:\n"
            f"- 취미: {hobby}\n"
            f"- 기분: {mood}\n"
            f"- 신용한도: {credit_limit:,}원\n"
            f"- 현재잔액: {balance:,}원"
        )

    @staticmethod
    def build_summary_user_message(
            current_summary: str,
            user_message: str,
            assistant_message: str,
            flow: str,
            keyword: str | None = None,
    ) -> str:
        """요약 갱신용 입력 데이터 블록 (지시문은 프롬프트 템플릿에 있다)"""
        cur = (current_summary or "").strip()
        assistant = (assistant_message or "").strip() or "대화 내용을 요약에 반영했어."
        kw_line = f"\n추천 키워드: {keyword.strip()}" if keyword and keyword.strip() else ""
        return (
            f"현재 흐름(flow): {flow}{kw_line}\n\n"
            f"기존 요약본(없으면 빈 문자열): {cur}\n\n"
            "신규 대화:\n"
            f"사용자: {user_message}\n"
            f"레이: {assistant}"
        )


    @staticmethod
    def normalize_hobby(hobby: str) -> str:
        """취미 정규화"""
        return hobby.strip() if hobby else ""
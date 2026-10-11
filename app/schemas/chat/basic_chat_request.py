from typing import Optional

from pydantic import BaseModel, Field
from app.llm.models import PromptSlots
from app.schemas.cache_dto.customer_cache_dto import CustomerCacheDto

"""
FAST API - LLM 요청 기본 DTO

챗 요청을 위한 기본 DTO
"""

class BasicChatRequest(BaseModel):
    
    session_id: str = Field(..., alias="sessionId", description="채팅 세션 아이디")
    customer_info: CustomerCacheDto = Field(..., alias="customerInfo", description="채팅 생성 요청을 위한 고객 기본 정보")
    mood: str = Field(..., description="채팅 생성 요청을 위한 고객 현재 기분")
    hobby: str = Field(..., description="채팅 생성 요청을 위한 고객 취미")
    balance: int = Field(..., description="채팅 생성 요청을 위한 고객 현재 잔액")
    input_message: str = Field(..., alias="inputMessage", description="사용자가 입력한 메세지")
    summary: str = Field(..., description="대화 요약본")
    flow: str = Field(..., description="Spring Boot에서 관리하는 채팅 플로우 상태")
    prev_keyword: Optional[str] = Field(default=None, alias="prevKeyword", description="직전 추천 키워드 (재추천 시 금지어)")
    prompts: PromptSlots = Field(..., description="BE가 DB에서 조회해 내려주는 프롬프트 (main 필수, summary 선택)")

    class Config:
        # camelCase alias 허용
        allow_population_by_field_name = True
        
    @classmethod
    def of(
        cls,
        session_id: str,
        customer_info: CustomerCacheDto,
        mood: str,
        hobby: str,
        balance: int,
        input_message: str,
        summary: str,
        flow: str,
        prompts: PromptSlots,
        prev_keyword: Optional[str] = None,
    ) -> "BasicChatRequest":
        """
        BasicChatRequest 생성을 위한 팩토리 메서드
        자바와 동일한 방식으로 만들기 위함이다.
        """
        return cls(
            session_id=session_id,
            customer_info=customer_info,
            mood=mood,
            hobby=hobby,
            balance=balance,
            input_message=input_message,
            summary=summary,
            flow=flow,
            prev_keyword=prev_keyword,
            prompts=prompts,
        )
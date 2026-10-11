from pydantic import BaseModel, Field, ConfigDict
from app.llm.models import PromptSlots

"""
FAST API - LLM VALID 요청 DTO
"""
class ValidInputRequest(BaseModel):

    customer_id: int = Field(..., alias="customerId", description="고객 아이디")
    session_id: str = Field(..., alias="sessionId", description="채팅 세션 아이디")
    input_message: str = Field(..., alias="inputMessage", description="고객이 입력한 메세지")
    flow: str = Field(..., description="Spring Boot에서 관리하는 채팅 플로우 상태")
    prompts: PromptSlots = Field(..., description="BE가 DB에서 조회해 내려주는 검증 프롬프트 (main 필수)")

    model_config = ConfigDict(
        # camelCase alias 허용
        populate_by_name=True
    )

    @classmethod
    def of(
            cls,
            customer_id: int,
            session_id: str,
            input_message: str,
            flow: str,
            prompts: PromptSlots
    ) -> "ValidInputRequest":
        """
        ValidInputRequest 생성을 위한 팩토리 메서드
        자바와 동일한 방식으로 만들기 위함이다.
        """
        return cls(
            customer_id=customer_id,
            session_id=session_id,
            input_message=input_message,
            flow=flow,
            prompts=prompts
        )
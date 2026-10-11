from typing import Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict
from app.schemas.recommend.product_dto import ProductDto
from app.llm.models import CallMeta

"""
채팅 기본 응답 DTO
"""

class BasicChatResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    session_id: str = Field(..., alias="sessionId", description="채팅 세션 아이디")
    message: Optional[str] = Field(default=None, description="llm에서 응답한 chat message. main 호출 실패 시 null")
    new_hobby: str = Field(..., alias="newHobby", description="새로 바뀐 취미")
    products: Optional[List[ProductDto]] = Field(default=None, description="추천 상품 목록")
    summary: str = Field(..., description="대화 요약본")
    flow: str = Field(..., description="다음 플로우 정의")
    keyword: Optional[str] = Field(default=None, description="추천 단계에서 실제로 검색에 쓴 키워드")
    result: dict[str, Any] = Field(default_factory=dict, description="슬롯별 구조화 결과 {main:{...}, summary:{...}}")
    calls: List[CallMeta] = Field(default_factory=list, description="LLM 호출 메타데이터 (호출 1건당 1개)")
    
    @classmethod
    def of(cls, message: Optional[str], session_id: str, new_hobby: str, flow: str, summary: str,
           products: Optional[List[ProductDto]] = None, keyword: Optional[str] = None,
           result: Optional[dict] = None, calls: Optional[List[CallMeta]] = None) -> "BasicChatResponse":
        return cls(
            session_id=session_id,
            message=message,
            new_hobby=new_hobby,
            products=products,
            summary=summary,
            flow=flow,
            keyword=keyword,
            result=result or {},
            calls=calls or [],
        )
"""BE가 요청에 실어 보내는 프롬프트와, 응답에 돌려주는 호출 메타데이터.

ML은 프롬프트를 저장하지 않는다. 받은 내용으로 호출만 하고 메타데이터를 만든다.
"""
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class PromptSpec(BaseModel):
    """프롬프트 버전 1건 (BE의 prompt_version 행)."""
    model_config = ConfigDict(populate_by_name=True)

    prompt_key: str = Field(..., alias="promptKey", min_length=1)
    prompt_version: int = Field(..., alias="promptVersion", ge=1)
    content: str = Field(..., description="{{변수}} 자리표시자가 있는 시스템 프롬프트 템플릿")
    response_schema: dict[str, Any] = Field(..., alias="responseSchema")
    model: Optional[str] = Field(default=None, description="비우면 ML 기본 모델")


class PromptSlots(BaseModel):
    """한 요청에서 쓰는 프롬프트. main 은 필수, summary(요약 갱신)는 선택."""
    model_config = ConfigDict(populate_by_name=True)

    main: PromptSpec
    summary: Optional[PromptSpec] = None


class CallMeta(BaseModel):
    """LLM 호출 1건의 메타데이터. BE가 llm_call_log 로 저장한다."""
    model_config = ConfigDict(populate_by_name=True)

    slot: str
    prompt_key: str = Field(..., alias="promptKey")
    prompt_version: int = Field(..., alias="promptVersion")
    model: str
    latency_ms: int = Field(..., alias="latencyMs")
    tokens_in: int = Field(0, alias="tokensIn")
    tokens_out: int = Field(0, alias="tokensOut")
    valid: bool
    error_code: Optional[str] = Field(default=None, alias="errorCode")

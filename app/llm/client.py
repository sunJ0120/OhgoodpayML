"""LLM 호출 클라이언트. 테스트에서는 같은 인터페이스의 가짜 클라이언트로 바꿔 끼운다."""
import re
from dataclasses import dataclass
from typing import Any, Protocol


class LlmError(Exception):
    """LLM 호출 실패 (타임아웃 제외)."""


class LlmTimeout(LlmError):
    """LLM 호출 타임아웃."""


@dataclass
class LlmCompletion:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0


class LlmClient(Protocol):
    async def complete(
        self, *, model: str, system: str, user: str,
        response_schema: dict[str, Any], schema_name: str,
    ) -> LlmCompletion: ...


class OpenAiLlmClient:
    """OpenAI Chat Completions + 구조화 출력(json_schema, strict)."""

    default_model: str

    def __init__(self) -> None:
        from app.config.openai_config import openai_config  # 키가 없으면 여기서 실패 → 테스트에서는 import 안 함

        self._cfg = openai_config
        self.default_model = openai_config.model

    async def complete(self, *, model, system, user, response_schema, schema_name) -> LlmCompletion:
        from openai import APITimeoutError

        try:
            resp = await self._cfg.get_client().chat.completions.create(
                model=model or self.default_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=self._cfg.max_tokens,
                temperature=self._cfg.temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": re.sub(r"[^A-Za-z0-9_-]", "_", schema_name)[:64] or "response",
                        "schema": response_schema,
                        "strict": True,
                    },
                },
            )
        except APITimeoutError as e:
            raise LlmTimeout(str(e)) from e
        except Exception as e:
            raise LlmError(str(e)) from e

        usage = resp.usage
        return LlmCompletion(
            text=resp.choices[0].message.content or "",   # refusal 이면 빈 문자열 → SCHEMA_INVALID
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
        )

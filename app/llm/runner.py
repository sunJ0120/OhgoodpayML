"""템플릿 렌더링 → LLM 호출 → 스키마 재검증 → 메타데이터 생성.

실패해도 예외로 죽지 않고 meta.valid=False 와 error_code 로 돌려준다.
error_code: TIMEOUT, SCHEMA_INVALID, LLM_ERROR, TEMPLATE_ERROR
"""
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import jsonschema

from app.llm.client import LlmClient, LlmTimeout
from app.llm.models import CallMeta, PromptSpec

logger = logging.getLogger(__name__)

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


class TemplateError(ValueError):
    pass


def render_template(content: str, variables: dict[str, Any]) -> str:
    """{{name}} 을 한 번만 치환한다 (치환된 값 안의 {{…}} 는 다시 펼치지 않는다)."""
    def sub(m: re.Match) -> str:
        name = m.group(1)
        if name not in variables:
            raise TemplateError(f"템플릿 변수 누락: {name}")
        return str(variables[name])

    return _PLACEHOLDER.sub(sub, content)


@dataclass
class CallResult:
    result: Optional[dict[str, Any]]
    meta: CallMeta

    @property
    def ok(self) -> bool:
        return self.meta.valid and self.result is not None


class LlmRunner:
    def __init__(self, client: LlmClient, *, clock: Callable[[], float] = time.perf_counter):
        self._client = client
        self._clock = clock
        self._default_model = getattr(client, "default_model", "gpt-4o-mini")

    async def run(self, slot: str, spec: PromptSpec, variables: dict[str, Any], user_message: str) -> CallResult:
        model = spec.model or self._default_model

        def meta(valid: bool, code: Optional[str] = None, latency_ms: int = 0, tin: int = 0, tout: int = 0) -> CallMeta:
            return CallMeta(slot=slot, promptKey=spec.prompt_key, promptVersion=spec.prompt_version,
                            model=model, latencyMs=latency_ms, tokensIn=tin, tokensOut=tout,
                            valid=valid, errorCode=code)

        try:
            system = render_template(spec.content, variables)
        except TemplateError as e:
            logger.error("프롬프트 렌더링 실패 key=%s v=%s: %s", spec.prompt_key, spec.prompt_version, e)
            return CallResult(None, meta(False, "TEMPLATE_ERROR"))

        start = self._clock()
        try:
            completion = await self._client.complete(
                model=model, system=system, user=user_message,
                response_schema=spec.response_schema, schema_name=spec.prompt_key,
            )
        except LlmTimeout:
            return CallResult(None, meta(False, "TIMEOUT", self._ms(start)))
        except Exception as e:  # LlmError 포함 모든 실패
            logger.error("LLM 호출 실패 key=%s v=%s: %s", spec.prompt_key, spec.prompt_version, e)
            return CallResult(None, meta(False, "LLM_ERROR", self._ms(start)))
        latency = self._ms(start)

        parsed = self._parse(completion.text, spec.response_schema)
        if parsed is None:
            return CallResult(None, meta(False, "SCHEMA_INVALID", latency, completion.tokens_in, completion.tokens_out))
        return CallResult(parsed, meta(True, None, latency, completion.tokens_in, completion.tokens_out))

    def _ms(self, start: float) -> int:
        return int(round((self._clock() - start) * 1000))

    @staticmethod
    def _parse(text: str, schema: dict[str, Any]) -> Optional[dict[str, Any]]:
        try:
            data = json.loads(text)
            jsonschema.Draft202012Validator(schema).validate(data)
        except (json.JSONDecodeError, jsonschema.ValidationError, jsonschema.SchemaError, TypeError):
            return None
        return data if isinstance(data, dict) else None

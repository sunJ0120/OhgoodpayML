"""평가 스크립트 테스트용 가짜 클라이언트. 실제 LLM은 부르지 않는다."""
import json
from typing import Any

from app.llm.client import LlmCompletion


class ScriptedClient:
    """순서대로 응답/예외를 돌려준다. 응답이 모자라면 같은 규칙으로 계속 쓰는 fallback(callable)을 쓴다."""

    def __init__(self, *responses: Any, fallback=None):
        self._responses = list(responses)
        self._fallback = fallback
        self.calls: list[dict] = []

    async def complete(self, *, model, system, user, response_schema, schema_name):
        self.calls.append({"model": model, "system": system, "user": user, "schema_name": schema_name})
        if self._responses:
            item = self._responses.pop(0)
        elif self._fallback is not None:
            item = self._fallback(system, user)
        else:
            raise AssertionError("ScriptedClient: 준비된 응답이 없다")
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LlmCompletion):
            return item
        return LlmCompletion(text=item if isinstance(item, str) else json.dumps(item, ensure_ascii=False),
                             tokens_in=100, tokens_out=10)


class StepClock:
    """호출할 때마다 step 초씩 증가 → 호출 1건의 지연이 항상 step*1000 ms"""

    def __init__(self, step: float = 0.1):
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        # start 와 end 가 쌍으로 호출되므로 짝수 번째 호출에서만 시간이 흐른다
        self.n = getattr(self, "n", 0) + 1
        if self.n % 2 == 0:
            self.t += self.step
        return self.t

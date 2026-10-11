"""전체 테스트 공용 안전장치."""
import pytest


@pytest.fixture(autouse=True)
def block_real_openai(monkeypatch):
    """실수로 진짜 OpenAI 클라이언트가 쓰여도 네트워크로 나가지 않고(=과금 없이) 즉시 실패시킨다."""
    from app.llm.client import OpenAiLlmClient

    async def _blocked(self, **kwargs):
        raise AssertionError("테스트에서 실제 OpenAI 호출은 금지되어 있다 (가짜 클라이언트를 쓸 것)")

    monkeypatch.setattr(OpenAiLlmClient, "complete", _blocked)

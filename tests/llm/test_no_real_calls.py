"""테스트에서 실제 OpenAI 호출(과금)이 일어나지 않도록 막는다.

conftest 의 autouse 픽스처가 OpenAiLlmClient.complete 를 실패시킨다.
가짜 클라이언트를 쓰는 테스트에는 영향이 없고, 실수로 진짜 클라이언트가 쓰이면 즉시 실패한다.
"""
import asyncio

import pytest

from app.llm.client import OpenAiLlmClient


def test_real_openai_client_is_blocked_in_tests():
    client = OpenAiLlmClient()
    with pytest.raises(AssertionError, match="실제 OpenAI"):
        asyncio.run(client.complete(model="m", system="s", user="u", response_schema={}, schema_name="x"))

"""LLM 테스트 공용 픽스처.

app.config.* 싱글톤은 import 시점에 환경변수를 요구하므로 더미 값을 먼저 넣는다.
실제 LLM/네이버는 부르지 않는다. 가짜 클라이언트와 가짜 검색으로 대체한다.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("NAVER_CLIENT_ID", "test-id")
os.environ.setdefault("NAVER_CLIENT_SECRET", "test-secret")

import json
from typing import Any

from app.llm.client import LlmCompletion
from app.schemas.recommend.product_dto import ProductDto


class FakeLlmClient:
    """미리 정해 둔 응답/예외를 순서대로 돌려주고, 받은 인자를 기록한다."""

    def __init__(self, *responses: Any):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, *, model, system, user, response_schema, schema_name):
        self.calls.append(
            {"model": model, "system": system, "user": user,
             "response_schema": response_schema, "schema_name": schema_name}
        )
        if not self._responses:
            raise AssertionError("FakeLlmClient: 준비된 응답이 없다")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LlmCompletion):
            return item
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return LlmCompletion(text=text, tokens_in=100, tokens_out=20)


class FakeNaver:
    def __init__(self, products: list[ProductDto] | None = None):
        self.products = products if products is not None else []
        self.queries: list[tuple[str, int]] = []

    async def search_products(self, query: str, display: int):
        self.queries.append((query, display))
        return self.products


def make_product(rank: int, price: int, name: str = "상품") -> ProductDto:
    return ProductDto(rank=rank, name=f"{name}{rank}", price=price,
                      image="http://img", url="http://url", category="c")


MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
    "additionalProperties": False,
}
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}
KEYWORD_SCHEMA = {
    "type": "object",
    "properties": {
        "keyword": {"type": "string"},
        "price_min": {"type": "integer"},
        "price_max": {"type": "integer"},
    },
    "required": ["keyword", "price_min", "price_max"],
    "additionalProperties": False,
}
VALID_SCHEMA = {
    "type": "object",
    "properties": {"is_valid": {"type": "boolean"}, "normalized": {"type": "string"}},
    "required": ["is_valid", "normalized"],
    "additionalProperties": False,
}


def prompt(key: str, version: int, schema: dict, content: str = "시스템 {{user_name}}") -> dict:
    return {"promptKey": key, "promptVersion": version, "content": content,
            "responseSchema": schema, "model": "gpt-4o-mini"}


def chat_payload(flow: str = "choose", *, main: dict | None = None,
                 summary: dict | None = None, **overrides) -> dict:
    prompts: dict = {}
    prompts["main"] = main if main is not None else prompt("chat_choose", 3, MESSAGE_SCHEMA)
    if summary is not None:
        prompts["summary"] = summary
    body = {
        "sessionId": "s-1",
        "flow": flow,
        "inputMessage": "요즘 러닝에 빠졌어",
        "customerInfo": {"customerId": 1, "name": "오선정", "creditLimit": 300000},
        "mood": "positive",
        "hobby": "게임",
        "balance": 120000,
        "summary": "기존 요약",
        "prevKeyword": None,
        "prompts": prompts,
    }
    body.update(overrides)
    return body


def summary_prompt(version: int = 2) -> dict:
    return prompt("summary_update", version, SUMMARY_SCHEMA, content="요약 시스템")



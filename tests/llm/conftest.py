"""LLM 테스트 공용 픽스처. 더미 환경변수는 llm_helpers 에서 먼저 설정한다."""
import llm_helpers  # noqa: F401  (환경변수 더미 설정이 app import 보다 먼저 일어나야 한다)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import chat
from llm_helpers import FakeLlmClient, FakeNaver


@pytest.fixture()
def naver() -> FakeNaver:
    return FakeNaver()


@pytest.fixture()
def make_client(naver):
    """make_client(FakeLlmClient(...)) -> (TestClient)"""
    from app.llm import deps

    def _make(fake_llm: FakeLlmClient) -> TestClient:
        app = FastAPI()
        app.include_router(chat.router)
        app.dependency_overrides[deps.get_llm_client] = lambda: fake_llm
        app.dependency_overrides[deps.get_naver_service] = lambda: naver
        return TestClient(app)

    return _make

"""FastAPI 의존성. 테스트에서는 get_llm_client / get_naver_service 를 override 한다."""
from fastapi import Depends

from app.domain.chat.chat_service import ChatService
from app.domain.recommend.recommend_service import RecommendService
from app.domain.recommend.validation_service import ValidationService
from app.llm.client import LlmClient, OpenAiLlmClient
from app.llm.runner import LlmRunner
from app.services.naver_shopping_service import NaverShoppingService


def get_llm_client() -> LlmClient:
    return OpenAiLlmClient()


def get_naver_service() -> NaverShoppingService:
    return NaverShoppingService()


def get_llm_runner(client: LlmClient = Depends(get_llm_client)) -> LlmRunner:
    return LlmRunner(client)


def get_chat_service(
        runner: LlmRunner = Depends(get_llm_runner),
        naver: NaverShoppingService = Depends(get_naver_service),
) -> ChatService:
    return ChatService(runner, RecommendService(runner, naver))


def get_validation_service(runner: LlmRunner = Depends(get_llm_runner)) -> ValidationService:
    return ValidationService(runner)

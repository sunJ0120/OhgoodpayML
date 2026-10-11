import logging

from app.domain.chat.flows import Flow
from app.domain.recommend.recommend_service import RecommendService
from app.llm.runner import LlmRunner
from app.schemas.chat.basic_chat_request import BasicChatRequest
from app.schemas.chat.basic_chat_response import BasicChatResponse
from app.services.narratives.chat_payload_builder import ChatPayloadBuilder

logger = logging.getLogger(__name__)


class ChatService:
    """프롬프트는 요청(prompts)으로 받는다. 이 서비스는 단계별 호출 순서와 결과 조립만 맡는다."""

    def __init__(self, runner: LlmRunner, recommend_service: RecommendService):
        self.runner = runner
        self.recommend_service = recommend_service

    async def generate_chat(self, request: BasicChatRequest) -> BasicChatResponse:
        request.hobby = ChatPayloadBuilder.normalize_hobby(request.hobby)

        calls: list = []
        result: dict = {}
        message: str | None = None
        keyword: str | None = None
        new_hobby = ""
        products: list = []
        main_ok = True

        if request.flow in (Flow.RECOMMENDATION.value, Flow.RE_RECOMMENDATION.value):
            # 추천 단계: 키워드 생성(LLM) → 상품 검색(네이버)
            outcome = await self.recommend_service.generate_keywords(
                request.prompts.main,
                user_name=request.customer_info.name,
                hobby=request.hobby,
                mood=request.mood,
                credit_limit=request.customer_info.credit_limit,
                balance=request.balance,
                prev_keyword=request.prev_keyword,
            )
            calls.extend(outcome.calls)
            if outcome.keyword is None:
                main_ok = False
            else:
                keyword = outcome.keyword
                result["main"] = outcome.result
                message = ""
                products = await self.recommend_service.search_products_async(keyword, outcome.price_range) or []
        else:
            # 그 외 단계(choose/hobby_check/알 수 없는 flow): 프롬프트는 BE가 단계에 맞게 골라 내려준다
            res = await self.runner.run(
                "main", request.prompts.main,
                {"user_name": request.customer_info.name, "hobby": request.hobby},
                ChatPayloadBuilder.build_user_message(request),
            )
            calls.append(res.meta)
            if res.ok:
                message = res.result["message"]
                result["main"] = res.result
                if request.flow == Flow.CHOOSE.value:
                    new_hobby = request.input_message
            else:
                main_ok = False

        # 요약 갱신: 본 응답이 성공했고 summary 프롬프트가 있을 때만. 실패해도 대화는 막지 않는다.
        summary = request.summary
        if main_ok and request.prompts.summary is not None:
            res = await self.runner.run(
                "summary", request.prompts.summary, {},
                ChatPayloadBuilder.build_summary_user_message(
                    request.summary, request.input_message, message or "", request.flow, keyword),
            )
            calls.append(res.meta)
            if res.ok:
                summary = res.result["summary"]
                result["summary"] = res.result

        return BasicChatResponse.of(
            message=message,
            session_id=request.session_id,
            new_hobby=new_hobby,
            products=products,
            summary=summary,
            flow=request.flow,
            keyword=keyword,
            result=result,
            calls=calls,
        )

import logging
import re
from dataclasses import dataclass, field

from app.llm.models import PromptSpec
from app.llm.runner import CallResult, LlmRunner
from app.schemas.recommend.product_dto import ProductDto
from app.services.narratives.chat_payload_builder import ChatPayloadBuilder

logger = logging.getLogger(__name__)

"""
Recommend domain module

추천 서비스 관련 비즈니스 로직을 담당합니다.
키워드 생성 프롬프트는 BE가 내려주고(keyword_extract), 여기서는 호출·금지어 검사·상품 검색만 한다.
"""


@dataclass
class KeywordOutcome:
    keyword: str | None                 # None 이면 LLM 호출 실패
    price_range: str | None
    result: dict | None                 # 마지막 성공 호출의 구조화 결과
    calls: list = field(default_factory=list)  # list[CallMeta]
    fallback: bool = False              # 재시도 후에도 금지어와 겹쳐 안전 기본값으로 대체했는가


class RecommendService:
    def __init__(self, runner: LlmRunner, naver_shopping_service):
        self.runner = runner
        self.naver_shopping_service = naver_shopping_service

    @staticmethod
    def _normalize_kw(s: str) -> str:
        """비교용 정규화: 소문자, 공백/특수문자 제거."""
        if not s:
            return ""
        return re.sub(r"[\s\-\_·,./\\()\"'`~!@#$%^&*+=?:;]", "", s.lower())

    async def generate_keywords(
            self,
            spec: PromptSpec,
            *,
            user_name: str,
            hobby: str,
            mood: str,
            credit_limit: int,
            balance: int,
            prev_keyword: str | None,
    ) -> KeywordOutcome:
        """'새로운' 키워드와 가격대 생성. 취미/직전 키워드와 겹치면 1회 재시도, 그래도 겹치면 안전 기본값."""
        ban_set = {self._normalize_kw(hobby)} if hobby else set()
        if prev_keyword:
            ban_set.add(self._normalize_kw(prev_keyword))
        ban_set.discard("")

        banned = [w for w in (hobby, prev_keyword) if w]
        ban_clause = ""
        if banned:
            ban_clause = (
                "\n[금지 키워드]\n"
                f"- 다음 단어/표현 및 그 동의어/띄어쓰기 변형/영문표기/복수형은 절대 사용 금지: {', '.join(banned)}\n"
            )
        user_message = ChatPayloadBuilder.build_keyword_user_message(hobby, mood, credit_limit, balance)
        calls: list = []

        async def one_shot(avoid_note: str) -> CallResult:
            res = await self.runner.run(
                "main", spec,
                {"user_name": user_name, "hobby": hobby, "mood": mood,
                 "ban_clause": ban_clause, "avoid_note": avoid_note},
                user_message,
            )
            calls.append(res.meta)
            return res

        res = await one_shot("")
        if not res.ok:
            return KeywordOutcome(None, None, None, calls)

        if self._normalize_kw(res.result["keyword"]) in ban_set:
            logger.warning("생성 키워드가 금지어와 충돌: %r (retry 1)", res.result["keyword"])
            res = await one_shot(
                f"- 참고: 직전 시도 키워드 '{res.result['keyword']}'가 금지어와 충돌했다. "
                "금지어와 의미가 겹치지 않는 완전히 다른 항목으로 새 키워드를 제시하라."
            )
            if not res.ok:
                return KeywordOutcome(None, None, None, calls)

        keyword = res.result["keyword"].strip()
        lo, hi = sorted((res.result["price_min"], res.result["price_max"]))
        price_range = f"{lo}-{hi}"

        fallback = False
        if self._normalize_kw(keyword) in ban_set:
            logger.error("재시도 후에도 금지어 충돌. 안전 기본값으로 대체.")
            keyword = f"{hobby} 소모품" if hobby else "소모품"
            price_range = "10000-30000"
            fallback = True

        return KeywordOutcome(keyword, price_range, res.result, calls, fallback)

    async def search_products_async(self, keyword: str, price_range: str) -> list[ProductDto]:
        """
        내부 서비스용: 상품 검색 - 네이버 쇼핑 API 연동

        필터링 결과 없을때도 나올 수 있도록 로직 수정.
        """
        try:
            # 가격대 필터링이 있을 때는 더 많은 상품을 가져와서 필터링 후 10개 확보
            display_count = 50 if price_range and "-" in price_range else 10

            # 네이버 쇼핑 API로 상품 검색
            products = await self.naver_shopping_service.search_products(keyword, display=display_count)

            # 가격대 필터링 (선택적)
            if price_range and "-" in price_range:
                try:
                    min_price, max_price = map(int, price_range.split("-"))
                    filtered_products = [
                        p for p in products
                        if min_price <= p.price <= max_price
                    ]

                    # 필터링 결과가 3개 이하면 원본에서 채워넣기
                    if len(filtered_products) <= 3:
                        logger.info(f"필터링 결과가 {len(filtered_products)}개로 부족하여 원본 데이터로 채움")
                        products = products[:10]
                    else:
                        # 필터링 결과가 충분하면 최대 10개만 반환
                        products = filtered_products[:10]
                        logger.info(f"필터링 결과: {len(filtered_products)}개 중 {len(products)}개 반환")

                except ValueError:
                    logger.warning(f"가격대 파싱 실패: {price_range}")
                    products = products[:10]  # 필터링 실패시 원본에서 10개
            else:
                products = products[:10]

            return products

        except Exception as e:
            logger.error(f"상품 검색 실패: keyword={keyword}, error={e}")
            return []

import logging
from dataclasses import dataclass

from app.domain.chat.messages import canonical_invalid
from app.llm.models import CallMeta, PromptSpec
from app.llm.runner import LlmRunner

logger = logging.getLogger(__name__)


@dataclass
class ValidationOutcome:
    is_valid: bool
    normalized: str
    message: str
    call: CallMeta


class ValidationService:
    def __init__(self, runner: LlmRunner):
        self.runner = runner

    async def validate_input_for_flow(self, flow: str, input_message: str, spec: PromptSpec) -> ValidationOutcome:
        """현재 flow에 맞는 입력인지 LLM으로 검증한다.

        LLM 호출 자체가 실패하면 is_valid=False 에 안내문을 비우고 call.error_code 로 알린다
        (사용자 입력 탓의 INVALID 와 구분하기 위함).
        """
        res = await self.runner.run("main", spec, {}, input_message)
        logger.info("검증 flow=%s key=%s v=%s valid=%s error=%s",
                    flow, spec.prompt_key, spec.prompt_version, res.meta.valid, res.meta.error_code)

        if not res.ok:
            return ValidationOutcome(False, "", "", res.meta)
        if res.result["is_valid"]:
            return ValidationOutcome(True, res.result["normalized"], "", res.meta)
        return ValidationOutcome(False, "", canonical_invalid(flow), res.meta)

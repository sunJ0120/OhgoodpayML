"""평가 계산 검증: 가짜 클라이언트로 돌려 스키마 통과율·기대 결과 일치율 등을 정확히 계산하는지 본다."""
import asyncio

import pytest

from app.llm.client import LlmError, LlmTimeout
from app.llm.models import PromptSpec
from app.llm.runner import LlmRunner
from eval.evalset import EvalCase, EvalSet, Expected, KeywordContext
from eval.metrics import evaluate
from eval_helpers import ScriptedClient, StepClock

VALID_SCHEMA = {"type": "object",
                "properties": {"is_valid": {"type": "boolean"}, "normalized": {"type": "string"}},
                "required": ["is_valid", "normalized"], "additionalProperties": False}
KEYWORD_SCHEMA = {"type": "object",
                  "properties": {"keyword": {"type": "string"}, "price_min": {"type": "integer"}, "price_max": {"type": "integer"}},
                  "required": ["keyword", "price_min", "price_max"], "additionalProperties": False}


def run(coro):
    return asyncio.run(coro)


def mood_set() -> EvalSet:
    def case(i, cat, text, valid, norm):
        return EvalCase(id=i, category=cat, input=text, expected=Expected(is_valid=valid, normalized=norm))
    return EvalSet(prompt_key="validate_mood", kind="validation", cases=[
        case("a", "normal", "기분 좋아", True, "positive"),       # → 정답
        case("b", "normal", "우울해", True, "negative"),           # → normalized 틀림
        case("c", "edge_nonsense", "asdf", False, ""),             # → 스키마 위반
        case("d", "edge_empty", "", False, ""),                    # → 타임아웃
    ])


def mood_prompt() -> PromptSpec:
    return PromptSpec(promptKey="validate_mood", promptVersion=1, content="규칙", responseSchema=VALID_SCHEMA, model="m")


def test_validation_report_numbers():
    client = ScriptedClient(
        {"is_valid": True, "normalized": "positive"},
        {"is_valid": True, "normalized": "neutral"},
        "not json",
        LlmTimeout("slow"),
    )
    rep = run(evaluate(mood_set(), mood_prompt(), LlmRunner(client, clock=StepClock(0.1)), label="v1", concurrency=1))

    assert rep.total == 4
    assert rep.schema_ok == 2 and rep.schema_pass_rate == 0.5
    assert rep.correct == 1 and rep.correct_rate == 0.25
    assert dict(rep.errors) == {"SCHEMA_INVALID": 1, "TIMEOUT": 1}
    assert rep.tokens_in == 300 and rep.tokens_out == 30            # 응답을 받은 3건(스키마 위반 응답 포함)만 토큰 집계, 타임아웃은 0
    assert rep.avg_latency_ms == 100
    by = rep.by_category()
    assert by["normal"] == (1, 2) and by["edge_nonsense"] == (0, 1) and by["edge_empty"] == (0, 1)


def test_invalid_expected_ignores_normalized_and_wrong_validity_is_a_miss():
    s = EvalSet(prompt_key="validate_mood", kind="validation", cases=[
        EvalCase(id="x", category="normal", input="안녕", expected=Expected(is_valid=False, normalized=None)),
        EvalCase(id="y", category="normal", input="안녕", expected=Expected(is_valid=False, normalized=None)),
    ])
    client = ScriptedClient({"is_valid": False, "normalized": ""}, {"is_valid": True, "normalized": "neutral"})
    rep = run(evaluate(s, mood_prompt(), LlmRunner(client), label="v1", concurrency=1))
    assert rep.correct == 1 and rep.schema_ok == 2


def test_any_of_normalized_options():
    s = EvalSet(prompt_key="validate_hobby", kind="validation", cases=[
        EvalCase(id="h", category="normal", input="넷플 봐", expected=Expected(is_valid=True, normalized=["넷플", "넷플릭스"]))])
    rep = run(evaluate(s, mood_prompt(), LlmRunner(ScriptedClient({"is_valid": True, "normalized": "넷플릭스"})), label="v1"))
    assert rep.correct == 1


def test_results_keep_case_order_even_when_run_concurrently():
    s = mood_set()
    client = ScriptedClient(fallback=lambda system, user: {"is_valid": True, "normalized": "positive"})
    rep = run(evaluate(s, mood_prompt(), LlmRunner(client), label="v1", concurrency=4))
    assert [r.case_id for r in rep.results] == ["a", "b", "c", "d"]


# ---------- keyword_extract ----------

def kw_case(cid, hobby="러닝", prev=None, credit=300_000, balance=120_000, cat="normal"):
    return EvalCase(id=cid, category=cat, input="", context=KeywordContext(
        hobby=hobby, mood="positive", credit_limit=credit, balance=balance, prev_keyword=prev))


def kw_prompt() -> PromptSpec:
    return PromptSpec(promptKey="keyword_extract", promptVersion=1, content="{{ban_clause}}규칙{{avoid_note}}",
                      responseSchema=KEYWORD_SCHEMA, model="m")


def test_keyword_checks_ban_price_cap_and_fallback():
    s = EvalSet(prompt_key="keyword_extract", kind="keyword", cases=[
        kw_case("ok"),
        kw_case("over_cap"),          # price_max 가 min(한도, 잔액)=120000 초과
        kw_case("banned_twice"),      # 두 번 다 금지어 → 안전 기본값 폴백
        kw_case("retry_ok"),          # 첫 시도 금지어 → 재시도 성공(정상)
        kw_case("bad_json"),
    ])
    client = ScriptedClient(
        {"keyword": "러닝화", "price_min": 30000, "price_max": 100000},        # ok
        {"keyword": "러닝화", "price_min": 30000, "price_max": 200000},        # over_cap
        {"keyword": "러닝", "price_min": 1, "price_max": 2},                   # banned_twice (1)
        {"keyword": "러닝", "price_min": 1, "price_max": 2},                   # banned_twice (2)
        {"keyword": "러닝", "price_min": 1, "price_max": 2},                   # retry_ok (1)
        {"keyword": "아대", "price_min": 10000, "price_max": 50000},           # retry_ok (2)
        "garbage",                                                             # bad_json
    )
    rep = run(evaluate(s, kw_prompt(), LlmRunner(client), label="v1", concurrency=1))

    by_id = {r.case_id: r for r in rep.results}
    assert by_id["ok"].correct is True
    assert by_id["over_cap"].correct is False and "price" in by_id["over_cap"].detail
    assert by_id["banned_twice"].correct is False and "fallback" in by_id["banned_twice"].detail
    assert by_id["retry_ok"].correct is True and by_id["retry_ok"].llm_calls == 2
    assert by_id["bad_json"].schema_valid is False
    assert rep.schema_ok == 4 and rep.correct == 2
    assert rep.llm_calls == 1 + 1 + 2 + 2 + 1


def test_prev_keyword_is_enforced_as_banned():
    s = EvalSet(prompt_key="keyword_extract", kind="keyword", cases=[kw_case("p", hobby="운동", prev="러닝화")])
    client = ScriptedClient({"keyword": "러닝화", "price_min": 1, "price_max": 2}, {"keyword": "러닝화", "price_min": 1, "price_max": 2})
    rep = run(evaluate(s, kw_prompt(), LlmRunner(client), label="v1", concurrency=1))
    assert rep.results[0].correct is False


def test_llm_error_counts_as_schema_failure():
    s = EvalSet(prompt_key="keyword_extract", kind="keyword", cases=[kw_case("e")])
    rep = run(evaluate(s, kw_prompt(), LlmRunner(ScriptedClient(LlmError("boom"))), label="v1", concurrency=1))
    assert rep.schema_ok == 0 and dict(rep.errors) == {"LLM_ERROR": 1}

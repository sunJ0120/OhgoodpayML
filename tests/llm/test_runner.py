import asyncio

import pytest

from app.llm.client import LlmCompletion, LlmError, LlmTimeout
from app.llm.models import PromptSpec
from app.llm.runner import LlmRunner, TemplateError, render_template
from llm_helpers import MESSAGE_SCHEMA, FakeLlmClient


def spec(content="안녕 {{user_name}}") -> PromptSpec:
    return PromptSpec(promptKey="chat_choose", promptVersion=7, content=content,
                      responseSchema=MESSAGE_SCHEMA, model="gpt-4o-mini")


def make_runner(fake, ticks=(0.0, 0.25)):
    it = iter(ticks)
    return LlmRunner(fake, clock=lambda: next(it))


def run(coro):
    return asyncio.run(coro)


def test_render_template_replaces_placeholders():
    assert render_template("a {{x}} b {{ y }}", {"x": "1", "y": 2}) == "a 1 b 2"


def test_render_template_missing_variable_raises():
    with pytest.raises(TemplateError):
        render_template("a {{x}}", {})


def test_render_template_does_not_reexpand_values():
    assert render_template("{{a}}", {"a": "{{b}}", "b": "X"}) == "{{b}}"


def test_valid_response_is_parsed_with_meta():
    fake = FakeLlmClient(LlmCompletion(text='{"message": "안녕"}', tokens_in=11, tokens_out=3))
    res = run(make_runner(fake).run("main", spec(), {"user_name": "오"}, "유저 메시지"))

    assert res.ok
    assert res.result == {"message": "안녕"}
    m = res.meta
    assert (m.slot, m.prompt_key, m.prompt_version, m.model) == ("main", "chat_choose", 7, "gpt-4o-mini")
    assert (m.tokens_in, m.tokens_out, m.latency_ms) == (11, 3, 250)
    assert m.valid is True and m.error_code is None


def test_client_receives_rendered_system_and_schema():
    fake = FakeLlmClient({"message": "x"})
    run(make_runner(fake).run("main", spec(), {"user_name": "오선정"}, "유저"))

    call = fake.calls[0]
    assert call["system"] == "안녕 오선정"
    assert call["user"] == "유저"
    assert call["response_schema"] == MESSAGE_SCHEMA
    assert call["model"] == "gpt-4o-mini"


@pytest.mark.parametrize("bad", ['{"message": 3}', '{"other": "x"}', "not json", "", '{"message": "x", "extra": 1}'])
def test_schema_violation_returns_invalid_without_raising(bad):
    fake = FakeLlmClient(bad)
    res = run(make_runner(fake).run("main", spec(), {"user_name": "a"}, "u"))

    assert not res.ok
    assert res.result is None
    assert res.meta.valid is False
    assert res.meta.error_code == "SCHEMA_INVALID"
    assert res.meta.tokens_in == 100  # 응답은 받았으므로 토큰은 기록


def test_timeout_maps_to_timeout_code():
    res = run(make_runner(FakeLlmClient(LlmTimeout("slow"))).run("main", spec(), {"user_name": "a"}, "u"))
    assert (res.meta.valid, res.meta.error_code) == (False, "TIMEOUT")
    assert res.meta.latency_ms == 250


def test_other_client_error_maps_to_llm_error():
    res = run(make_runner(FakeLlmClient(LlmError("boom"))).run("main", spec(), {"user_name": "a"}, "u"))
    assert (res.meta.valid, res.meta.error_code) == (False, "LLM_ERROR")


def test_unexpected_exception_maps_to_llm_error():
    res = run(make_runner(FakeLlmClient(RuntimeError("boom"))).run("main", spec(), {"user_name": "a"}, "u"))
    assert res.meta.error_code == "LLM_ERROR"


def test_missing_template_variable_does_not_call_llm():
    fake = FakeLlmClient()
    res = run(make_runner(fake, ticks=(0.0, 0.0)).run("main", spec(), {}, "u"))

    assert fake.calls == []
    assert (res.meta.valid, res.meta.error_code) == (False, "TEMPLATE_ERROR")

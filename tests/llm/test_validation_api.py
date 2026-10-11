import pytest

from app.llm.client import LlmTimeout
from llm_helpers import VALID_SCHEMA, FakeLlmClient, prompt


def payload(flow="mood_check", with_prompts=True):
    body = {"customerId": 1, "sessionId": "s-1", "inputMessage": "기분 좋아", "flow": flow}
    if with_prompts:
        body["prompts"] = {"main": prompt("validate_mood", 5, VALID_SCHEMA, content="검증 규칙")}
    return body


def post(client, body):
    return client.post("/chat/validation", json=body)


def test_missing_prompts_returns_422(make_client):
    assert post(make_client(FakeLlmClient()), payload(with_prompts=False)).status_code == 422


def test_valid_input_returns_normalized_value_and_meta(make_client):
    fake = FakeLlmClient({"is_valid": True, "normalized": "positive"})
    res = post(make_client(fake), payload())

    assert res.status_code == 200
    body = res.json()
    assert body["isValid"] is True
    assert body["normalized"] == "positive"
    assert body["message"] == ""
    assert body["flow"] == "mood_check"
    call = body["calls"][0]
    assert (call["slot"], call["promptKey"], call["promptVersion"]) == ("main", "validate_mood", 5)
    assert call["valid"] is True
    assert fake.calls[0]["system"] == "검증 규칙"
    assert fake.calls[0]["user"] == "기분 좋아"


@pytest.mark.parametrize("flow,expected", [
    ("mood_check", "기분을 파악하지 못했어ㅠㅠ 다시 입력해줘!"),
    ("hobby_check", "취미/관심사를 알아듣지 못했어ㅠㅠ 다시 입력해줘!"),
    ("choose", "선택을 알아듣지 못했어ㅠㅠ 둘 중 하나를 선택해줘!"),
    ("something-else", "입력을 알아듣지 못했어ㅠㅠ 다시 입력해줘!"),
])
def test_invalid_input_uses_canonical_message_per_flow(make_client, flow, expected):
    fake = FakeLlmClient({"is_valid": False, "normalized": ""})
    body = post(make_client(fake), payload(flow)).json()

    assert body["isValid"] is False
    assert body["normalized"] == ""
    assert body["message"] == expected


def test_schema_violation_is_reported_as_error_not_as_user_invalid(make_client):
    body = post(make_client(FakeLlmClient("VALID: 좋아")), payload()).json()

    assert body["isValid"] is False
    assert body["message"] == ""                      # 사용자 안내문이 아님
    assert body["calls"][0]["valid"] is False
    assert body["calls"][0]["errorCode"] == "SCHEMA_INVALID"


def test_timeout_is_reported_as_timeout(make_client):
    body = post(make_client(FakeLlmClient(LlmTimeout("slow"))), payload()).json()

    assert body["isValid"] is False
    assert body["calls"][0]["errorCode"] == "TIMEOUT"

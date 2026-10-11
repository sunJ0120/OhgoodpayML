from app.llm.client import LlmTimeout
from llm_helpers import (
    KEYWORD_SCHEMA, FakeLlmClient, chat_payload, make_product, prompt, summary_prompt,
)


def post(client, body):
    return client.post("/chat", json=body)


# ---------- 요청 검증 ----------

def test_missing_prompts_returns_422(make_client):
    body = chat_payload()
    del body["prompts"]
    assert post(make_client(FakeLlmClient()), body).status_code == 422


def test_missing_prompt_fields_return_422(make_client):
    for field in ("promptKey", "promptVersion", "content", "responseSchema"):
        body = chat_payload()
        del body["prompts"]["main"][field]
        assert post(make_client(FakeLlmClient()), body).status_code == 422, field


# ---------- 일반 단계 (choose / hobby_check) ----------

def test_choose_flow_returns_message_summary_and_call_meta(make_client):
    fake = FakeLlmClient({"message": "오 좋다 🎁 vs 📊"}, {"summary": "새 요약"})
    res = post(make_client(fake), chat_payload("choose", summary=summary_prompt(2)))

    assert res.status_code == 200
    body = res.json()
    assert body["message"] == "오 좋다 🎁 vs 📊"
    assert body["summary"] == "새 요약"
    assert body["newHobby"] == "요즘 러닝에 빠졌어"
    assert body["flow"] == "choose"
    assert body["result"]["main"] == {"message": "오 좋다 🎁 vs 📊"}

    main, summary = body["calls"]
    assert (main["slot"], main["promptKey"], main["promptVersion"]) == ("main", "chat_choose", 3)
    assert (summary["slot"], summary["promptKey"], summary["promptVersion"]) == ("summary", "summary_update", 2)
    for c in (main, summary):
        assert c["valid"] is True and c["errorCode"] is None
        assert c["model"] == "gpt-4o-mini"
        assert c["tokensIn"] == 100 and c["tokensOut"] == 20
        assert isinstance(c["latencyMs"], int)


def test_system_prompt_is_rendered_from_request_content(make_client):
    fake = FakeLlmClient({"message": "m"})
    post(make_client(fake), chat_payload("choose"))

    assert fake.calls[0]["system"] == "시스템 오선정"
    assert "요즘 러닝에 빠졌어" in fake.calls[0]["user"]


def test_main_schema_violation_returns_error_without_message(make_client):
    fake = FakeLlmClient({"wrong": 1})
    res = post(make_client(fake), chat_payload("choose", summary=summary_prompt()))

    assert res.status_code == 200
    body = res.json()
    assert body["message"] is None
    assert body["summary"] == "기존 요약"          # 요약 갱신은 건너뛴다
    assert len(body["calls"]) == 1
    assert body["calls"][0]["valid"] is False
    assert body["calls"][0]["errorCode"] == "SCHEMA_INVALID"
    assert len(fake.calls) == 1


def test_main_timeout_is_reported_as_timeout(make_client):
    res = post(make_client(FakeLlmClient(LlmTimeout("slow"))), chat_payload("hobby_check"))

    body = res.json()
    assert res.status_code == 200
    assert body["message"] is None
    assert body["calls"][0]["errorCode"] == "TIMEOUT"


def test_summary_failure_keeps_message_and_old_summary(make_client):
    fake = FakeLlmClient({"message": "ok"}, "not json")
    body = post(make_client(fake), chat_payload("choose", summary=summary_prompt())).json()

    assert body["message"] == "ok"
    assert body["summary"] == "기존 요약"
    assert body["calls"][1]["slot"] == "summary"
    assert body["calls"][1]["valid"] is False
    assert body["calls"][1]["errorCode"] == "SCHEMA_INVALID"


def test_summary_slot_is_optional(make_client):
    fake = FakeLlmClient({"message": "ok"})
    body = post(make_client(fake), chat_payload("choose")).json()

    assert len(body["calls"]) == 1
    assert body["summary"] == "기존 요약"


def test_unknown_flow_uses_given_main_prompt(make_client):
    fake = FakeLlmClient({"message": "먼저 기분부터"})
    body = post(make_client(fake), chat_payload("weird", main=prompt("chat_fallback", 1, {
        "type": "object", "properties": {"message": {"type": "string"}},
        "required": ["message"], "additionalProperties": False}))).json()

    assert body["message"] == "먼저 기분부터"
    assert body["calls"][0]["promptKey"] == "chat_fallback"
    assert body["newHobby"] == ""


# ---------- 추천 단계 ----------

def rec_main():
    return prompt("keyword_extract", 4, KEYWORD_SCHEMA,
                  content="{{ban_clause}}키워드 규칙{{avoid_note}}")


def test_recommendation_uses_structured_keyword_and_filters_by_price(make_client, naver):
    naver.products = [make_product(1, 10_000), make_product(2, 40_000), make_product(3, 50_000),
                      make_product(4, 60_000), make_product(5, 70_000), make_product(6, 500_000)]
    fake = FakeLlmClient({"keyword": "러닝화", "price_min": 30000, "price_max": 80000},
                         {"summary": "요약"})
    body = post(make_client(fake), chat_payload("recommendation", main=rec_main(),
                                                summary=summary_prompt())).json()

    assert naver.queries[0][0] == "러닝화"
    assert [p["price"] for p in body["products"]] == [40_000, 50_000, 60_000, 70_000]
    assert body["result"]["main"]["keyword"] == "러닝화"
    assert body["calls"][0]["promptKey"] == "keyword_extract"
    assert body["calls"][0]["promptVersion"] == 4


def test_banned_keyword_triggers_one_retry_and_both_calls_are_recorded(make_client):
    fake = FakeLlmClient({"keyword": "게임", "price_min": 1, "price_max": 2},
                         {"keyword": "보드게임", "price_min": 10000, "price_max": 30000})
    body = post(make_client(fake), chat_payload("recommendation", main=rec_main())).json()

    assert [c["slot"] for c in body["calls"]] == ["main", "main"]
    assert body["result"]["main"]["keyword"] == "보드게임"
    assert body["keyword"] == "보드게임"
    assert "게임" in fake.calls[0]["system"]         # 금지어 안내가 렌더링됨
    assert "게임" in fake.calls[1]["system"] and fake.calls[1]["system"] != fake.calls[0]["system"]


def test_prev_keyword_is_banned_without_parsing_summary(make_client):
    fake = FakeLlmClient({"keyword": "러닝화", "price_min": 1, "price_max": 2},
                         {"keyword": "트레일화", "price_min": 10000, "price_max": 30000})
    req = chat_payload("re-recommendation", main=rec_main(), prevKeyword="러닝화")
    req["summary"] = "요약 끝에 키워드: 다른거"      # 예전에는 이 꼬리표를 정규식으로 파싱했다
    body = post(make_client(fake), req).json()

    assert body["result"]["main"]["keyword"] == "트레일화"
    assert len(body["calls"]) == 2


def test_still_banned_after_retry_falls_back_to_safe_keyword(make_client, naver):
    fake = FakeLlmClient({"keyword": "게임", "price_min": 1, "price_max": 2},
                         {"keyword": "게임", "price_min": 1, "price_max": 2})
    body = post(make_client(fake), chat_payload("recommendation", main=rec_main())).json()

    assert body["keyword"] == "게임 소모품"
    assert naver.queries[0][0] == "게임 소모품"
    assert len(body["calls"]) == 2


def test_keyword_call_failure_reports_error_and_skips_search(make_client, naver):
    fake = FakeLlmClient("garbage")
    body = post(make_client(fake), chat_payload("recommendation", main=rec_main(),
                                                summary=summary_prompt())).json()

    assert body["calls"][0]["errorCode"] == "SCHEMA_INVALID"
    assert naver.queries == []
    assert body["products"] in (None, [])
    assert body["summary"] == "기존 요약"

"""Phase 0: /score 입력 검증.

app.main 은 OPENAI_API_KEY 가 없으면 import 되지 않으므로 score 라우터만 올린 최소 앱을 쓴다.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers import score

FULL_PAYLOAD = {
    "extension_this_month": True,
    "auto_extension_this_month": False,
    "auto_extension_cnt_12m": 3,
    "grade_point": 10,
    "is_blocked": False,
    "payment_cnt_12m": 16,
    "payment_amount_12m": 6_800_000,
    "current_cycle_spend": 180_000,
}

# 변경 전 엔진의 출력을 고정한 값 (특성화 테스트)
GOLDEN = {
    "score": 668,
    "band": "B",
    "grade_name": "브론즈",
    "limit": 100_000,
    "point_percent": 0.002,
    "est_rewards_this_cycle": 360,
    "to_next_grade_points": 10,
}


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(score.router)
    return TestClient(app)


@pytest.mark.parametrize("path", ["/score", "/score/wrapped"])
def test_unknown_key_returns_422(client, path):
    res = client.post(path, json={**FULL_PAYLOAD, "typo_key": 1})
    assert res.status_code == 422


@pytest.mark.parametrize("path", ["/score", "/score/wrapped"])
@pytest.mark.parametrize("missing", sorted(FULL_PAYLOAD))
def test_missing_required_key_returns_422(client, path, missing):
    payload = {k: v for k, v in FULL_PAYLOAD.items() if k != missing}
    res = client.post(path, json=payload)
    assert res.status_code == 422


def test_valid_payload_matches_previous_response(client):
    res = client.post("/score", json=FULL_PAYLOAD)
    assert res.status_code == 200
    body = res.json()
    for key, expected in GOLDEN.items():
        assert body[key] == expected
    assert [r["code"] for r in body["top_negative"]] == ["AUTO_EX_CNT", "GRADE", "UTIL"]
    assert body["top_positive"]["code"] == "COUNT"


def test_wrapped_envelope_keeps_shape(client):
    res = client.post("/score/wrapped", json=FULL_PAYLOAD)
    assert res.status_code == 200
    body = res.json()
    assert (body["success"], body["code"], body["message"]) == ("true", "200", "success")
    assert body["data"]["score"] == GOLDEN["score"]


def test_profile_query_still_works(client):
    res = client.post("/score", params={"profile": "conservative"}, json=FULL_PAYLOAD)
    assert res.status_code == 200
    assert res.json()["score"] != GOLDEN["score"]

"""Phase 6: 런타임 채점 (고정 카드 fixture, 손 계산값과 대조)."""
import copy
import json

import pytest

from app.domain.scoring.scorecard.card import CurrencyMismatchError, ScoreCardError, parse_scorecard
from app.domain.scoring.scorecard.scorer import (
    InvalidFeatureError,
    MissingFeatureError,
    UnknownFeatureError,
    round_score,
    score_customer,
)


def _bin(lower, upper, points, bad_rate=0.2, woe=0.0):
    return {"lower": lower, "upper": upper, "bad_rate": bad_rate, "woe": woe, "points": points}


def card_dict():
    """손으로 계산하기 쉬운 배점. 변수 2개, 최고 배점 합 = 110 + 120 = 230."""
    return {
        "model_version": "fixture-v1",
        "purpose": "pipeline-validation",
        "data_source": "unit test fixture",
        "currency": "TWD",
        "scale": {"base_score": 600, "base_good_odds": 50, "points_per_double": 20},
        "intercept": -1.0,
        "variables": {
            "max_delinquency_months": {
                "beta": -0.9,
                "bins": [_bin(None, 0, 110.0), _bin(0, 1, 95.5), _bin(1, None, 60.25)],
            },
            "paid_amount_6m": {
                "beta": -0.6,
                "bins": [_bin(None, 5000.5, 80.0), _bin(5000.5, None, 120.0)],
            },
        },
        "bands": [
            {"name": "A", "min_score": 220},
            {"name": "B", "min_score": 180},
            {"name": "C", "min_score": 140},
        ],
    }


def four_variable_card_dict():
    d = card_dict()
    d["variables"]["avg_billed_amount_6m"] = {
        "beta": -0.2, "bins": [_bin(None, 100, 20.0), _bin(100, None, 50.0)],
    }
    d["variables"]["active_months"] = {
        "beta": -0.3, "bins": [_bin(None, 3, 5.0), _bin(3, None, 15.0)],
    }
    return d


@pytest.fixture()
def card():
    return parse_scorecard(card_dict())


def run(card, features, currency="TWD"):
    return score_customer(card, features, input_currency=currency)


# ---- 입력 → 구간 → 배점 → 합계 --------------------------------------------
def test_best_customer_scores_the_sum_of_max_points(card):
    r = run(card, {"max_delinquency_months": 0, "paid_amount_6m": 6000})
    assert r.raw_score == pytest.approx(230.0)
    assert r.score == 230
    assert r.band == "A"
    assert r.top_negative == ()


def test_mid_customer_matches_hand_calculation(card):
    # (0, 1] → 95.5,  x == upper 인 5000.5 는 첫 구간 → 80.0
    r = run(card, {"max_delinquency_months": 1, "paid_amount_6m": 5000.5})
    assert r.raw_score == pytest.approx(175.5)
    assert r.score == 176
    assert r.band == "C"


def test_worst_customer_matches_hand_calculation(card):
    r = run(card, {"max_delinquency_months": 5, "paid_amount_6m": 100})
    assert r.raw_score == pytest.approx(60.25 + 80.0)
    assert r.score == 140
    assert r.band == "C"  # 경계값 정확히 140 → C


# ---- 경계값 ---------------------------------------------------------------
@pytest.mark.parametrize(
    "value, expected_points",
    [(-100, 110.0), (0, 110.0), (0.0001, 95.5), (1, 95.5), (1.0001, 60.25), (999, 60.25)],
)
def test_bin_boundary_is_lower_exclusive_upper_inclusive(card, value, expected_points):
    r = run(card, {"max_delinquency_months": value, "paid_amount_6m": 6000})
    assert r.variables["max_delinquency_months"].points == expected_points


def test_values_outside_training_range_fall_into_outer_bins(card):
    low = run(card, {"max_delinquency_months": -3, "paid_amount_6m": -10})
    assert low.raw_score == pytest.approx(110.0 + 80.0)
    assert low.band == "B"

    high = run(card, {"max_delinquency_months": 99, "paid_amount_6m": 1e9})
    assert high.raw_score == pytest.approx(60.25 + 120.0)
    assert high.score == 180  # 180.25 → 180
    assert high.band == "B"  # 반올림된 점수가 경계 180 에 닿는다


# ---- 반올림 ----------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [(140.4, 140), (140.5, 141), (141.5, 142), (180.25, 180), (0.5, 1), (-0.5, 0), (230.0, 230)],
)
def test_final_score_rounds_half_up(raw, expected):
    assert round_score(raw) == expected


def test_result_keeps_unrounded_score_and_unrounded_points(card):
    r = run(card, {"max_delinquency_months": 1, "paid_amount_6m": 5000.5})
    assert r.raw_score == pytest.approx(175.5)
    assert r.variables["max_delinquency_months"].points == 95.5


# ---- 감점 사유 -------------------------------------------------------------
def test_top_negative_is_ordered_by_points_lost(card):
    r = run(card, {"max_delinquency_months": 1, "paid_amount_6m": 5000.5})
    # paid: 120 - 80 = 40, delinquency: 110 - 95.5 = 14.5
    assert [x.variable for x in r.top_negative] == ["paid_amount_6m", "max_delinquency_months"]
    assert [x.points_lost for x in r.top_negative] == pytest.approx([40.0, 14.5])


def test_top_negative_drops_variables_with_zero_loss(card):
    r = run(card, {"max_delinquency_months": 0, "paid_amount_6m": 100})
    assert [x.variable for x in r.top_negative] == ["paid_amount_6m"]


def test_top_negative_is_limited_to_three_and_keeps_the_biggest():
    card = parse_scorecard(four_variable_card_dict())
    r = run(card, {
        "max_delinquency_months": 5,   # 110 - 60.25 = 49.75
        "paid_amount_6m": 100,         # 120 - 80    = 40
        "avg_billed_amount_6m": 0,     #  50 - 20    = 30
        "active_months": 0,            #  15 - 5     = 10
    })
    assert [x.variable for x in r.top_negative] == [
        "max_delinquency_months", "paid_amount_6m", "avg_billed_amount_6m",
    ]


def test_top_negative_ties_keep_card_variable_order():
    d = card_dict()
    d["variables"]["paid_amount_6m"]["bins"] = [_bin(None, 5000.5, 80.0), _bin(5000.5, None, 110.0)]
    d["variables"]["max_delinquency_months"]["bins"] = [_bin(None, 0, 110.0), _bin(0, 1, 100.0), _bin(1, None, 80.0)]
    r = run(parse_scorecard(d), {"max_delinquency_months": 5, "paid_amount_6m": 100})
    assert [x.points_lost for x in r.top_negative] == [30.0, 30.0]
    assert [x.variable for x in r.top_negative] == ["max_delinquency_months", "paid_amount_6m"]


def test_reason_describes_value_bin_and_points(card):
    r = run(card, {"max_delinquency_months": 1, "paid_amount_6m": 5000.5})
    reason = r.top_negative[0]
    assert reason.variable == "paid_amount_6m"
    assert reason.value == 5000.5
    assert reason.bin == "(-inf, 5000.5]"
    assert reason.points == 80.0
    assert reason.max_points == 120.0


# ---- 등급 ------------------------------------------------------------------
@pytest.mark.parametrize(
    "delinq, paid, expected_score, expected_band",
    [
        (0, 6000, 230, "A"),
        (0, 5000.5, 190, "B"),   # 110 + 80
        (1, 6000, 216, "B"),     # 95.5 + 120 = 215.5 → 216
        (5, 6000, 180, "B"),     # 60.25 + 120 = 180.25 → 180, 경계값 정확히 B
        (5, 100, 140, "C"),      # 140.25 → 140, 경계값 정확히 C
    ],
)
def test_band_is_assigned_by_the_cards_boundaries(card, delinq, paid, expected_score, expected_band):
    r = run(card, {"max_delinquency_months": delinq, "paid_amount_6m": paid})
    assert (r.score, r.band) == (expected_score, expected_band)


def test_score_just_below_a_boundary_gets_the_next_band():
    d = card_dict()
    d["bands"] = [{"name": "A", "min_score": 217}, {"name": "B", "min_score": 100}]
    r = run(parse_scorecard(d), {"max_delinquency_months": 1, "paid_amount_6m": 6000})  # 216
    assert (r.score, r.band) == (216, "B")


def test_score_below_every_boundary_gets_the_lowest_band():
    d = card_dict()
    d["bands"] = [{"name": "A", "min_score": 220}, {"name": "B", "min_score": 200}]
    r = run(parse_scorecard(d), {"max_delinquency_months": 5, "paid_amount_6m": 100})  # 140
    assert r.band == "B"


# ---- 응답 구성 -------------------------------------------------------------
def test_response_contains_the_required_fields(card):
    out = run(card, {"max_delinquency_months": 1, "paid_amount_6m": 5000.5}).to_dict()
    assert out["score"] == 176
    assert out["band"] == "C"
    assert out["model_version"] == "fixture-v1"
    assert [x["variable"] for x in out["top_negative"]] == ["paid_amount_6m", "max_delinquency_months"]
    assert out["variables"]["paid_amount_6m"]["points"] == 80.0
    assert out["variables"]["paid_amount_6m"]["max_points"] == 120.0
    assert out["variables"]["max_delinquency_months"]["bin"] == "(0, 1]"
    assert set(out["variables"]) == {"max_delinquency_months", "paid_amount_6m"}


def test_response_is_json_serializable(card):
    json.dumps(run(card, {"max_delinquency_months": 1, "paid_amount_6m": 5000.5}).to_dict())


# ---- 입력 검증 -------------------------------------------------------------
def test_missing_variable_raises_and_is_not_filled_with_a_default(card):
    with pytest.raises(MissingFeatureError, match="paid_amount_6m"):
        run(card, {"max_delinquency_months": 0})


def test_all_missing_variables_are_named(card):
    with pytest.raises(MissingFeatureError) as exc:
        run(card, {})
    assert "max_delinquency_months" in str(exc.value) and "paid_amount_6m" in str(exc.value)


def test_unknown_variable_raises(card):
    with pytest.raises(UnknownFeatureError, match="AGE"):
        run(card, {"max_delinquency_months": 0, "paid_amount_6m": 6000, "AGE": 30})


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "5", True, [1]])
def test_non_numeric_value_raises(card, bad):
    with pytest.raises(InvalidFeatureError, match="paid_amount_6m"):
        run(card, {"max_delinquency_months": 0, "paid_amount_6m": bad})


def test_feature_errors_are_scorecard_errors():
    for exc in (MissingFeatureError, UnknownFeatureError, InvalidFeatureError):
        assert issubclass(exc, ScoreCardError)


def test_currency_mismatch_is_rejected_before_scoring(card):
    with pytest.raises(CurrencyMismatchError, match="KRW"):
        run(card, {"max_delinquency_months": 0, "paid_amount_6m": 6000}, currency="KRW")


def test_input_currency_is_required():
    card = parse_scorecard(card_dict())
    with pytest.raises(TypeError):
        score_customer(card, {"max_delinquency_months": 0, "paid_amount_6m": 6000})


def test_scoring_does_not_mutate_the_input_or_the_card(card):
    features = {"max_delinquency_months": 1, "paid_amount_6m": 5000.5}
    before_features, before_card = copy.deepcopy(features), card.to_dict()
    run(card, features)
    assert features == before_features
    assert card.to_dict() == before_card

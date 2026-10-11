"""Phase 5: 런타임 스코어카드 로드·검증 (training 을 import 하지 않는 순수 JSON 검증)."""
import copy
import json

import pytest

from app.domain.scoring.scorecard.card import (
    FORBIDDEN_VARIABLES,
    CurrencyMismatchError,
    ScoreCardError,
    ScoreCardPurposeError,
    load_scorecard,
    parse_scorecard,
)


def valid_card():
    return {
        "model_version": "test-v1",
        "purpose": "pipeline-validation",
        "data_source": "unit test",
        "currency": "TWD",
        "created_at": "2026-10-11",
        "scale": {"base_score": 600, "base_good_odds": 50, "points_per_double": 20},
        "intercept": -1.23,
        "variables": {
            "max_delinquency_months": {
                "beta": -0.87,
                "bins": [
                    {"lower": None, "upper": 0, "bad_rate": 0.12, "woe": 0.62, "points": 41.5},
                    {"lower": 0, "upper": 1, "bad_rate": 0.30, "woe": -0.41, "points": 15.25},
                    {"lower": 1, "upper": None, "bad_rate": 0.65, "woe": -1.8, "points": -20.0},
                ],
            },
            "paid_amount_6m": {
                "beta": -0.64,
                "bins": [
                    {"lower": None, "upper": 5000.5, "bad_rate": 0.3, "woe": -0.5, "points": 10.0},
                    {"lower": 5000.5, "upper": None, "bad_rate": 0.1, "woe": 0.7, "points": 30.0},
                ],
            },
        },
        "bands": [{"name": "A", "min_score": 650}, {"name": "B", "min_score": 620}],
    }


def _bins(card, var="max_delinquency_months"):
    return card["variables"][var]["bins"]


# ---- 정상 로드 / 왕복 ------------------------------------------------------
def test_valid_card_parses():
    card = parse_scorecard(valid_card())
    assert card.model_version == "test-v1"
    assert card.purpose == "pipeline-validation"
    assert card.currency == "TWD"
    assert card.scale.base_score == 600
    assert list(card.variables) == ["max_delinquency_months", "paid_amount_6m"]
    assert card.variables["max_delinquency_months"].beta == -0.87
    assert card.variables["max_delinquency_months"].bins[0].lower is None
    assert [b.name for b in card.bands] == ["A", "B"]


def test_to_dict_roundtrips_exactly():
    data = valid_card()
    assert parse_scorecard(data).to_dict() == data


def test_optional_bands_status_roundtrips():
    data = valid_card()
    data["bands_status"] = "draft"
    card = parse_scorecard(data)
    assert card.bands_status == "draft"
    assert card.to_dict() == data


def test_load_from_file(tmp_path):
    path = tmp_path / "card.json"
    path.write_text(json.dumps(valid_card(), ensure_ascii=False), encoding="utf-8")
    assert load_scorecard(path).model_version == "test-v1"


def test_load_rejects_invalid_json(tmp_path):
    path = tmp_path / "card.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ScoreCardError):
        load_scorecard(path)


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(ScoreCardError):
        load_scorecard(tmp_path / "nope.json")


# ---- 필수 필드 -------------------------------------------------------------
@pytest.mark.parametrize(
    "field",
    ["model_version", "purpose", "data_source", "currency", "scale", "intercept", "variables", "bands"],
)
def test_missing_required_field_fails(field):
    data = valid_card()
    del data[field]
    with pytest.raises(ScoreCardError, match=field):
        parse_scorecard(data)


@pytest.mark.parametrize("field", ["base_score", "base_good_odds", "points_per_double"])
def test_missing_scale_field_fails(field):
    data = valid_card()
    del data["scale"][field]
    with pytest.raises(ScoreCardError, match=field):
        parse_scorecard(data)


def test_unknown_top_level_field_fails():
    data = valid_card()
    data["surprise"] = 1
    with pytest.raises(ScoreCardError, match="surprise"):
        parse_scorecard(data)


@pytest.mark.parametrize("field", ["model_version", "purpose", "data_source", "currency"])
def test_empty_string_field_fails(field):
    data = valid_card()
    data[field] = ""
    with pytest.raises(ScoreCardError, match=field):
        parse_scorecard(data)


def test_non_numeric_intercept_fails():
    data = valid_card()
    data["intercept"] = "abc"
    with pytest.raises(ScoreCardError, match="intercept"):
        parse_scorecard(data)


def test_empty_variables_fails():
    data = valid_card()
    data["variables"] = {}
    with pytest.raises(ScoreCardError, match="variables"):
        parse_scorecard(data)


# ---- 구간 경계 -------------------------------------------------------------
def test_overlapping_bins_fail():
    data = valid_card()
    _bins(data)[1]["upper"] = 2  # 다음 구간 lower 는 1 → 겹침
    with pytest.raises(ScoreCardError, match="겹"):
        parse_scorecard(data)


def test_gap_between_bins_fails():
    data = valid_card()
    _bins(data)[2]["lower"] = 3  # 이전 upper 는 1 → 빈 구간
    with pytest.raises(ScoreCardError, match="빈"):
        parse_scorecard(data)


def test_first_bin_must_be_open_below():
    data = valid_card()
    _bins(data)[0]["lower"] = -5
    with pytest.raises(ScoreCardError, match="lower"):
        parse_scorecard(data)


def test_last_bin_must_be_open_above():
    data = valid_card()
    _bins(data)[-1]["upper"] = 9
    with pytest.raises(ScoreCardError, match="upper"):
        parse_scorecard(data)


def test_bin_with_lower_not_below_upper_fails():
    data = valid_card()
    _bins(data)[1]["lower"] = 1
    _bins(data)[1]["upper"] = 1
    _bins(data)[0]["upper"] = 1
    _bins(data)[2]["lower"] = 1
    with pytest.raises(ScoreCardError, match="lower"):
        parse_scorecard(data)


def test_empty_bins_fail():
    data = valid_card()
    data["variables"]["max_delinquency_months"]["bins"] = []
    with pytest.raises(ScoreCardError, match="bins"):
        parse_scorecard(data)


def test_single_open_bin_is_allowed():
    data = valid_card()
    data["variables"]["paid_amount_6m"]["bins"] = [
        {"lower": None, "upper": None, "bad_rate": 0.2, "woe": 0.0, "points": 20.0}
    ]
    assert len(parse_scorecard(data).variables["paid_amount_6m"].bins) == 1


@pytest.mark.parametrize("field", ["bad_rate", "woe", "points"])
def test_bin_numbers_must_be_finite(field):
    data = valid_card()
    _bins(data)[0][field] = float("nan")
    with pytest.raises(ScoreCardError, match=field):
        parse_scorecard(data)


def test_bad_rate_must_be_a_probability():
    data = valid_card()
    _bins(data)[0]["bad_rate"] = 1.5
    with pytest.raises(ScoreCardError, match="bad_rate"):
        parse_scorecard(data)


def test_beta_must_be_negative():
    data = valid_card()
    data["variables"]["paid_amount_6m"]["beta"] = 0.1
    with pytest.raises(ScoreCardError, match="beta"):
        parse_scorecard(data)


# ---- 금지 변수 -------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "SEX", "EDUCATION", "MARRIAGE", "AGE", "LIMIT_BAL",
        "age", "Sex",
        "extension_this_month", "auto_extension_this_month", "auto_extension_cnt_12m",
        "my_extension_feature", "자동연장_횟수",
    ],
)
def test_forbidden_variable_fails(name):
    data = valid_card()
    data["variables"][name] = copy.deepcopy(data["variables"]["paid_amount_6m"])
    with pytest.raises(ScoreCardError, match="금지"):
        parse_scorecard(data)


def test_forbidden_list_covers_the_training_exclusions():
    """런타임은 training 을 import 하지 않으므로 상수가 어긋나지 않는지 테스트로 묶는다."""
    from training.scorecard.preprocess import FORBIDDEN_COLUMNS

    assert {c.lower() for c in FORBIDDEN_COLUMNS} <= FORBIDDEN_VARIABLES


# ---- purpose / currency ----------------------------------------------------
def test_non_production_card_cannot_load_in_production_mode():
    with pytest.raises(ScoreCardPurposeError, match="pipeline-validation"):
        parse_scorecard(valid_card(), production=True)


def test_non_production_card_loads_in_default_mode():
    assert parse_scorecard(valid_card()).purpose == "pipeline-validation"


def test_production_card_loads_in_production_mode():
    data = valid_card()
    data["purpose"] = "production"
    data["currency"] = "KRW"
    assert parse_scorecard(data, production=True).purpose == "production"


def test_production_mode_applies_to_file_loading(tmp_path):
    path = tmp_path / "card.json"
    path.write_text(json.dumps(valid_card()), encoding="utf-8")
    with pytest.raises(ScoreCardPurposeError):
        load_scorecard(path, production=True)


def test_currency_mismatch_raises():
    card = parse_scorecard(valid_card())  # TWD
    with pytest.raises(CurrencyMismatchError, match="KRW"):
        card.assert_currency("KRW")


def test_same_currency_passes():
    parse_scorecard(valid_card()).assert_currency("TWD")


def test_errors_share_a_base_class():
    assert issubclass(ScoreCardPurposeError, ScoreCardError)
    assert issubclass(CurrencyMismatchError, ScoreCardError)
    assert issubclass(ScoreCardError, ValueError)


# ---- 등급 ------------------------------------------------------------------
def test_bands_must_be_strictly_descending():
    data = valid_card()
    data["bands"] = [{"name": "A", "min_score": 620}, {"name": "B", "min_score": 650}]
    with pytest.raises(ScoreCardError, match="bands"):
        parse_scorecard(data)


def test_band_names_must_be_unique():
    data = valid_card()
    data["bands"] = [{"name": "A", "min_score": 650}, {"name": "A", "min_score": 620}]
    with pytest.raises(ScoreCardError, match="bands"):
        parse_scorecard(data)


def test_empty_bands_fail():
    data = valid_card()
    data["bands"] = []
    with pytest.raises(ScoreCardError, match="bands"):
        parse_scorecard(data)

"""Phase 4: 오즈 → 점수, 배점표.

점수 = Offset + Factor · ln(정상 오즈)
기준: 정상 오즈 50 일 때 600점, 정상 오즈가 2배가 될 때마다 +20점.
"""
import math

import numpy as np
import pandas as pd
import pytest

from training.scorecard.binning import Bin, BinTable, apply_woe, fit_woe_tables
from training.scorecard.fit import FitResult, fit_logit
from training.scorecard.preprocess import LABEL
from training.scorecard.scale import (
    FACTOR,
    OFFSET,
    build_points,
    score_frame,
    score_from_good_odds,
)


def test_factor_and_offset_match_the_spec():
    assert FACTOR == pytest.approx(28.8539, abs=1e-4)
    assert FACTOR == pytest.approx(20 / math.log(2))
    assert OFFSET == pytest.approx(487.1229, abs=1e-4)
    assert OFFSET == pytest.approx(600 - FACTOR * math.log(50))


@pytest.mark.parametrize(
    "good_odds, expected",
    [(50, 600.0), (100, 620.0), (25, 580.0), (70, 609.7)],
)
def test_good_odds_to_score(good_odds, expected):
    assert score_from_good_odds(good_odds) == pytest.approx(expected, abs=0.1)


@pytest.mark.parametrize("odds", [1.5, 8, 50, 333])
def test_doubling_good_odds_adds_exactly_twenty_points(odds):
    assert score_from_good_odds(2 * odds) - score_from_good_odds(odds) == pytest.approx(20.0)


def test_score_rejects_non_positive_odds():
    with pytest.raises(ValueError, match="오즈"):
        score_from_good_odds(0)


# ---- 손 계산 가능한 작은 예 -------------------------------------------------
def _hand_tables():
    # a: 정상 80/부실 20 vs 정상 20/부실 80 → WoE = ±ln 4
    a = BinTable("a", "discrete", "increasing", (Bin(None, 0, 80, 20), Bin(0, None, 20, 80)))
    # b: 정상 60/부실 30 vs 정상 40/부실 70 (전체 정상 100, 부실 100)
    b = BinTable("b", "discrete", "increasing", (Bin(None, 0, 60, 30), Bin(0, None, 40, 70)))
    return {"a": a, "b": b}


def test_points_match_hand_calculation():
    tables = _hand_tables()
    fit = FitResult(intercept=-1.0, betas={"a": -2.0, "b": -1.0}, bse={"a": 0.1, "b": 0.1})
    points = build_points(fit, tables)

    n = 2
    woe_a = [math.log((80 / 100) / (20 / 100)), math.log((20 / 100) / (80 / 100))]
    woe_b = [math.log((60 / 100) / (30 / 100)), math.log((40 / 100) / (70 / 100))]
    expect_a = [OFFSET / n - FACTOR * (-1.0 / n + -2.0 * w) for w in woe_a]
    expect_b = [OFFSET / n - FACTOR * (-1.0 / n + -1.0 * w) for w in woe_b]
    assert points["a"] == pytest.approx(expect_a)
    assert points["b"] == pytest.approx(expect_b)


def test_points_within_a_variable_increase_with_woe():
    tables = _hand_tables()
    fit = FitResult(intercept=-1.0, betas={"a": -2.0, "b": -1.0}, bse={"a": 0.1, "b": 0.1})
    points = build_points(fit, tables)
    for name, table in tables.items():
        order = np.argsort(table.woes)
        sorted_points = [points[name][i] for i in order]
        assert all(x < y for x, y in zip(sorted_points, sorted_points[1:]))


def test_build_points_rejects_non_negative_beta():
    fit = FitResult(intercept=0.0, betas={"a": 0.5, "b": -1.0}, bse={"a": 0.1, "b": 0.1})
    with pytest.raises(ValueError, match="a"):
        build_points(fit, _hand_tables())


def test_build_points_rejects_variable_mismatch():
    fit = FitResult(intercept=0.0, betas={"a": -1.0}, bse={"a": 0.1})
    with pytest.raises(ValueError, match="변수"):
        build_points(fit, _hand_tables())


# ---- 파이프라인 전체: 합성 데이터 (동작 검증용) ------------------------------
KINDS = {"delinq": "discrete", "paid": "continuous"}


def _frame(seed, n=8000):
    rng = np.random.default_rng(seed)
    delinq = rng.choice([0, 1, 2, 3], size=n, p=[0.6, 0.2, 0.15, 0.05])
    paid = rng.gamma(2.0, 1000.0, n)
    logit = -2.0 + 0.9 * delinq - 0.0004 * paid
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"delinq": delinq, "paid": paid, LABEL: y})


@pytest.fixture(scope="module")
def pipeline():
    train, valid = _frame(1), _frame(2, n=1500)
    tables = fit_woe_tables(train, KINDS)
    fit = fit_logit(apply_woe(train, tables), train[LABEL])
    return tables, fit, build_points(fit, tables), valid


def test_sum_of_points_equals_offset_minus_factor_times_log_bad_odds(pipeline):
    tables, fit, points, valid = pipeline
    woe = apply_woe(valid, tables)
    expected = OFFSET - FACTOR * (fit.intercept + sum(b * woe[v] for v, b in fit.betas.items()))
    actual = score_frame(valid, tables, points)
    assert actual.to_numpy() == pytest.approx(expected.to_numpy(), abs=0.01)


def test_score_equals_score_from_predicted_good_odds(pipeline):
    tables, fit, points, valid = pipeline
    woe = apply_woe(valid, tables)
    good_odds = 1 / fit.bad_odds(woe)
    expected = good_odds.map(score_from_good_odds)
    actual = score_frame(valid, tables, points)
    assert actual.to_numpy() == pytest.approx(expected.to_numpy(), abs=0.01)


def test_safer_customers_get_higher_scores(pipeline):
    tables, _, points, _ = pipeline
    safe = pd.DataFrame({"delinq": [0], "paid": [9000.0]})
    risky = pd.DataFrame({"delinq": [3], "paid": [10.0]})
    assert score_frame(safe, tables, points).iloc[0] > score_frame(risky, tables, points).iloc[0]


def test_points_are_not_rounded_here(pipeline):
    # 반올림은 Phase 5 내보내기에서 정한다. 여기서는 합이 정확히 맞아야 한다.
    _, _, points, _ = pipeline
    assert any(p != round(p) for ps in points.values() for p in ps)

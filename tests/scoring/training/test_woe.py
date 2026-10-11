"""Phase 2: 부실비와 WoE 공식 (손 계산 대조)."""
import math

import numpy as np
import pytest

from training.scorecard.woe import bad_rate, woe, woe_by_odds

# 정상 1000, 부실 1000 인 모집단에서 구간별 비율을 지시서 표대로 맞춘 것
GOOD_TOTAL = 1000
BAD_TOTAL = 1000
BINS = {
    "A": (100, 300),  # 정상 10%, 부실 30%
    "B": (400, 450),  # 정상 40%, 부실 45%
    "C": (500, 250),  # 정상 50%, 부실 25%
}


def test_bad_rate_nine_good_one_bad_is_ten_percent():
    assert bad_rate(good=9, bad=1) == pytest.approx(0.1)


@pytest.mark.parametrize(
    "name, expected",
    [("A", -1.0986), ("B", -0.1178), ("C", 0.6931)],
)
def test_woe_matches_hand_calculation(name, expected):
    good, bad = BINS[name]
    assert woe(good, bad, GOOD_TOTAL, BAD_TOTAL) == pytest.approx(expected, abs=1e-4)


def test_woe_sign_positive_means_safer_than_average():
    good, bad = BINS["C"]
    assert woe(good, bad, GOOD_TOTAL, BAD_TOTAL) > 0
    good, bad = BINS["A"]
    assert woe(good, bad, GOOD_TOTAL, BAD_TOTAL) < 0


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_woe_equals_log_of_bin_odds_over_overall_odds(name):
    good, bad = BINS[name]
    assert woe(good, bad, GOOD_TOTAL, BAD_TOTAL) == pytest.approx(
        woe_by_odds(good, bad, GOOD_TOTAL, BAD_TOTAL)
    )


def test_equivalence_holds_with_unbalanced_totals():
    # 정상 700 / 부실 300, 구간: 정상 70 / 부실 60
    assert woe(70, 60, 700, 300) == pytest.approx(math.log((70 / 60) / (700 / 300)))
    assert woe(70, 60, 700, 300) == pytest.approx(woe_by_odds(70, 60, 700, 300))


def test_zero_bad_bin_gets_finite_woe_with_half_count_correction():
    value = woe(good=50, bad=0, good_total=GOOD_TOTAL, bad_total=BAD_TOTAL)
    assert np.isfinite(value)
    assert value > 0
    # 두 값 모두에 0.5 를 더해서 계산한다
    assert value == pytest.approx(math.log((50.5 / GOOD_TOTAL) / (0.5 / BAD_TOTAL)))


def test_zero_good_bin_gets_finite_negative_woe():
    value = woe(good=0, bad=40, good_total=GOOD_TOTAL, bad_total=BAD_TOTAL)
    assert np.isfinite(value)
    assert value < 0
    assert value == pytest.approx(math.log((0.5 / GOOD_TOTAL) / (40.5 / BAD_TOTAL)))


def test_correction_applies_only_to_empty_bins():
    # 0건이 아니면 보정 없이 순수 공식
    assert woe(10, 20, 100, 100) == pytest.approx(math.log((10 / 100) / (20 / 100)))

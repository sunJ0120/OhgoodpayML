"""Phase 2: 구간화, 단조 병합, 최소 구간 크기, 누수 방지."""
import numpy as np
import pandas as pd
import pytest

from training.scorecard.binning import (
    MIN_BIN_SHARE,
    Bin,
    apply_woe,
    assign_bins,
    count_bins,
    fit_bins,
    fit_woe_tables,
    initial_uppers,
    merge_monotone,
    merge_small,
)
from training.scorecard.preprocess import LABEL
from training.scorecard.split import stratified_split


def _bins_from_rates(bad_rates, size=100):
    """구간 크기가 같고 부실비가 주어진 값인 Bin 목록 (lower/upper 는 순번)."""
    out = []
    for i, r in enumerate(bad_rates):
        bad = round(size * r)
        out.append(Bin(lower=None if i == 0 else float(i - 1), upper=None if i == len(bad_rates) - 1 else float(i),
                       good=size - bad, bad=bad))
    return out


def _rates(bins):
    return [b.bad / (b.good + b.bad) for b in bins]


# ---- 단조 병합 -------------------------------------------------------------
def test_monotone_merge_pools_the_violating_pair():
    bins = _bins_from_rates([0.10, 0.25, 0.20, 0.40])
    merged = merge_monotone(bins, "increasing")
    assert _rates(merged) == pytest.approx([0.10, 0.225, 0.40])
    assert merged[1].good + merged[1].bad == 200
    # 경계는 합쳐진 두 구간의 바깥 경계를 따른다
    assert merged[1].lower == bins[1].lower and merged[1].upper == bins[2].upper


def test_monotone_merge_leaves_already_monotone_bins_alone():
    bins = _bins_from_rates([0.10, 0.20, 0.40])
    assert merge_monotone(bins, "increasing") == bins


def test_monotone_merge_decreasing_direction():
    bins = _bins_from_rates([0.40, 0.20, 0.25, 0.10])
    merged = merge_monotone(bins, "decreasing")
    assert _rates(merged) == pytest.approx([0.40, 0.225, 0.10])


def test_monotone_merge_merges_equal_neighbors_to_keep_strict_order():
    merged = merge_monotone(_bins_from_rates([0.10, 0.10, 0.30]), "increasing")
    assert _rates(merged) == pytest.approx([0.10, 0.30])


def test_monotone_merge_can_collapse_to_one_bin():
    merged = merge_monotone(_bins_from_rates([0.30, 0.10, 0.20, 0.05]), "increasing")
    rates = _rates(merged)
    assert all(a < b for a, b in zip(rates, rates[1:]))


# ---- 최소 구간 크기 --------------------------------------------------------
def test_min_bin_share_is_five_percent():
    assert MIN_BIN_SHARE == 0.05


def test_small_bin_is_merged_into_neighbor_with_closest_bad_rate():
    bins = [
        Bin(None, 0.0, good=480, bad=20),    # 부실비 .04, 50%
        Bin(0.0, 1.0, good=18, bad=2),       # 부실비 .10, 2% → 작음
        Bin(1.0, None, good=380, bad=120),   # 부실비 .24, 50%
    ]
    merged = merge_small(bins, MIN_BIN_SHARE)
    assert len(merged) == 2
    # .10 은 .04 쪽이 더 가깝다
    assert (merged[0].good, merged[0].bad) == (498, 22)
    assert merged[0].upper == 1.0
    assert (merged[1].good, merged[1].bad) == (380, 120)


def test_small_first_bin_can_only_merge_to_the_right():
    bins = [Bin(None, 0.0, 9, 1), Bin(0.0, 1.0, 490, 10), Bin(1.0, None, 400, 90)]
    merged = merge_small(bins, MIN_BIN_SHARE)
    assert len(merged) == 2
    assert merged[0].lower is None and merged[0].upper == 1.0


def test_merge_small_keeps_monotone_order():
    bins = _bins_from_rates([0.05, 0.15, 0.30, 0.50], size=100)
    bins[1] = Bin(bins[1].lower, bins[1].upper, good=13, bad=2)  # 15/315 = 4.8% 로 작은 구간, 부실비 .13
    assert bins[1].n / sum(b.n for b in bins) < MIN_BIN_SHARE
    merged = merge_small(bins, MIN_BIN_SHARE)
    rates = _rates(merged)
    assert all(a < b for a, b in zip(rates, rates[1:]))


# ---- 초기 구간과 할당 ------------------------------------------------------
def test_discrete_initial_uppers_are_observed_values_except_the_max():
    x = pd.Series([0, 0, 1, 1, 2, 5, 5])
    assert initial_uppers(x, "discrete") == [0, 1, 2]


def test_continuous_initial_uppers_are_quantiles_with_at_most_five_bins():
    x = pd.Series(np.arange(1, 101))
    uppers = initial_uppers(x, "continuous")
    assert len(uppers) <= 4
    assert uppers == sorted(set(uppers))
    assert all(u < x.max() for u in uppers)


def test_continuous_initial_uppers_collapse_when_many_ties():
    x = pd.Series([0] * 70 + list(range(1, 31)))
    uppers = initial_uppers(x, "continuous")
    assert uppers == sorted(set(uppers))
    assert uppers[0] == 0


def test_boundary_rule_is_lower_exclusive_upper_inclusive():
    bins = [Bin(None, 0.0, 1, 1), Bin(0.0, 2.0, 1, 1), Bin(2.0, None, 1, 1)]
    idx = assign_bins(pd.Series([-5, 0, 0.1, 2, 2.1, 99]), bins)
    assert idx.tolist() == [0, 0, 1, 1, 2, 2]


def test_count_bins_counts_good_and_bad_per_bin():
    x = pd.Series([0, 0, 1, 1, 2, 2])
    y = pd.Series([0, 1, 0, 0, 1, 1])
    bins = count_bins(x, y, [0, 1])
    assert [(b.good, b.bad) for b in bins] == [(1, 1), (2, 0), (0, 2)]
    assert [b.lower for b in bins] == [None, 0, 1]
    assert [b.upper for b in bins] == [0, 1, None]


# ---- fit_bins 전체 ---------------------------------------------------------
def _synthetic(n=4000, seed=7):
    rng = np.random.default_rng(seed)
    delinq = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.55, 0.2, 0.15, 0.07, 0.03])
    p_bad = np.array([0.08, 0.30, 0.60, 0.75, 0.85])[delinq]
    y = (rng.random(n) < p_bad).astype(int)
    return pd.Series(delinq), pd.Series(y)


def test_fit_bins_returns_monotone_bins_with_min_size_and_direction():
    x, y = _synthetic()
    table = fit_bins(x, y, "discrete")
    rates = [b.bad_rate for b in table.bins]
    assert table.direction == "increasing"
    assert all(a < b for a, b in zip(rates, rates[1:]))
    total = sum(b.n for b in table.bins)
    assert total == len(x)
    assert all(b.n / total >= MIN_BIN_SHARE for b in table.bins)
    assert table.bins[0].lower is None and table.bins[-1].upper is None


def test_woe_is_larger_for_safer_bins():
    x, y = _synthetic()
    table = fit_bins(x, y, "discrete")
    woes = table.woes
    assert all(a > b for a, b in zip(woes, woes[1:]))  # 부실비 ↑ → WoE ↓


def test_direction_is_detected_for_decreasing_relationship():
    rng = np.random.default_rng(3)
    x = pd.Series(rng.integers(0, 5, 3000))
    y = pd.Series((rng.random(3000) < (0.8 - 0.15 * x.to_numpy())).astype(int))
    table = fit_bins(x, y, "discrete")
    assert table.direction == "decreasing"
    rates = [b.bad_rate for b in table.bins]
    assert all(a > b for a, b in zip(rates, rates[1:]))


def test_values_outside_training_range_fall_into_outer_bins():
    x, y = _synthetic()
    table = fit_bins(x, y, "discrete")
    woe_values = table.transform_woe(pd.Series([-100, 100]))
    assert woe_values.iloc[0] == pytest.approx(table.woes[0])
    assert woe_values.iloc[1] == pytest.approx(table.woes[-1])


def test_fit_bins_rejects_unknown_kind():
    x, y = _synthetic()
    with pytest.raises(ValueError, match="kind"):
        fit_bins(x, y, "weird")


# ---- 누수 방지 -------------------------------------------------------------
def _frame(seed, n=3000):
    rng = np.random.default_rng(seed)
    delinq = rng.choice([0, 1, 2, 3], size=n, p=[0.6, 0.2, 0.15, 0.05])
    paid = rng.gamma(2.0, 1000.0, n) * (1 + (delinq == 0))
    y = (rng.random(n) < np.array([0.08, 0.3, 0.6, 0.8])[delinq]).astype(int)
    return pd.DataFrame({"max_delinquency_months": delinq, "paid_amount_6m": paid, LABEL: y})


KINDS = {"max_delinquency_months": "discrete", "paid_amount_6m": "continuous"}


def test_tables_are_fitted_on_train_only_and_only_applied_to_validation():
    train, valid = _frame(1), _frame(2)
    tables = fit_woe_tables(train, KINDS)

    # 검증 데이터가 완전히 달라져도(라벨을 뒤집어도) 구간과 WoE 는 그대로다
    flipped = valid.copy()
    flipped[LABEL] = 1 - flipped[LABEL]
    for df in (valid, flipped):
        out = apply_woe(df, tables)
        for col, table in tables.items():
            expected = table.transform_woe(df[col])
            pd.testing.assert_series_equal(out[col], expected, check_names=False)

    again = fit_woe_tables(train, KINDS)
    for col in tables:
        assert tables[col] == again[col]


def test_apply_woe_output_has_one_column_per_variable_and_no_label():
    train = _frame(1)
    tables = fit_woe_tables(train, KINDS)
    out = apply_woe(train, tables)
    assert list(out.columns) == list(KINDS)
    assert not out.isna().any().any()


# ---- 분할 ------------------------------------------------------------------
def test_stratified_split_is_70_30_and_keeps_label_rate():
    df = _frame(5, n=2000)
    train, valid = stratified_split(df, LABEL, test_size=0.3, seed=42)
    assert len(train) + len(valid) == len(df)
    assert len(valid) == pytest.approx(600, abs=2)
    assert train[LABEL].mean() == pytest.approx(df[LABEL].mean(), abs=0.005)
    assert valid[LABEL].mean() == pytest.approx(df[LABEL].mean(), abs=0.005)
    assert set(train.index).isdisjoint(valid.index)


def test_stratified_split_is_deterministic_for_a_seed():
    df = _frame(5, n=500)
    a_train, _ = stratified_split(df, LABEL, 0.3, seed=42)
    b_train, _ = stratified_split(df, LABEL, 0.3, seed=42)
    c_train, _ = stratified_split(df, LABEL, 0.3, seed=43)
    assert a_train.index.tolist() == b_train.index.tolist()
    assert a_train.index.tolist() != c_train.index.tolist()

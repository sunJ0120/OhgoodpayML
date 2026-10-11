"""Phase 2: 실제 UCI 데이터로 구간표를 만든다. 네트워크 또는 로컬 캐시가 필요하다."""
import pytest

from training.scorecard.binning import (
    DEFAULT_KINDS,
    EXPECTED_DIRECTION,
    MIN_BIN_SHARE,
    direction_mismatches,
    fit_woe_tables,
)
from training.scorecard.features import FEATURE_COLUMNS, build_training_frame
from training.scorecard.load import load_uci
from training.scorecard.preprocess import LABEL
from training.scorecard.split import stratified_split

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def tables():
    train, _ = stratified_split(build_training_frame(load_uci()), LABEL, test_size=0.3, seed=42)
    return fit_woe_tables(train, DEFAULT_KINDS)


def test_every_variable_has_a_table(tables):
    assert list(tables) == FEATURE_COLUMNS
    assert set(DEFAULT_KINDS) == set(FEATURE_COLUMNS)
    assert set(EXPECTED_DIRECTION) == set(FEATURE_COLUMNS)


def test_all_bins_are_monotone_and_at_least_five_percent(tables):
    for name, table in tables.items():
        total = sum(b.n for b in table.bins)
        rates = [b.bad_rate for b in table.bins]
        steps = [b - a for a, b in zip(rates, rates[1:])]
        assert all(s > 0 for s in steps) or all(s < 0 for s in steps), name
        assert all(b.n / total >= MIN_BIN_SHARE for b in table.bins), name


def test_mismatch_report_is_a_list_of_variable_names(tables):
    mismatches = direction_mismatches(tables)
    assert set(mismatches) <= set(FEATURE_COLUMNS)

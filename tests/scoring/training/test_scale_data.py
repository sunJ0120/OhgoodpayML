"""Phase 4: 실제 UCI 데이터로 배점표를 만든다. 네트워크 또는 로컬 캐시가 필요하다."""
import pytest

from training.scorecard.binning import DEFAULT_KINDS, apply_woe, fit_woe_tables
from training.scorecard.features import FEATURE_COLUMNS, build_training_frame
from training.scorecard.fit import fit_logit
from training.scorecard.load import load_uci
from training.scorecard.preprocess import LABEL
from training.scorecard.scale import FACTOR, OFFSET, build_points, score_frame
from training.scorecard.split import stratified_split

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def built():
    train, valid = stratified_split(build_training_frame(load_uci()), LABEL, test_size=0.3, seed=42)
    tables = fit_woe_tables(train, DEFAULT_KINDS)
    fit = fit_logit(apply_woe(train, tables), train[LABEL])
    return tables, fit, build_points(fit, tables), train, valid


def test_points_cover_every_variable_and_bin(built):
    tables, _, points, _, _ = built
    assert list(points) == FEATURE_COLUMNS
    for name, table in tables.items():
        assert len(points[name]) == len(table.bins)


def test_sum_of_points_matches_log_odds_formula_on_real_rows(built):
    tables, fit, points, _, valid = built
    woe = apply_woe(valid, tables)
    expected = OFFSET - FACTOR * (fit.intercept + sum(b * woe[v] for v, b in fit.betas.items()))
    actual = score_frame(valid, tables, points)
    assert actual.to_numpy() == pytest.approx(expected.to_numpy(), abs=0.01)

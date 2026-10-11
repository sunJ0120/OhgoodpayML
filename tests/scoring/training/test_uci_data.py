"""Phase 1: 실제 UCI 데이터 확인. 네트워크 또는 로컬 캐시가 필요하다."""
import pytest

from training.scorecard.features import FEATURE_COLUMNS, build_training_frame
from training.scorecard.load import load_uci
from training.scorecard.preprocess import LABEL, STATUS_COLUMNS, repay_status_distribution

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def raw():
    return load_uci()


def test_shape_and_label(raw):
    assert len(raw) == 30_000
    assert set(raw[LABEL].unique()) == {0, 1}
    assert raw[LABEL].sum() == 6_636


def test_training_frame_has_no_missing_values(raw):
    t = build_training_frame(raw)
    assert list(t.columns) == FEATURE_COLUMNS + [LABEL]
    assert not t.isna().any().any()


def test_distribution_report_covers_all_status_columns(raw):
    dist = repay_status_distribution(raw)
    assert set(dist["column"]) == set(STATUS_COLUMNS)
    assert (dist.groupby("column")["count"].sum() == 30_000).all()

"""Phase 3: 실제 UCI 데이터로 회귀를 돌린다. 네트워크 또는 로컬 캐시가 필요하다."""
import pytest

from training.scorecard.binning import DEFAULT_KINDS, apply_woe, fit_woe_tables
from training.scorecard.features import FEATURE_COLUMNS, build_training_frame
from training.scorecard.fit import fit_logit
from training.scorecard.load import load_uci
from training.scorecard.preprocess import LABEL
from training.scorecard.split import stratified_split

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def train_woe():
    train, _ = stratified_split(build_training_frame(load_uci()), LABEL, test_size=0.3, seed=42)
    tables = fit_woe_tables(train, DEFAULT_KINDS)
    return apply_woe(train, tables), train[LABEL]


def test_fit_runs_on_real_data_and_covers_every_variable(train_woe):
    woe, y = train_woe
    result = fit_logit(woe, y, check_signs=False)
    assert list(result.betas) == FEATURE_COLUMNS


def test_all_betas_are_negative_on_real_data(train_woe):
    """WoE 가 클수록 안전하므로 모든 β 는 음수여야 한다 (지시서 1장)."""
    woe, y = train_woe
    fit_logit(woe, y)  # 양수 β 가 있으면 CoefficientSignError

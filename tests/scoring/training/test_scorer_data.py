"""Phase 6: 커밋된 UCI 카드로 실제 검증 데이터 전체를 런타임 채점해 학습 쪽 점수와 대조한다."""
import pytest

from app.domain.scoring.scorecard.card import load_scorecard
from app.domain.scoring.scorecard.scorer import round_score, score_customer
from training.scorecard.binning import DEFAULT_KINDS, apply_woe, fit_woe_tables
from training.scorecard.features import FEATURE_COLUMNS, build_training_frame
from training.scorecard.fit import fit_logit
from training.scorecard.load import load_uci
from training.scorecard.preprocess import LABEL
from training.scorecard.run_uci import UCI_CARD_PATH
from training.scorecard.scale import build_points, score_frame
from training.scorecard.split import stratified_split

pytestmark = pytest.mark.data


def test_runtime_scores_match_training_scores_on_all_validation_rows():
    train, valid = stratified_split(build_training_frame(load_uci()), LABEL, test_size=0.3, seed=42)
    tables = fit_woe_tables(train, DEFAULT_KINDS)
    fit = fit_logit(apply_woe(train, tables), train[LABEL])
    expected = score_frame(valid, tables, build_points(fit, tables))

    card = load_scorecard(UCI_CARD_PATH)
    mismatches = 0
    for (_, row), exp in zip(valid.iterrows(), expected):
        r = score_customer(card, {c: float(row[c]) for c in FEATURE_COLUMNS}, input_currency="TWD")
        if abs(r.raw_score - exp) > 1e-6 or r.score != round_score(exp):
            mismatches += 1
    assert mismatches == 0
    assert len(valid) == 9000

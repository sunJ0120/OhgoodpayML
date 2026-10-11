"""Phase 7: 커밋된 UCI 카드로 실제 검증 데이터(30%)를 채점해 점수 구간별 부실비를 만든다."""
import pytest

from app.domain.scoring.scorecard.card import load_scorecard
from training.scorecard.report import (
    band_table,
    build_validation_report,
    check_monotone,
    score_bucket_table,
    validation_scores,
)
from training.scorecard.run_uci import UCI_CARD_PATH

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def scored():
    return validation_scores()


def test_validation_set_is_the_nine_thousand_held_out_rows(scored):
    scores, y = scored
    assert len(scores) == len(y) == 9000
    assert scores.dtype.kind == "i"


def test_bucket_table_covers_every_validation_row(scored):
    scores, y = scored
    t = score_bucket_table(scores, y)
    assert t["n"].sum() == 9000
    assert t["bad"].sum() == y.sum()
    assert 2 <= len(t) <= 5


def test_card_bands_cover_every_validation_row(scored):
    scores, y = scored
    card = load_scorecard(UCI_CARD_PATH)
    t = band_table(scores, y, [{"name": b.name, "min_score": b.min_score} for b in card.bands])
    assert t["n"].sum() == 9000


def test_verdict_is_reported_not_asserted(scored):
    """단조 여부는 데이터가 정한다. 여기서는 판정이 계산되는지만 확인한다."""
    scores, y = scored
    verdict = check_monotone(score_bucket_table(scores, y))
    assert isinstance(verdict.is_monotone, bool)


def test_full_report_renders():
    md = build_validation_report()
    assert "점수 구간별 부실비" in md
    assert "초안" in md

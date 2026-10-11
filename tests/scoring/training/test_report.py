"""Phase 7: 점수 구간별 부실비 리포트 (손으로 만든 데이터)."""
import pandas as pd
import pytest

from training.scorecard.report import (
    band_table,
    build_report,
    check_monotone,
    draft_band_proposal,
    score_bucket_table,
)


def _scores_and_labels(bad_per_bucket, size=20):
    """점수 0..(5*size-1) 를 오름차순으로 5구간(size명씩)으로 나누고, 구간별로 주어진 수만큼 부실을 둔다."""
    scores, labels = [], []
    for b, bad in enumerate(bad_per_bucket):
        for i in range(size):
            scores.append(b * size + i)
            labels.append(1 if i < bad else 0)
    return pd.Series(scores), pd.Series(labels)


# ---- 구간 표 ---------------------------------------------------------------
def test_scores_split_into_five_equal_buckets_with_counts_and_bad_rates():
    scores, y = _scores_and_labels([16, 12, 8, 4, 0])
    t = score_bucket_table(scores, y)
    assert len(t) == 5
    assert t["n"].tolist() == [20] * 5
    assert t["bad"].tolist() == [16, 12, 8, 4, 0]
    assert t["bad_rate"].tolist() == pytest.approx([0.8, 0.6, 0.4, 0.2, 0.0])
    assert t["share"].tolist() == pytest.approx([0.2] * 5)


def test_buckets_are_ordered_from_lowest_score_to_highest():
    scores, y = _scores_and_labels([16, 12, 8, 4, 0])
    t = score_bucket_table(scores, y)
    assert t["bucket"].tolist() == [1, 2, 3, 4, 5]
    assert t["score_min"].tolist() == [0, 20, 40, 60, 80]
    assert t["score_max"].tolist() == [19, 39, 59, 79, 99]


def test_totals_are_preserved():
    scores, y = _scores_and_labels([16, 12, 8, 4, 0])
    t = score_bucket_table(scores, y)
    assert t["n"].sum() == len(scores)
    assert t["bad"].sum() == y.sum()


def test_heavy_ties_give_fewer_buckets_and_never_split_one_score():
    # 점수가 3종류뿐이다 → 5구간으로 나눌 수 없다
    scores = pd.Series([500] * 60 + [510] * 30 + [520] * 10)
    y = pd.Series([1] * 30 + [0] * 30 + [1] * 10 + [0] * 20 + [0] * 10)
    t = score_bucket_table(scores, y)
    assert 1 < len(t) < 5
    assert t["n"].sum() == 100
    # 구간끼리 점수 범위가 겹치지 않는다
    for upper_of_prev, lower_of_next in zip(t["score_max"], t["score_min"].iloc[1:]):
        assert upper_of_prev < lower_of_next


def test_all_identical_scores_cannot_be_split():
    with pytest.raises(ValueError, match="구간"):
        score_bucket_table(pd.Series([500] * 50), pd.Series([0, 1] * 25))


@pytest.mark.parametrize("labels", [[0, 1, 2, 0, 1], [0, 1, None, 0, 1]])
def test_labels_must_be_zero_or_one(labels):
    with pytest.raises(ValueError, match="라벨"):
        score_bucket_table(pd.Series([1, 2, 3, 4, 5]), pd.Series(labels))


def test_scores_and_labels_must_have_the_same_length():
    with pytest.raises(ValueError, match="길이"):
        score_bucket_table(pd.Series([1, 2, 3]), pd.Series([0, 1]))


# ---- 단조 판정 -------------------------------------------------------------
def test_monotone_when_bad_rate_falls_as_score_rises():
    scores, y = _scores_and_labels([16, 12, 8, 4, 0])
    verdict = check_monotone(score_bucket_table(scores, y))
    assert verdict.is_monotone is True
    assert verdict.violations == []


def test_non_monotone_is_detected_and_located():
    # 부실 수 [16, 8, 12, 4, 0] → 구간 2→3 에서 부실비가 오른다
    scores, y = _scores_and_labels([16, 8, 12, 4, 0])
    verdict = check_monotone(score_bucket_table(scores, y))
    assert verdict.is_monotone is False
    assert verdict.violations == [(2, 3)]


def test_equal_bad_rates_are_not_monotone():
    scores, y = _scores_and_labels([16, 12, 12, 4, 0])
    verdict = check_monotone(score_bucket_table(scores, y))
    assert verdict.is_monotone is False
    assert verdict.violations == [(2, 3)]


def test_rising_bad_rate_is_not_monotone_at_every_step():
    scores, y = _scores_and_labels([0, 4, 8, 12, 16])
    verdict = check_monotone(score_bucket_table(scores, y))
    assert verdict.is_monotone is False
    assert verdict.violations == [(1, 2), (2, 3), (3, 4), (4, 5)]


# ---- 등급별 표 -------------------------------------------------------------
BANDS = [{"name": "A", "min_score": 80}, {"name": "B", "min_score": 40}, {"name": "C", "min_score": 20}]


def test_band_table_counts_each_band_with_the_runtime_band_rule():
    scores, y = _scores_and_labels([16, 12, 8, 4, 0])
    t = band_table(scores, y, BANDS)
    assert t["band"].tolist() == ["A", "B", "C"]
    # A(>=80): 80~99 → 20명, 부실 0
    # B(>=40): 40~79 → 40명, 부실 8 + 4 = 12
    # C(>=20): 20~39 → 20명, 거기에 모든 경계 아래인 0~19 도 가장 낮은 등급이라 40명, 부실 12 + 16 = 28
    assert t["n"].tolist() == [20, 40, 40]
    assert t["bad"].tolist() == [0, 12, 28]
    assert t["bad_rate"].tolist() == pytest.approx([0.0, 12 / 40, 28 / 40])


def test_band_boundary_score_belongs_to_the_higher_band():
    t = band_table(pd.Series([39, 40, 79, 80]), pd.Series([0, 0, 0, 0]), BANDS)
    assert t["n"].tolist() == [1, 2, 1]


def test_empty_band_has_no_bad_rate():
    t = band_table(pd.Series([90, 95]), pd.Series([0, 1]), BANDS)
    assert t["n"].tolist() == [2, 0, 0]
    assert pd.isna(t["bad_rate"].iloc[1])


# ---- 등급 경계 초안 --------------------------------------------------------
def test_draft_band_proposal_uses_bucket_minimums_from_best_to_worst():
    scores, y = _scores_and_labels([16, 12, 8, 4, 0])
    proposal = draft_band_proposal(score_bucket_table(scores, y))
    assert [p["name"] for p in proposal] == ["A", "B", "C", "D", "E"]
    assert [p["min_score"] for p in proposal] == [80, 60, 40, 20, 0]


def test_draft_band_proposal_reproduces_the_buckets_when_applied():
    scores, y = _scores_and_labels([16, 12, 8, 4, 0])
    table = score_bucket_table(scores, y)
    applied = band_table(scores, y, draft_band_proposal(table))
    assert applied["n"].tolist() == table["n"].tolist()[::-1]
    assert applied["bad_rate"].tolist() == pytest.approx(table["bad_rate"].tolist()[::-1])


def test_draft_band_proposal_with_fewer_buckets_uses_the_first_letters():
    scores = pd.Series([500] * 60 + [510] * 30 + [520] * 10)
    y = pd.Series([1] * 30 + [0] * 30 + [1] * 10 + [0] * 20 + [0] * 10)
    proposal = draft_band_proposal(score_bucket_table(scores, y))
    names = [p["name"] for p in proposal]
    assert names == ["A", "B", "C", "D", "E"][: len(proposal)]


# ---- 마크다운 --------------------------------------------------------------
def test_markdown_report_contains_table_verdict_and_draft_notice():
    scores, y = _scores_and_labels([16, 12, 8, 4, 0])
    md = build_report(scores, y)
    assert "| 구간 | 점수 범위 | 건수 | 부실 | 부실비 |" in md
    assert "80.0%" in md and "0.0%" in md
    assert "단조" in md and "비단조" not in md
    assert "초안" in md
    assert md.count("\n|") >= 5 + 2  # 헤더, 구분선, 구간 5개


def test_markdown_report_flags_non_monotone_result():
    scores, y = _scores_and_labels([16, 8, 12, 4, 0])
    md = build_report(scores, y)
    assert "비단조" in md
    assert "구간 2 → 3" in md


def test_markdown_report_shows_card_bands_when_given():
    scores, y = _scores_and_labels([16, 12, 8, 4, 0])
    md = build_report(scores, y, bands=BANDS, bands_status="draft")
    assert "| 등급 | 최소 점수 | 건수 | 부실 | 부실비 |" in md
    assert "draft" in md

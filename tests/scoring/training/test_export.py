"""Phase 5: 학습 결과 → 스코어카드 JSON 내보내기 → 런타임 로드 왕복."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.domain.scoring.scorecard.card import ScoreCardError, load_scorecard
from training.scorecard.binning import apply_woe, fit_woe_tables
from training.scorecard.export import draft_bands, export_scorecard
from training.scorecard.fit import fit_logit
from training.scorecard.preprocess import LABEL
from training.scorecard.scale import FACTOR, OFFSET, build_points, score_frame

KINDS = {"delinq": "discrete", "paid": "continuous"}
META = dict(
    model_version="synthetic-test-v1",
    purpose="pipeline-validation",
    data_source="synthetic data for unit tests",
    currency="TWD",
    created_at="2026-10-11",
)
REPO_ROOT = Path(__file__).resolve().parents[3]


def _frame(seed, n=8000):
    rng = np.random.default_rng(seed)
    delinq = rng.choice([0, 1, 2, 3], size=n, p=[0.6, 0.2, 0.15, 0.05])
    paid = rng.gamma(2.0, 1000.0, n)
    logit = -2.0 + 0.9 * delinq - 0.0004 * paid
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"delinq": delinq, "paid": paid, LABEL: y})


@pytest.fixture(scope="module")
def trained():
    train = _frame(1)
    tables = fit_woe_tables(train, KINDS)
    fit = fit_logit(apply_woe(train, tables), train[LABEL])
    points = build_points(fit, tables)
    scores = score_frame(train, tables, points)
    return tables, fit, points, draft_bands(scores)


def _export(tmp_path, trained, **overrides):
    tables, fit, points, bands = trained
    path = tmp_path / "card.json"
    export_scorecard(path, fit, tables, points, bands=bands, **{**META, **overrides})
    return path


# ---- 왕복 ------------------------------------------------------------------
def test_export_then_load_matches_in_memory_values(tmp_path, trained):
    tables, fit, points, bands = trained
    card = load_scorecard(_export(tmp_path, trained))

    assert card.intercept == fit.intercept
    assert list(card.variables) == list(tables)
    for name, table in tables.items():
        var = card.variables[name]
        assert var.beta == fit.betas[name]
        assert len(var.bins) == len(table.bins)
        for b, src, woe, pts in zip(var.bins, table.bins, table.woes, points[name]):
            assert (b.lower, b.upper) == (src.lower, src.upper)
            assert b.bad_rate == src.bad_rate
            assert b.woe == woe
            assert b.points == pts


def test_export_load_export_gives_identical_content(tmp_path, trained):
    path = _export(tmp_path, trained)
    first = json.loads(path.read_text(encoding="utf-8"))
    card = load_scorecard(path)
    assert card.to_dict() == first

    again = tmp_path / "again.json"
    again.write_text(json.dumps(card.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    assert load_scorecard(again).to_dict() == first


def test_exported_file_has_the_spec_fields_and_scale(tmp_path, trained):
    data = json.loads(_export(tmp_path, trained).read_text(encoding="utf-8"))
    for key in ["model_version", "purpose", "data_source", "currency", "created_at",
                "scale", "intercept", "variables", "bands"]:
        assert key in data
    assert data["scale"] == {"base_score": 600, "base_good_odds": 50, "points_per_double": 20}
    assert data["purpose"] == "pipeline-validation"
    assert data["currency"] == "TWD"


def test_exported_points_still_sum_to_the_score_formula(tmp_path, trained):
    tables, fit, _, _ = trained
    card = load_scorecard(_export(tmp_path, trained))
    n = len(card.variables)
    for name, var in card.variables.items():
        for b in var.bins:
            expected = OFFSET / n - FACTOR * (card.intercept / n + var.beta * b.woe)
            assert b.points == pytest.approx(expected, abs=1e-9)


def test_exported_file_is_utf8_json_without_nan(tmp_path, trained):
    text = _export(tmp_path, trained).read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)


# ---- 내보내기 쪽 방어 ------------------------------------------------------
def test_export_refuses_non_finite_values(tmp_path, trained):
    tables, fit, points, bands = trained
    bad_points = {k: list(v) for k, v in points.items()}
    bad_points["delinq"][0] = float("nan")
    with pytest.raises(ScoreCardError):
        export_scorecard(tmp_path / "x.json", fit, tables, bad_points, bands=bands, **META)
    assert not (tmp_path / "x.json").exists()


def test_export_refuses_forbidden_variable(tmp_path):
    train = _frame(1)
    train = train.rename(columns={"delinq": "AGE"})
    tables = fit_woe_tables(train, {"AGE": "discrete", "paid": "continuous"})
    fit = fit_logit(apply_woe(train, tables), train[LABEL])
    points = build_points(fit, tables)
    bands = draft_bands(score_frame(train, tables, points))
    with pytest.raises(ScoreCardError, match="금지"):
        export_scorecard(tmp_path / "x.json", fit, tables, points, bands=bands, **META)
    assert not (tmp_path / "x.json").exists()


def test_export_creates_missing_parent_directory(tmp_path, trained):
    tables, fit, points, bands = trained
    out = tmp_path / "models" / "scorecards" / "card.json"
    export_scorecard(out, fit, tables, points, bands=bands, **META)
    assert out.exists()


# ---- 임시 등급 경계 --------------------------------------------------------
def test_draft_bands_use_score_quantiles():
    scores = pd.Series(range(100))
    bands = draft_bands(scores)
    assert [b["name"] for b in bands] == ["A", "B", "C", "D", "E"]
    assert [b["min_score"] for b in bands] == [79, 59, 39, 19, 0]


def test_draft_bands_are_strictly_descending(trained):
    mins = [b["min_score"] for b in trained[3]]
    assert all(a > b for a, b in zip(mins, mins[1:]))


def test_draft_bands_reject_scores_that_cannot_be_split():
    with pytest.raises(ValueError, match="등급"):
        draft_bands(pd.Series([500.0] * 50))


def test_exported_card_is_marked_draft_when_bands_are_drafts(tmp_path, trained):
    data = json.loads(_export(tmp_path, trained, bands_status="draft").read_text(encoding="utf-8"))
    assert data["bands_status"] == "draft"


# ---- 저장소에 커밋된 UCI 검증용 카드 ---------------------------------------
UCI_CARD = REPO_ROOT / "models" / "scorecards" / "uci-validation-v1.json"


def test_committed_uci_card_loads_and_is_pipeline_validation_only():
    card = load_scorecard(UCI_CARD)
    assert card.model_version == "uci-validation-v1"
    assert card.purpose == "pipeline-validation"
    assert card.currency == "TWD"
    assert "UCI" in card.data_source and "CC BY 4.0" in card.data_source
    assert card.bands_status == "draft"


def test_committed_uci_card_can_never_be_used_in_production():
    from app.domain.scoring.scorecard.card import ScoreCardPurposeError

    with pytest.raises(ScoreCardPurposeError):
        load_scorecard(UCI_CARD, production=True)


def test_committed_uci_card_has_the_five_to_be_variables():
    from training.scorecard.features import FEATURE_COLUMNS

    assert list(load_scorecard(UCI_CARD).variables) == FEATURE_COLUMNS

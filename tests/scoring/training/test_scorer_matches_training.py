"""Phase 6: 런타임 채점이 학습 쪽 계산(training.scale.score_frame)과 같은 점수를 내는지 교차 확인."""
import numpy as np
import pandas as pd
import pytest

from app.domain.scoring.scorecard.card import load_scorecard
from app.domain.scoring.scorecard.scorer import round_score, score_customer
from training.scorecard.binning import apply_woe, fit_woe_tables
from training.scorecard.export import draft_bands, export_scorecard
from training.scorecard.fit import fit_logit
from training.scorecard.preprocess import LABEL
from training.scorecard.scale import build_points, score_frame

KINDS = {"delinq": "discrete", "paid": "continuous"}


def _frame(seed, n):
    rng = np.random.default_rng(seed)
    delinq = rng.choice([0, 1, 2, 3], size=n, p=[0.6, 0.2, 0.15, 0.05])
    paid = rng.gamma(2.0, 1000.0, n)
    logit = -2.0 + 0.9 * delinq - 0.0004 * paid
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"delinq": delinq, "paid": paid, LABEL: y})


def test_runtime_scores_equal_training_scores_on_synthetic_pipeline(tmp_path):
    train, valid = _frame(1, 8000), _frame(2, 600)
    tables = fit_woe_tables(train, KINDS)
    fit = fit_logit(apply_woe(train, tables), train[LABEL])
    points = build_points(fit, tables)
    path = tmp_path / "card.json"
    export_scorecard(
        path, fit, tables, points,
        bands=draft_bands(score_frame(train, tables, points)),
        model_version="x", purpose="pipeline-validation", data_source="synthetic", currency="TWD",
    )
    card = load_scorecard(path)

    expected = score_frame(valid, tables, points)
    for (_, row), exp in zip(valid.iterrows(), expected):
        r = score_customer(card, {"delinq": row["delinq"], "paid": row["paid"]}, input_currency="TWD")
        assert r.raw_score == pytest.approx(exp, abs=1e-9)
        assert r.score == round_score(exp)
        expected_band = next((b.name for b in card.bands if r.score >= b.min_score), card.bands[-1].name)
        assert r.band == expected_band

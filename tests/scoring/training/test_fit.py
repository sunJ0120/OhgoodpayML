"""Phase 3: 로지스틱 회귀 (정규화 없음), β 부호 검사.

합성 데이터는 파이프라인 동작 검증용이다. 여기서 나온 계수는 어디에도 쓰지 않는다.
"""
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from training.scorecard.fit import CoefficientSignError, FitResult, fit_logit


def _synthetic(n=30_000, seed=11, coefs=(-0.9, -0.5), intercept=-1.2):
    """WoE 가 클수록 안전하도록(부실 오즈 ↓) 알려진 관계로 라벨을 만든다. 라벨 1 = 부실."""
    rng = np.random.default_rng(seed)
    woe = pd.DataFrame({f"w{i}": rng.normal(size=n) for i in range(len(coefs))})
    logit = intercept + sum(c * woe[f"w{i}"] for i, c in enumerate(coefs))
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), name="bad")
    return woe, y


def test_known_relationship_gives_all_negative_betas_close_to_truth():
    woe, y = _synthetic()
    result = fit_logit(woe, y)
    assert all(b < 0 for b in result.betas.values())
    assert result.betas["w0"] == pytest.approx(-0.9, abs=0.12)
    assert result.betas["w1"] == pytest.approx(-0.5, abs=0.12)
    assert result.intercept == pytest.approx(-1.2, abs=0.12)
    assert list(result.betas) == ["w0", "w1"]


def test_fit_matches_unregularized_statsmodels_logit():
    woe, y = _synthetic(n=5000, seed=3)
    result = fit_logit(woe, y)
    ref = sm.Logit(y, sm.add_constant(woe)).fit(disp=0)
    assert result.intercept == pytest.approx(ref.params["const"], abs=1e-6)
    for name in woe.columns:
        assert result.betas[name] == pytest.approx(ref.params[name], abs=1e-6)
        assert result.bse[name] == pytest.approx(ref.bse[name], abs=1e-6)


def test_predicted_bad_odds_equal_exp_of_linear_predictor():
    result = FitResult(intercept=-1.0, betas={"a": -2.0, "b": -0.5}, bse={"a": 0.1, "b": 0.1})
    woe = pd.DataFrame({"a": [0.5, 0.0], "b": [-1.0, 2.0]})
    odds = result.bad_odds(woe)
    assert odds.tolist() == pytest.approx([math.exp(-1.0 - 2.0 * 0.5 + 0.5), math.exp(-1.0 - 1.0)])


def test_predicted_bad_odds_equal_statsmodels_probability_odds():
    woe, y = _synthetic(n=5000, seed=5)
    result = fit_logit(woe, y)
    ref = sm.Logit(y, sm.add_constant(woe)).fit(disp=0)
    p = ref.predict(sm.add_constant(woe))
    assert result.bad_odds(woe).to_numpy() == pytest.approx((p / (1 - p)).to_numpy(), rel=1e-6)


def test_positive_beta_raises_and_names_the_variable():
    # w1 은 값이 클수록 부실이 늘어나는 관계 → β 가 양수가 된다
    woe, y = _synthetic(coefs=(-0.9, 0.6))
    with pytest.raises(CoefficientSignError, match="w1") as exc:
        fit_logit(woe, y)
    assert exc.value.variables == ["w1"]
    assert "w0" not in str(exc.value)


def test_sign_check_can_be_turned_off_for_reporting():
    woe, y = _synthetic(coefs=(-0.9, 0.6))
    result = fit_logit(woe, y, check_signs=False)
    assert result.betas["w1"] > 0


def test_sign_error_is_a_value_error():
    assert issubclass(CoefficientSignError, ValueError)


def test_no_variables_raises():
    with pytest.raises(ValueError, match="변수"):
        fit_logit(pd.DataFrame(index=range(5)), pd.Series([0, 1, 0, 1, 0]))


@pytest.mark.parametrize("bad_label", [[0, 1, 2, 0], [0, 1, None, 0]])
def test_label_must_be_zero_or_one(bad_label):
    woe = pd.DataFrame({"a": [0.1, 0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="라벨"):
        fit_logit(woe, pd.Series(bad_label))


def test_missing_woe_value_raises():
    woe = pd.DataFrame({"a": [0.1, None, 0.3, 0.4]})
    with pytest.raises(ValueError, match="결측"):
        fit_logit(woe, pd.Series([0, 1, 0, 1]))


def test_length_mismatch_raises():
    woe = pd.DataFrame({"a": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="길이"):
        fit_logit(woe, pd.Series([0, 1]))


def test_bad_odds_requires_exactly_the_fitted_variables():
    result = FitResult(intercept=0.0, betas={"a": -1.0}, bse={"a": 0.1})
    with pytest.raises(KeyError, match="a"):
        result.bad_odds(pd.DataFrame({"b": [1.0]}))

"""로지스틱 회귀 (정규화 없음).

    ln(부실 오즈) = β0 + Σ β_j · WoE_j        (라벨 1 = 부실)

WoE 가 클수록 안전하므로 모든 β_j 는 음수여야 한다. 양수이면 그 변수를 지목하는 예외를 낸다.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


class CoefficientSignError(ValueError):
    """β 가 0 이상인 변수가 있다."""

    def __init__(self, variables: list[str], betas: dict[str, float]):
        self.variables = variables
        detail = ", ".join(f"{v}={betas[v]:+.4f}" for v in variables)
        super().__init__(f"β 가 음수가 아닌 변수가 있다: {detail}")


@dataclass(frozen=True)
class FitResult:
    intercept: float
    betas: dict[str, float]
    bse: dict[str, float]  # 계수의 표준오차

    def bad_odds(self, woe: pd.DataFrame) -> pd.Series:
        """예측 부실 오즈 = exp(β0 + Σ β_j · WoE_j)."""
        missing = [v for v in self.betas if v not in woe.columns]
        if missing:
            raise KeyError(f"WoE 컬럼이 없다: {', '.join(missing)}")
        linear = self.intercept + sum(b * woe[v] for v, b in self.betas.items())
        return np.exp(linear)


def fit_logit(woe: pd.DataFrame, y: pd.Series, check_signs: bool = True) -> FitResult:
    if woe.shape[1] == 0:
        raise ValueError("변수가 없다")
    if len(woe) != len(y):
        raise ValueError(f"WoE 와 라벨의 길이가 다르다: {len(woe)} != {len(y)}")
    if woe.isna().any().any():
        raise ValueError("WoE 에 결측이 있다")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("라벨은 0 또는 1 만 허용한다")

    X = sm.add_constant(woe.reset_index(drop=True), has_constant="raise")
    fitted = sm.Logit(y.reset_index(drop=True).astype(int), X).fit(disp=0)
    if not fitted.mle_retvals.get("converged", False):
        raise RuntimeError("로지스틱 회귀가 수렴하지 않았다")

    names = list(woe.columns)
    result = FitResult(
        intercept=float(fitted.params["const"]),
        betas={n: float(fitted.params[n]) for n in names},
        bse={n: float(fitted.bse[n]) for n in names},
    )
    if check_signs:
        bad = [n for n, b in result.betas.items() if b >= 0]
        if bad:
            raise CoefficientSignError(bad, result.betas)
    return result

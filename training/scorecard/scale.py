"""오즈 → 점수, 배점표.

    점수  = Offset + Factor · ln(정상 오즈)
    기준  = 정상 오즈 50 일 때 600점, 정상 오즈가 2배가 될 때마다 +20점
    Factor = 20 / ln 2,   Offset = 600 − Factor · ln 50

    변수 j 가 구간 b 에 속할 때의 배점:
        points_j,b = Offset / n − Factor · ( β0 / n + β_j · WoE_j,b )      (n = 변수 개수)
    모든 변수의 배점 합 = Offset − Factor · ( β0 + Σ β_j · WoE_j ) = 점수

배점은 여기서 반올림하지 않는다 (합이 정확히 맞아야 한다). 반올림은 내보낼 때 정한다.
"""
import math

import numpy as np
import pandas as pd

from training.scorecard.binning import BinTable, assign_bins
from training.scorecard.fit import FitResult

BASE_SCORE = 600
BASE_GOOD_ODDS = 50
POINTS_PER_DOUBLE = 20

FACTOR = POINTS_PER_DOUBLE / math.log(2)
OFFSET = BASE_SCORE - FACTOR * math.log(BASE_GOOD_ODDS)


def score_from_good_odds(good_odds: float) -> float:
    if not good_odds > 0:
        raise ValueError(f"정상 오즈는 0 보다 커야 한다: {good_odds}")
    return OFFSET + FACTOR * math.log(good_odds)


def build_points(fit: FitResult, tables: dict[str, BinTable]) -> dict[str, list[float]]:
    """변수별로 구간 순서대로 배점 목록을 만든다."""
    if set(fit.betas) != set(tables):
        raise ValueError(
            f"회귀 변수와 구간표 변수가 다르다: 회귀={sorted(fit.betas)}, 구간표={sorted(tables)}"
        )
    non_negative = [v for v, b in fit.betas.items() if b >= 0]
    if non_negative:
        raise ValueError(f"β 가 음수가 아닌 변수가 있어 배점을 만들 수 없다: {', '.join(non_negative)}")

    n = len(tables)
    return {
        name: [
            OFFSET / n - FACTOR * (fit.intercept / n + beta * w)
            for w in tables[name].woes
        ]
        for name, beta in fit.betas.items()
    }


def score_frame(
    df: pd.DataFrame, tables: dict[str, BinTable], points: dict[str, list[float]]
) -> pd.Series:
    """고객별 점수 = 변수별 배점의 합."""
    total = pd.Series(0.0, index=df.index)
    for name, table in tables.items():
        idx = assign_bins(df[name], table.bins).to_numpy()
        total += np.asarray(points[name])[idx]
    return total

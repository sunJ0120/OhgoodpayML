"""구간화.

흐름: 초기 구간 → 단조 병합 → 최소 구간 크기 병합. 구간 경계와 WoE 는 학습 데이터로만 만들고,
검증 데이터에는 그 경계를 적용만 한다.

구간 경계 규칙: lower < x <= upper. None 은 무한대. 학습 범위 밖 값은 가장 바깥 구간으로 들어간다.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from training.scorecard import woe as woe_math
from training.scorecard.preprocess import LABEL

MIN_BIN_SHARE = 0.05
MAX_INITIAL_BINS = 5

# 연속 변수는 분위수, 이산 변수는 값 기준으로 시작한다
DEFAULT_KINDS = {
    "max_delinquency_months": "discrete",
    "on_time_ratio": "discrete",
    "paid_amount_6m": "continuous",
    "avg_billed_amount_6m": "continuous",
    "active_months": "discrete",
}

# 가설. 데이터가 반대로 나오면 억지로 맞추지 않고 보고한다. None 은 데이터로 확인.
EXPECTED_DIRECTION = {
    "max_delinquency_months": "increasing",  # 클수록 부실비 ↑
    "on_time_ratio": "decreasing",
    "paid_amount_6m": "decreasing",
    "avg_billed_amount_6m": None,
    "active_months": "decreasing",
}


@dataclass(frozen=True)
class Bin:
    lower: float | None
    upper: float | None
    good: int
    bad: int

    @property
    def n(self) -> int:
        return self.good + self.bad

    @property
    def bad_rate(self) -> float:
        return woe_math.bad_rate(self.good, self.bad)


@dataclass(frozen=True)
class BinTable:
    variable: str
    kind: str
    direction: str  # 'increasing': x 가 클수록 부실비 ↑
    bins: tuple[Bin, ...]

    @property
    def good_total(self) -> int:
        return sum(b.good for b in self.bins)

    @property
    def bad_total(self) -> int:
        return sum(b.bad for b in self.bins)

    @property
    def woes(self) -> list[float]:
        return [woe_math.woe(b.good, b.bad, self.good_total, self.bad_total) for b in self.bins]

    def transform_woe(self, x: pd.Series) -> pd.Series:
        idx = assign_bins(x, self.bins)
        return pd.Series(np.asarray(self.woes)[idx.to_numpy()], index=x.index, name=x.name)

    def to_frame(self) -> pd.DataFrame:
        total = self.good_total + self.bad_total
        return pd.DataFrame(
            {
                "lower": [b.lower for b in self.bins],
                "upper": [b.upper for b in self.bins],
                "n": [b.n for b in self.bins],
                "share": [b.n / total for b in self.bins],
                "good": [b.good for b in self.bins],
                "bad": [b.bad for b in self.bins],
                "bad_rate": [b.bad_rate for b in self.bins],
                "woe": self.woes,
            }
        )


# ---- 초기 구간과 할당 ------------------------------------------------------
def initial_uppers(x: pd.Series, kind: str, max_bins: int = MAX_INITIAL_BINS) -> list[float]:
    """유한한 상한 경계 목록. 마지막 구간(상한 없음)은 포함하지 않는다."""
    if kind == "discrete":
        uppers = np.unique(x.to_numpy())
    elif kind == "continuous":
        qs = np.arange(1, max_bins) / max_bins
        # 관측된 값만 경계로 쓴다 (빈 구간 방지)
        uppers = np.unique(np.quantile(x.to_numpy(), qs, method="lower"))
    else:
        raise ValueError(f"kind 는 'discrete' 또는 'continuous' 여야 한다: {kind!r}")
    return [u for u in uppers.tolist() if u < x.max()]


def assign_bins(x: pd.Series, bins) -> pd.Series:
    if x.isna().any():
        raise ValueError("구간을 할당할 값에 결측이 있다")
    uppers = [b.upper for b in bins[:-1]]
    idx = np.searchsorted(uppers, x.to_numpy(), side="left")  # lower < x <= upper
    return pd.Series(idx, index=x.index)


def count_bins(x: pd.Series, y: pd.Series, uppers: list[float]) -> list[Bin]:
    edges = [None, *uppers, None]
    shell = [Bin(edges[i], edges[i + 1], 0, 0) for i in range(len(edges) - 1)]
    idx = assign_bins(x, shell).to_numpy()
    yv = y.to_numpy()
    out = []
    for i, b in enumerate(shell):
        in_bin = idx == i
        out.append(Bin(b.lower, b.upper, int((yv[in_bin] == 0).sum()), int((yv[in_bin] == 1).sum())))
    return out


# ---- 병합 ------------------------------------------------------------------
def _merge(a: Bin, b: Bin) -> Bin:
    return Bin(a.lower, b.upper, a.good + b.good, a.bad + b.bad)


def merge_monotone(bins: list[Bin], direction: str) -> list[Bin]:
    """부실비가 direction 방향으로 엄격히 단조가 될 때까지 위반한 인접 구간을 합친다."""
    if direction not in ("increasing", "decreasing"):
        raise ValueError(f"direction 은 'increasing' 또는 'decreasing' 이어야 한다: {direction!r}")
    sign = 1 if direction == "increasing" else -1
    bins = list(bins)
    while True:
        for i in range(len(bins) - 1):
            if sign * (bins[i + 1].bad_rate - bins[i].bad_rate) <= 0:
                bins[i : i + 2] = [_merge(bins[i], bins[i + 1])]
                break
        else:
            return bins


def merge_small(bins: list[Bin], min_share: float = MIN_BIN_SHARE) -> list[Bin]:
    """전체의 min_share 미만인 구간을 부실비가 가장 가까운 이웃에 합친다."""
    bins = list(bins)
    total = sum(b.n for b in bins)
    while len(bins) > 1:
        i = min(range(len(bins)), key=lambda k: bins[k].n)
        if bins[i].n / total >= min_share:
            break
        neighbors = [j for j in (i - 1, i + 1) if 0 <= j < len(bins)]
        j = min(neighbors, key=lambda k: abs(bins[k].bad_rate - bins[i].bad_rate))
        lo, hi = sorted((i, j))
        bins[lo : hi + 1] = [_merge(bins[lo], bins[hi])]
    return bins


def _choose_direction(x: pd.Series, y: pd.Series, bins: list[Bin]) -> str:
    inc = merge_monotone(bins, "increasing")
    dec = merge_monotone(bins, "decreasing")
    if len(inc) != len(dec):
        return "increasing" if len(inc) > len(dec) else "decreasing"
    corr = x.rank().corr(y)
    return "decreasing" if corr < 0 else "increasing"


# ---- 학습 / 적용 -----------------------------------------------------------
def fit_bins(x: pd.Series, y: pd.Series, kind: str) -> BinTable:
    bins = count_bins(x, y, initial_uppers(x, kind))
    direction = _choose_direction(x, y, bins)
    bins = merge_small(merge_monotone(bins, direction), MIN_BIN_SHARE)
    return BinTable(variable=str(x.name), kind=kind, direction=direction, bins=tuple(bins))


def fit_woe_tables(train: pd.DataFrame, kinds: dict[str, str]) -> dict[str, BinTable]:
    """학습 데이터로만 구간과 WoE 를 만든다. 라벨 컬럼은 preprocess.LABEL."""
    return {name: fit_bins(train[name], train[LABEL], kind) for name, kind in kinds.items()}


def apply_woe(df: pd.DataFrame, tables: dict[str, BinTable]) -> pd.DataFrame:
    """학습에서 만든 구간표를 적용만 한다. 라벨은 쓰지 않는다."""
    return pd.DataFrame({name: t.transform_woe(df[name]) for name, t in tables.items()}, index=df.index)


def direction_mismatches(tables: dict[str, BinTable]) -> list[str]:
    """가설(EXPECTED_DIRECTION)과 반대 방향으로 나온 변수."""
    return [
        name
        for name, t in tables.items()
        if EXPECTED_DIRECTION.get(name) is not None and t.direction != EXPECTED_DIRECTION[name]
    ]


def unusable_variables(tables: dict[str, BinTable]) -> list[str]:
    """병합 결과 구간이 1개뿐이라 구분력이 없는 변수."""
    return [name for name, t in tables.items() if len(t.bins) < 2]

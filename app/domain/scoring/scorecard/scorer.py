"""런타임 채점. 스코어카드 파일(ScoreCard)만 읽는다.

    점수  = 변수별 배점의 합 (소수 그대로 합산한 뒤 마지막에 한 번 반올림, 0.5 는 올림)
    구간  = lower < x <= upper, 학습 범위 밖 값은 가장 바깥 구간
    감점  = (해당 변수 최고 배점 − 실제 배점) 이 큰 순서로 상위 3개, 손실 0 인 변수는 제외
    등급  = 반올림된 점수가 처음으로 min_score 이상인 등급, 모든 경계 아래면 가장 낮은 등급

입력 변수가 하나라도 없으면 임의의 기본값으로 채우지 않고 예외를 낸다.
"""
import math
import numbers
from bisect import bisect_left
from dataclasses import dataclass
from typing import Mapping

from .card import ScoreCard, ScoreCardError, VariableCard

TOP_NEGATIVE = 3


class FeatureError(ScoreCardError):
    """채점 입력이 카드와 맞지 않는다."""


class MissingFeatureError(FeatureError):
    """카드에 있는 변수가 입력에 없다."""


class UnknownFeatureError(FeatureError):
    """카드에 없는 변수가 입력에 있다."""


class InvalidFeatureError(FeatureError):
    """입력 값이 유한한 숫자가 아니다."""


@dataclass(frozen=True)
class VariableScore:
    variable: str
    value: float
    bin: str
    points: float
    max_points: float

    @property
    def points_lost(self) -> float:
        return self.max_points - self.points


@dataclass(frozen=True)
class ScoreResult:
    score: int
    raw_score: float
    band: str
    model_version: str
    variables: dict[str, VariableScore]
    top_negative: tuple[VariableScore, ...]

    def to_dict(self) -> dict:
        def reason(v: VariableScore) -> dict:
            return {
                "variable": v.variable,
                "value": v.value,
                "bin": v.bin,
                "points": v.points,
                "max_points": v.max_points,
                "points_lost": v.points_lost,
            }

        return {
            "score": self.score,
            "raw_score": self.raw_score,
            "band": self.band,
            "model_version": self.model_version,
            "variables": {
                name: {"value": v.value, "bin": v.bin, "points": v.points, "max_points": v.max_points}
                for name, v in self.variables.items()
            },
            "top_negative": [reason(v) for v in self.top_negative],
        }


def round_score(raw: float) -> int:
    """0.5 는 올림. 파이썬 round() 는 짝수로 반올림하므로 쓰지 않는다."""
    return math.floor(raw + 0.5)


def _fmt(v) -> str:
    return str(int(v)) if float(v).is_integer() else repr(v)


def _bin_label(var: VariableCard, index: int) -> str:
    b = var.bins[index]
    lower = "-inf" if b.lower is None else _fmt(b.lower)
    upper = "inf" if b.upper is None else _fmt(b.upper)
    return f"({lower}, {upper}]"


def _bin_index(var: VariableCard, value: float) -> int:
    uppers = [b.upper for b in var.bins[:-1]]
    return bisect_left(uppers, value)  # lower < x <= upper


def _check_features(card: ScoreCard, features: Mapping[str, float]) -> None:
    missing = [name for name in card.variables if name not in features]
    if missing:
        raise MissingFeatureError(f"입력에 없는 변수가 있다: {', '.join(missing)}")
    unknown = sorted(set(features) - set(card.variables))
    if unknown:
        raise UnknownFeatureError(f"카드에 없는 변수가 입력에 있다: {', '.join(unknown)}")
    for name in card.variables:
        value = features[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
        ):
            raise InvalidFeatureError(f"{name} 은(는) 유한한 숫자여야 한다: {value!r}")


def _band(card: ScoreCard, score: int) -> str:
    for band in card.bands:
        if score >= band.min_score:
            return band.name
    return card.bands[-1].name


def score_customer(card: ScoreCard, features: Mapping[str, float], *, input_currency: str) -> ScoreResult:
    """고객 1명을 채점한다. input_currency 가 카드 통화와 다르면 채점하지 않는다."""
    card.assert_currency(input_currency)
    _check_features(card, features)

    scored: dict[str, VariableScore] = {}
    for name, var in card.variables.items():
        value = features[name]
        index = _bin_index(var, value)
        scored[name] = VariableScore(
            variable=name,
            value=value,
            bin=_bin_label(var, index),
            points=var.bins[index].points,
            max_points=max(b.points for b in var.bins),
        )

    raw = sum(v.points for v in scored.values())
    score = round_score(raw)

    # sorted 는 안정 정렬이라 손실이 같으면 카드의 변수 순서를 따른다
    losses = sorted(
        (v for v in scored.values() if v.points_lost > 0),
        key=lambda v: v.points_lost,
        reverse=True,
    )
    return ScoreResult(
        score=score,
        raw_score=raw,
        band=_band(card, score),
        model_version=card.model_version,
        variables=scored,
        top_negative=tuple(losses[:TOP_NEGATIVE]),
    )

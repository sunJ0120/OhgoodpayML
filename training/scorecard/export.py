"""학습 결과를 스코어카드 JSON 으로 내보낸다. 내보내기 전에 런타임 로더로 같은 검증을 한 번 통과시킨다."""
import json
import math
from pathlib import Path

import pandas as pd

from app.domain.scoring.scorecard.card import ScoreCardError, parse_scorecard
from training.scorecard.binning import BinTable
from training.scorecard.fit import FitResult
from training.scorecard.scale import BASE_GOOD_ODDS, BASE_SCORE, POINTS_PER_DOUBLE

BAND_NAMES = ["A", "B", "C", "D", "E"]


def draft_bands(scores: pd.Series) -> list[dict]:
    """점수 5분위로 만든 임시 등급 경계 (A 가 가장 높음). 사람이 정하기 전까지의 초안이다."""
    cutoffs = [math.floor(scores.quantile(q)) for q in (0.8, 0.6, 0.4, 0.2)]
    cutoffs.append(math.floor(scores.min()))
    if any(a <= b for a, b in zip(cutoffs, cutoffs[1:])):
        raise ValueError(f"점수 분포로는 등급 경계를 나눌 수 없다: {cutoffs}")
    return [{"name": n, "min_score": int(c)} for n, c in zip(BAND_NAMES, cutoffs)]


def _py(value):
    """numpy 스칼라를 파이썬 값으로 바꾼다 (JSON 직렬화)."""
    return value.item() if hasattr(value, "item") else value


def build_card_dict(
    fit: FitResult,
    tables: dict[str, BinTable],
    points: dict[str, list[float]],
    *,
    bands: list[dict],
    model_version: str,
    purpose: str,
    data_source: str,
    currency: str,
    created_at: str | None = None,
    bands_status: str | None = None,
) -> dict:
    if not (set(fit.betas) == set(tables) == set(points)):
        raise ScoreCardError(
            f"회귀·구간표·배점의 변수가 다르다: {sorted(fit.betas)} / {sorted(tables)} / {sorted(points)}"
        )

    data = {
        "model_version": model_version,
        "purpose": purpose,
        "data_source": data_source,
        "currency": currency,
    }
    if created_at is not None:
        data["created_at"] = created_at
    data["scale"] = {
        "base_score": BASE_SCORE,
        "base_good_odds": BASE_GOOD_ODDS,
        "points_per_double": POINTS_PER_DOUBLE,
    }
    data["intercept"] = float(fit.intercept)
    data["variables"] = {
        name: {
            "beta": float(fit.betas[name]),
            "bins": [
                {
                    "lower": _py(b.lower),
                    "upper": _py(b.upper),
                    "bad_rate": float(b.bad_rate),
                    "woe": float(woe),
                    "points": float(pts),
                }
                for b, woe, pts in zip(tables[name].bins, tables[name].woes, points[name])
            ],
        }
        for name in fit.betas
    }
    data["bands"] = bands
    if bands_status is not None:
        data["bands_status"] = bands_status
    return data


def export_scorecard(path, fit, tables, points, **meta) -> Path:
    """검증을 통과한 경우에만 파일을 쓴다. 실패하면 파일을 만들지 않는다."""
    data = build_card_dict(fit, tables, points, **meta)
    parse_scorecard(data)  # 런타임과 같은 규칙으로 검증 (금지 변수, 경계, 유한한 숫자 …)
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return out

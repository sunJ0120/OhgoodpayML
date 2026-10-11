"""검증 리포트: 검증 데이터(30%)를 커밋된 카드로 채점해 점수 구간별 부실비를 만든다.

    python -m training.scorecard.report [저장할_마크다운_경로]

확인하려는 것은 하나다. **점수가 오를수록 부실비가 내려가는가.**
등급 경계값은 이 표를 보고 사람이 정한다. 여기서는 초안만 제시한다.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from app.domain.scoring.scorecard.card import ScoreCard, load_scorecard
from app.domain.scoring.scorecard.scorer import score_customer
from training.scorecard.features import FEATURE_COLUMNS, build_training_frame
from training.scorecard.load import load_uci
from training.scorecard.preprocess import LABEL
from training.scorecard.run_uci import CURRENCY, SEED, TEST_SIZE, UCI_CARD_PATH
from training.scorecard.split import stratified_split

N_BUCKETS = 5
BAND_NAMES = ["A", "B", "C", "D", "E"]


def _validate(scores: pd.Series, y: pd.Series) -> None:
    if len(scores) != len(y):
        raise ValueError(f"점수와 라벨의 길이가 다르다: {len(scores)} != {len(y)}")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("라벨은 0 또는 1 만 허용한다")


# ---- 구간 표 ---------------------------------------------------------------
def score_bucket_table(scores: pd.Series, y: pd.Series, n_buckets: int = N_BUCKETS) -> pd.DataFrame:
    """점수를 분위수로 n_buckets 개 구간(낮은 점수 → 높은 점수)으로 나눠 구간별 건수와 부실비를 센다.

    경계는 관측된 점수만 쓰고 lower < x <= upper 규칙이다. 같은 점수가 두 구간에 나뉘지 않으므로
    점수가 이산적으로 몰려 있으면 구간이 n_buckets 보다 적게 나올 수 있다.
    """
    _validate(scores, y)
    s = scores.to_numpy()
    edges = np.unique(np.quantile(s, np.linspace(0, 1, n_buckets + 1), method="lower"))
    if len(edges) < 2:
        raise ValueError("점수가 모두 같아 구간을 나눌 수 없다")

    idx = np.searchsorted(edges[1:-1], s, side="left")
    yv = y.to_numpy()
    rows = []
    for b in range(len(edges) - 1):
        in_bucket = idx == b
        n = int(in_bucket.sum())
        bad = int(yv[in_bucket].sum())
        rows.append(
            {
                "bucket": b + 1,
                "score_min": int(s[in_bucket].min()),
                "score_max": int(s[in_bucket].max()),
                "n": n,
                "share": n / len(s),
                "bad": bad,
                "bad_rate": bad / n,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Monotonicity:
    is_monotone: bool
    violations: list[tuple[int, int]]  # 부실비가 내려가지 않은 인접 구간 (구간 번호 쌍)


def check_monotone(table: pd.DataFrame) -> Monotonicity:
    """점수가 오를수록 부실비가 엄격히 내려가는가. 같거나 오르면 위반이다."""
    rates = table["bad_rate"].tolist()
    buckets = table["bucket"].tolist()
    violations = [
        (buckets[i], buckets[i + 1]) for i in range(len(rates) - 1) if rates[i + 1] >= rates[i]
    ]
    return Monotonicity(is_monotone=not violations, violations=violations)


# ---- 등급 ------------------------------------------------------------------
def _band_of(score: float, bands: list[dict]) -> int:
    for i, band in enumerate(bands):
        if score >= band["min_score"]:
            return i
    return len(bands) - 1  # 런타임과 같은 규칙: 모든 경계 아래면 가장 낮은 등급


def band_table(scores: pd.Series, y: pd.Series, bands: list[dict]) -> pd.DataFrame:
    """카드의 등급 경계로 나눴을 때 등급별 건수와 부실비 (런타임 등급 규칙과 같다)."""
    _validate(scores, y)
    idx = np.array([_band_of(s, bands) for s in scores.to_numpy()])
    yv = y.to_numpy()
    rows = []
    for i, band in enumerate(bands):
        in_band = idx == i
        n = int(in_band.sum())
        bad = int(yv[in_band].sum())
        rows.append(
            {
                "band": band["name"],
                "min_score": band["min_score"],
                "n": n,
                "share": n / len(scores),
                "bad": bad,
                "bad_rate": bad / n if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def draft_band_proposal(table: pd.DataFrame) -> list[dict]:
    """구간 표에서 만든 등급 경계 초안. 가장 높은 점수 구간이 A. 사람이 결정하기 전까지의 초안이다."""
    top_down = table.iloc[::-1]
    return [
        {"name": name, "min_score": int(row.score_min)}
        for name, row in zip(BAND_NAMES, top_down.itertuples())
    ]


# ---- 마크다운 --------------------------------------------------------------
def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _pct(v: float) -> str:
    return "-" if pd.isna(v) else f"{v:.1%}"


def build_report(
    scores: pd.Series,
    y: pd.Series,
    *,
    bands: list[dict] | None = None,
    bands_status: str | None = None,
    title: str = "점수 구간별 부실비",
) -> str:
    table = score_bucket_table(scores, y)
    verdict = check_monotone(table)

    out = [f"## {title}", ""]
    out.append(f"검증 데이터 {len(scores):,}명, 전체 부실비 {y.mean():.1%}. 구간은 점수가 낮은 쪽부터 번호를 붙였다.")
    out.append("")
    out.append(
        _md_table(
            ["구간", "점수 범위", "건수", "부실", "부실비"],
            [
                [str(r.bucket), f"{r.score_min} ~ {r.score_max}", f"{r.n:,} ({r.share:.1%})", f"{r.bad:,}", _pct(r.bad_rate)]
                for r in table.itertuples()
            ],
        )
    )
    out.append("")
    if verdict.is_monotone:
        out.append("**판정: 단조.** 점수가 오를수록 부실비가 내려간다.")
    else:
        pairs = ", ".join(f"구간 {a} → {b}" for a, b in verdict.violations)
        out.append(f"**판정: 비단조.** 점수가 올랐는데 부실비가 내려가지 않은 곳: {pairs}")
    if len(table) < N_BUCKETS:
        out.append(f"(같은 점수가 몰려 {N_BUCKETS}구간이 아니라 {len(table)}구간으로 나뉘었다)")
    out.append("")

    if bands is not None:
        status = f" (`bands_status` = {bands_status})" if bands_status else ""
        out.append(f"### 현재 카드의 등급 경계로 나눈 결과{status}")
        out.append("")
        bt = band_table(scores, y, bands)
        out.append(
            _md_table(
                ["등급", "최소 점수", "건수", "부실", "부실비"],
                [[r.band, str(r.min_score), f"{r.n:,} ({r.share:.1%})", f"{r.bad:,}", _pct(r.bad_rate)] for r in bt.itertuples()],
            )
        )
        out.append("")

    proposal = draft_band_proposal(table)
    out.append("### 등급 경계 초안 (사람이 결정)")
    out.append("")
    out.append("위 구간표의 구간 최소 점수를 그대로 쓴 **초안**이다. 확정하려면 부실비 차이와 운영 정책(승인 한도 등)을 보고 사람이 정한다.")
    out.append("")
    applied = band_table(scores, y, proposal)
    out.append(
        _md_table(
            ["등급", "최소 점수", "건수", "부실비"],
            [[r.band, str(r.min_score), f"{r.n:,}", _pct(r.bad_rate)] for r in applied.itertuples()],
        )
    )
    return "\n".join(out) + "\n"


# ---- 실제 데이터 -----------------------------------------------------------
def score_with_card(card: ScoreCard, df: pd.DataFrame, currency: str = CURRENCY) -> pd.Series:
    """런타임 채점기로 한 명씩 채점한 최종 점수(정수)."""
    scores = [
        score_customer(card, {c: float(row[c]) for c in FEATURE_COLUMNS}, input_currency=currency).score
        for _, row in df.iterrows()
    ]
    return pd.Series(scores, index=df.index, dtype="int64")


def validation_scores(card_path=UCI_CARD_PATH) -> tuple[pd.Series, pd.Series]:
    """학습에 쓰지 않은 검증 데이터(30%)의 (점수, 라벨)."""
    frame = build_training_frame(load_uci())
    _, valid = stratified_split(frame, LABEL, TEST_SIZE, SEED)
    card = load_scorecard(card_path)
    return score_with_card(card, valid), valid[LABEL]


def build_validation_report(card_path=UCI_CARD_PATH) -> str:
    card = load_scorecard(card_path)
    scores, y = validation_scores(card_path)
    return build_report(
        scores,
        y,
        bands=[{"name": b.name, "min_score": b.min_score} for b in card.bands],
        bands_status=card.bands_status,
        title=f"점수 구간별 부실비 ({card.model_version})",
    )


def main() -> None:
    report = build_validation_report()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(report, encoding="utf-8", newline="\n")
        print(f"저장: {sys.argv[1]}")
    print(report)


if __name__ == "__main__":
    main()

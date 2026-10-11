"""배점표와 점수 분포를 출력한다.

    python -m training.scorecard.report_points
"""
import math

from training.scorecard.binning import DEFAULT_KINDS, apply_woe, fit_woe_tables
from training.scorecard.features import build_training_frame
from training.scorecard.fit import fit_logit
from training.scorecard.load import load_uci
from training.scorecard.preprocess import LABEL
from training.scorecard.scale import FACTOR, OFFSET, build_points, score_frame, score_from_good_odds
from training.scorecard.split import stratified_split


def _edge(v) -> str:
    return "-inf" if v is None else f"{v:,.4g}"


def main() -> None:
    frame = build_training_frame(load_uci())
    train, valid = stratified_split(frame, LABEL, 0.3, 42)
    tables = fit_woe_tables(train, DEFAULT_KINDS)
    fit = fit_logit(apply_woe(train, tables), train[LABEL])
    points = build_points(fit, tables)

    print(f"Factor = {FACTOR:.4f}, Offset = {OFFSET:.4f}, 변수 수 n = {len(tables)}")
    print(f"절편 β0 = {fit.intercept:+.4f}\n")

    for name, table in tables.items():
        print(f"### {name} (β = {fit.betas[name]:+.4f})")
        print("| 구간 | 부실비 | WoE | 배점 |")
        print("|---|---:|---:|---:|")
        for b, w, p in zip(table.bins, table.woes, points[name]):
            upper = "inf" if b.upper is None else _edge(b.upper)
            print(f"| ({_edge(b.lower)}, {upper}] | {b.bad_rate:.1%} | {w:+.4f} | {p:.2f} |")
        print()

    best = sum(max(p) for p in points.values())
    worst = sum(min(p) for p in points.values())
    print(f"이론상 최고 점수 {best:.1f} / 최저 점수 {worst:.1f}")

    avg_good_odds = (1 - train[LABEL].mean()) / train[LABEL].mean()
    print(f"학습 데이터 평균 정상 오즈 {avg_good_odds:.2f} → 기준식 점수 {score_from_good_odds(avg_good_odds):.1f}")

    for label, df in (("학습", train), ("검증", valid)):
        s = score_frame(df, tables, points)
        q = s.quantile([0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        print(f"{label} 점수 분위수(0/5/25/50/75/95/100%): " + ", ".join(f"{v:.1f}" for v in q))
        print(f"  평균 {s.mean():.1f}, 600점 이상 {(s >= 600).mean():.1%}, 650점 이상 {(s >= 650).mean():.1%}")

    # 점수식이 확률과 맞는지: 평균 점수 → 오즈 역변환
    mean_score = score_frame(train, tables, points).mean()
    print(f"학습 평균 점수 {mean_score:.1f} → 정상 오즈 환산 {math.exp((mean_score - OFFSET) / FACTOR):.2f}")


if __name__ == "__main__":
    main()

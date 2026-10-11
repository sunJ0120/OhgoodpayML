"""변수별 구간표(구간, 건수, 부실비, WoE)와 가설과 다른 변수를 출력한다.

    python -m training.scorecard.report_bins
"""
from training.scorecard.binning import (
    DEFAULT_KINDS,
    EXPECTED_DIRECTION,
    direction_mismatches,
    fit_woe_tables,
    unusable_variables,
)
from training.scorecard.features import build_training_frame
from training.scorecard.load import load_uci
from training.scorecard.preprocess import LABEL
from training.scorecard.split import stratified_split

SEED = 42
TEST_SIZE = 0.3


def _edge(v) -> str:
    return "-inf" if v is None else f"{v:,.4g}"


def main() -> None:
    frame = build_training_frame(load_uci())
    train, valid = stratified_split(frame, LABEL, TEST_SIZE, SEED)
    print(f"학습 {len(train):,} (부실비 {train[LABEL].mean():.4f}) / 검증 {len(valid):,} (부실비 {valid[LABEL].mean():.4f}), seed={SEED}")

    tables = fit_woe_tables(train, DEFAULT_KINDS)
    for name, table in tables.items():
        expected = EXPECTED_DIRECTION[name]
        print(f"\n### {name} ({table.kind}, 데이터 방향={table.direction}, 가설={expected or '없음'})")
        print("| 구간 | 건수 | 비중 | 정상 | 부실 | 부실비 | WoE |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for b, w in zip(table.bins, table.woes):
            upper = "inf" if b.upper is None else _edge(b.upper)
            share = b.n / (table.good_total + table.bad_total)
            print(f"| ({_edge(b.lower)}, {upper}] | {b.n:,} | {share:.1%} | {b.good:,} | {b.bad:,} | {b.bad_rate:.1%} | {w:+.4f} |")

    print("\n가설과 다른 방향:", direction_mismatches(tables) or "없음")
    print("구간 1개뿐(구분력 없음):", unusable_variables(tables) or "없음")


if __name__ == "__main__":
    main()

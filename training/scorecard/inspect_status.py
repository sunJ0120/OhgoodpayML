"""상환 상태 실제 값 분포를 출력한다. 정규화 규칙(preprocess.normalize_repay_status)을 확정하기 전에 본다.

    python -m training.scorecard.inspect_status
"""
from training.scorecard.load import load_uci
from training.scorecard.preprocess import LABEL, STATUS_COLUMNS, repay_status_distribution


def main() -> None:
    raw = load_uci()
    dist = repay_status_distribution(raw)

    print("== 값별 건수 (열: 컬럼, 행: 상환 상태 값) ==")
    print(dist.pivot(index="value", columns="column", values="count")[STATUS_COLUMNS].fillna(0).astype(int).to_string())

    print("\n== PAY_0 값별 다음 달 부실비 ==")
    g = raw.groupby("PAY_0")[LABEL].agg(건수="size", 부실비="mean")
    g["부실비"] = g["부실비"].round(4)
    print(g.to_string())

    print("\n== 전체 부실비 ==")
    print(round(raw[LABEL].mean(), 4))


if __name__ == "__main__":
    main()

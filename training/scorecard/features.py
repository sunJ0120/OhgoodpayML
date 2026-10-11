"""TO-BE 변수 5개 계산. 입력은 UCI 원본 컬럼 이름을 가진 DataFrame."""
import pandas as pd

from training.scorecard.preprocess import (
    BILL_COLUMNS,
    LABEL,
    PAY_AMOUNT_COLUMNS,
    STATUS_COLUMNS,
    normalize_repay_status,
    validate_training_frame,
)

FEATURE_COLUMNS = [
    "max_delinquency_months",
    "on_time_ratio",
    "paid_amount_6m",
    "avg_billed_amount_6m",
    "active_months",
]


def _require(raw: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in raw.columns]
    if missing:
        raise KeyError(f"필수 컬럼이 없다: {', '.join(missing)}")


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    _require(raw, STATUS_COLUMNS + BILL_COLUMNS + PAY_AMOUNT_COLUMNS)

    status = normalize_repay_status(raw[STATUS_COLUMNS])
    bill = raw[BILL_COLUMNS]

    return pd.DataFrame(
        {
            "max_delinquency_months": status.max(axis=1),
            "on_time_ratio": (status == 0).sum(axis=1) / len(STATUS_COLUMNS),
            "paid_amount_6m": raw[PAY_AMOUNT_COLUMNS].sum(axis=1),
            "avg_billed_amount_6m": bill.mean(axis=1),
            "active_months": (bill > 0).sum(axis=1),
        },
        index=raw.index,
    )[FEATURE_COLUMNS]


def build_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """변수 5개 + 라벨. 제외 컬럼은 이 단계에서 구조적으로 빠진다."""
    _require(raw, [LABEL])
    frame = build_features(raw)
    frame[LABEL] = raw[LABEL]
    validate_training_frame(frame)
    return frame

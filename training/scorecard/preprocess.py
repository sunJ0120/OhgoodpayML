"""상환 상태 정규화, 제외 컬럼 차단, 학습 입력 검증."""
import numpy as np
import pandas as pd

LABEL = "default_next_month"

# UCI 월별 상환 상태: PAY_0 = 9월(가장 최근) ... PAY_6 = 4월. PAY_1 은 데이터에 없다.
STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLUMNS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMOUNT_COLUMNS = [f"PAY_AMT{i}" for i in range(1, 7)]

# 점수에 쓰지 않는 컬럼 (차별 요소, 순환 구조)
FORBIDDEN_COLUMNS = {"SEX", "EDUCATION", "MARRIAGE", "AGE", "LIMIT_BAL"}


def normalize_repay_status(values):
    """−2, −1, 0 → 0(연체 없음), k ≥ 1 → k(연체 개월 수).

    문서에 없는 −2, 0 을 연체 없음으로 보는 것은 가정이다. docs/scoring/scorecard-v2.md 의
    '데이터 결정 기록'에 근거와 함께 남긴다.
    """
    return np.clip(values, 0, None)


def repay_status_distribution(raw: pd.DataFrame) -> pd.DataFrame:
    """상환 상태 컬럼별 실제 값 분포 (column, value, count)."""
    rows = []
    for col in STATUS_COLUMNS:
        counts = raw[col].value_counts().sort_index()
        rows.extend((col, int(v), int(n)) for v, n in counts.items())
    return pd.DataFrame(rows, columns=["column", "value", "count"])


def validate_training_frame(df: pd.DataFrame) -> None:
    """학습 입력 게이트. 제외 컬럼이 있거나 라벨이 0/1 이 아니면 예외."""
    leaked = sorted(FORBIDDEN_COLUMNS & set(df.columns))
    if leaked:
        raise ValueError(f"학습 입력에 제외 컬럼이 있다: {', '.join(leaked)}")
    if LABEL not in df.columns:
        raise ValueError(f"라벨 컬럼 {LABEL} 이 없다")
    label = df[LABEL]
    if label.isna().any() or not label.isin([0, 1]).all():
        raise ValueError(f"라벨 {LABEL} 은 0 또는 1 만 허용한다")

"""UCI 'Default of Credit Card Clients' 수집. 원본은 로컬 캐시에만 두고 git 에는 올리지 않는다."""
from pathlib import Path

import pandas as pd

from training.scorecard.preprocess import LABEL

UCI_ID = 350
CACHE_PATH = Path(__file__).parent / ".cache" / "uci_default_credit_card.csv"

# ucimlrepo 는 컬럼을 X1..X23 으로 주고 원래 이름은 variables.description 에만 있다.
UCI_COLUMN_BY_DESCRIPTION = {
    name: name
    for name in [
        "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        *[f"BILL_AMT{i}" for i in range(1, 7)],
        *[f"PAY_AMT{i}" for i in range(1, 7)],
    ]
}
UCI_LABEL_DESCRIPTION = "default payment next month"


def rename_uci_columns(X: pd.DataFrame, y: pd.DataFrame, variables: pd.DataFrame) -> pd.DataFrame:
    """X1..X23, Y 를 원래 컬럼 이름과 LABEL 로 되돌린다. 모르는 설명이 있으면 예외."""
    desc_by_code = dict(zip(variables["name"], variables["description"]))
    mapping = {}
    for code in X.columns:
        desc = desc_by_code.get(code)
        if desc not in UCI_COLUMN_BY_DESCRIPTION:
            raise ValueError(f"알 수 없는 UCI 컬럼 설명: {desc!r} ({code})")
        mapping[code] = UCI_COLUMN_BY_DESCRIPTION[desc]

    out = X.rename(columns=mapping).copy()
    out[LABEL] = y.iloc[:, 0].to_numpy()
    return out


def load_uci(use_cache: bool = True) -> pd.DataFrame:
    if use_cache and CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH)

    from ucimlrepo import fetch_ucirepo  # 학습 환경에서만 필요

    ds = fetch_ucirepo(id=UCI_ID)
    raw = rename_uci_columns(ds.data.features, ds.data.targets, ds.variables)

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(CACHE_PATH, index=False)
    return raw

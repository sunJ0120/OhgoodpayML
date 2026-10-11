"""층화 학습/검증 분할. 시드를 고정하면 항상 같은 결과가 나온다."""
import numpy as np
import pandas as pd


def stratified_split(
    df: pd.DataFrame, label: str, test_size: float = 0.3, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """라벨 비율을 학습/검증에 똑같이 나눈다. 반환: (train, valid), 원래 인덱스와 순서를 유지."""
    rng = np.random.default_rng(seed)
    valid_positions: list[int] = []
    for value in sorted(df[label].unique()):
        positions = np.flatnonzero((df[label] == value).to_numpy())
        shuffled = rng.permutation(positions)
        valid_positions.extend(shuffled[: round(len(positions) * test_size)].tolist())

    is_valid = np.zeros(len(df), dtype=bool)
    is_valid[valid_positions] = True
    return df[~is_valid], df[is_valid]

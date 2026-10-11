"""Phase 1: 상환 상태 정규화, 제외 컬럼 차단, 라벨 검증."""
import pandas as pd
import pytest

from training.scorecard.preprocess import (
    FORBIDDEN_COLUMNS,
    LABEL,
    STATUS_COLUMNS,
    normalize_repay_status,
    repay_status_distribution,
    validate_training_frame,
)


def test_normalize_repay_status_maps_no_delay_codes_to_zero():
    assert list(normalize_repay_status([-2, -1, 0, 1, 3, 9])) == [0, 0, 0, 1, 3, 9]


def test_normalize_repay_status_works_on_dataframe():
    df = pd.DataFrame({"PAY_0": [-2, 2], "PAY_2": [0, -1]})
    out = normalize_repay_status(df)
    assert out.to_dict("list") == {"PAY_0": [0, 2], "PAY_2": [0, 0]}


def test_status_columns_are_the_six_uci_months():
    assert STATUS_COLUMNS == ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


def test_repay_status_distribution_counts_each_value_per_column():
    df = pd.DataFrame({c: [0, 0, -2] for c in STATUS_COLUMNS})
    dist = repay_status_distribution(df)
    assert list(dist.columns) == ["column", "value", "count"]
    row = dist[(dist["column"] == "PAY_0") & (dist["value"] == 0)]
    assert row["count"].tolist() == [2]
    assert dist[dist["column"] == "PAY_0"]["count"].sum() == 3


def test_forbidden_columns_are_the_five_from_the_spec():
    assert FORBIDDEN_COLUMNS == {"SEX", "EDUCATION", "MARRIAGE", "AGE", "LIMIT_BAL"}


@pytest.mark.parametrize("col", sorted(FORBIDDEN_COLUMNS))
def test_training_frame_with_forbidden_column_raises(col):
    df = pd.DataFrame({"max_delinquency_months": [0, 1], col: [1, 2], LABEL: [0, 1]})
    with pytest.raises(ValueError, match=col):
        validate_training_frame(df)


@pytest.mark.parametrize("bad_label", [[0, 2], [0, -1], [0.5, 1]])
def test_label_must_be_zero_or_one(bad_label):
    df = pd.DataFrame({"max_delinquency_months": [0, 1], LABEL: bad_label})
    with pytest.raises(ValueError, match=LABEL):
        validate_training_frame(df)


def test_label_with_missing_value_raises():
    df = pd.DataFrame({"max_delinquency_months": [0, 1], LABEL: [0, None]})
    with pytest.raises(ValueError, match=LABEL):
        validate_training_frame(df)


def test_missing_label_column_raises():
    with pytest.raises(ValueError, match=LABEL):
        validate_training_frame(pd.DataFrame({"max_delinquency_months": [0, 1]}))


def test_valid_training_frame_passes():
    df = pd.DataFrame({"max_delinquency_months": [0, 1], LABEL: [0, 1]})
    validate_training_frame(df)

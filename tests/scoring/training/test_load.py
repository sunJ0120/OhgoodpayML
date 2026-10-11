"""Phase 1: UCI 컬럼 이름 복원 (네트워크 없이)."""
import pandas as pd
import pytest

from training.scorecard.load import UCI_COLUMN_BY_DESCRIPTION, rename_uci_columns
from training.scorecard.preprocess import LABEL


def _uci_frames():
    descriptions = list(UCI_COLUMN_BY_DESCRIPTION)
    x_cols = [f"X{i}" for i in range(1, len(descriptions) + 1)]
    variables = pd.DataFrame({
        "name": x_cols + ["Y"],
        "description": descriptions + ["default payment next month"],
    })
    X = pd.DataFrame([list(range(len(x_cols)))], columns=x_cols)
    y = pd.DataFrame({"Y": [1]})
    return X, y, variables


def test_uci_has_23_feature_columns():
    assert len(UCI_COLUMN_BY_DESCRIPTION) == 23


def test_rename_restores_original_names_and_label():
    X, y, variables = _uci_frames()
    out = rename_uci_columns(X, y, variables)
    assert LABEL in out.columns
    assert out[LABEL].tolist() == [1]
    for name in ["LIMIT_BAL", "PAY_0", "PAY_6", "BILL_AMT1", "PAY_AMT6", "AGE"]:
        assert name in out.columns
    assert len(out.columns) == 24


def test_unknown_description_raises():
    X, y, variables = _uci_frames()
    variables.loc[variables["name"] == "X1", "description"] = "SOMETHING_ELSE"
    with pytest.raises(ValueError, match="SOMETHING_ELSE"):
        rename_uci_columns(X, y, variables)

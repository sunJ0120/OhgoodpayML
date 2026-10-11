"""Phase 1: TO-BE 변수 5개를 손 계산값과 대조한다."""
import pandas as pd
import pytest

from training.scorecard.features import FEATURE_COLUMNS, build_features, build_training_frame
from training.scorecard.preprocess import FORBIDDEN_COLUMNS, LABEL

STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT = [f"PAY_AMT{i}" for i in range(1, 7)]


def _raw(rows):
    """rows: [(status[6], bill[6], pay_amt[6], label)] -> UCI 형식 raw DataFrame."""
    data = []
    for i, (st, bill, pay, label) in enumerate(rows):
        rec = {"LIMIT_BAL": 50_000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30 + i}
        rec.update(dict(zip(STATUS, st)))
        rec.update(dict(zip(BILL, bill)))
        rec.update(dict(zip(PAYAMT, pay)))
        rec[LABEL] = label
        data.append(rec)
    return pd.DataFrame(data)


@pytest.fixture()
def raw():
    return _raw([
        # 고객 0: 연체 2개월이 한 번, 청구 3개월 활동
        ([0, 0, 2, 0, -1, -2], [0, 0, 100, 200, 0, 300], [10, 20, 30, 40, 50, 60], 0),
        # 고객 1: 전부 정상 납부, 매달 청구, 납부 0
        ([-1, -1, -1, -1, -1, -1], [500] * 6, [0] * 6, 0),
        # 고객 2: 연체 중, 음수 청구액 포함
        ([1, 2, 3, 0, 0, 0], [-200, 0, 0, 0, 0, 400], [100] * 6, 1),
    ])


def test_feature_columns_are_the_five_to_be_variables():
    assert FEATURE_COLUMNS == [
        "max_delinquency_months",
        "on_time_ratio",
        "paid_amount_6m",
        "avg_billed_amount_6m",
        "active_months",
    ]


def test_features_match_hand_calculation(raw):
    f = build_features(raw)
    assert list(f.columns) == FEATURE_COLUMNS

    assert f["max_delinquency_months"].tolist() == [2, 0, 3]
    assert f["on_time_ratio"].tolist() == pytest.approx([5 / 6, 1.0, 3 / 6])
    assert f["paid_amount_6m"].tolist() == [210, 0, 600]
    assert f["avg_billed_amount_6m"].tolist() == pytest.approx([100.0, 500.0, 200 / 6])
    assert f["active_months"].tolist() == [3, 6, 1]


def test_features_keep_row_order_and_index(raw):
    raw.index = [10, 20, 30]
    assert build_features(raw).index.tolist() == [10, 20, 30]


def test_build_features_does_not_mutate_input(raw):
    before = raw.copy(deep=True)
    build_features(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_features_exclude_forbidden_columns(raw):
    assert not (set(build_features(raw).columns) & FORBIDDEN_COLUMNS)


def test_training_frame_is_features_plus_label_only(raw):
    t = build_training_frame(raw)
    assert list(t.columns) == FEATURE_COLUMNS + [LABEL]
    assert t[LABEL].tolist() == [0, 0, 1]
    assert not (set(t.columns) & FORBIDDEN_COLUMNS)


def test_training_frame_rejects_bad_label(raw):
    raw[LABEL] = [0, 1, 2]
    with pytest.raises(ValueError, match=LABEL):
        build_training_frame(raw)


def test_missing_raw_column_raises_with_its_name(raw):
    with pytest.raises(KeyError, match="PAY_AMT3"):
        build_features(raw.drop(columns=["PAY_AMT3"]))

"""Phase 5: 실제 UCI 데이터로 카드를 다시 만들어 커밋된 파일과 같은지 확인한다."""
import json

import pytest

from training.scorecard.run_uci import UCI_CARD_PATH, build_uci_card

pytestmark = pytest.mark.data


def _assert_close(actual, expected, path="card"):
    """구조는 정확히, 부동소수점은 플랫폼 차이를 감안해 근사로 비교한다."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for k in expected:
            _assert_close(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, float) or isinstance(actual, float):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9), path
    else:
        assert actual == expected, path


def test_rebuilt_card_equals_committed_card(tmp_path):
    out = tmp_path / "card.json"
    build_uci_card(out, created_at="2026-10-11")
    rebuilt = json.loads(out.read_text(encoding="utf-8"))
    committed = json.loads(UCI_CARD_PATH.read_text(encoding="utf-8"))
    _assert_close(rebuilt, committed)

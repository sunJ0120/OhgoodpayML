"""UCI 데이터로 전체 파이프라인을 돌려 검증용 스코어카드를 만든다.

    python -m training.scorecard.run_uci [created_at(YYYY-MM-DD)]

이 카드는 **파이프라인 검증용**이다 (purpose = pipeline-validation, currency = TWD).
오굿페이 운영에 쓰지 않으며, 런타임이 purpose 와 currency 검사로 막는다.
"""
import sys
from datetime import date
from pathlib import Path

from training.scorecard.binning import DEFAULT_KINDS, apply_woe, fit_woe_tables
from training.scorecard.export import draft_bands, export_scorecard
from training.scorecard.features import build_training_frame
from training.scorecard.fit import fit_logit
from training.scorecard.load import load_uci
from training.scorecard.preprocess import LABEL
from training.scorecard.scale import build_points, score_frame
from training.scorecard.split import stratified_split

REPO_ROOT = Path(__file__).resolve().parents[2]
UCI_CARD_PATH = REPO_ROOT / "models" / "scorecards" / "uci-validation-v1.json"

MODEL_VERSION = "uci-validation-v1"
DATA_SOURCE = "UCI Default of Credit Card Clients (CC BY 4.0), Yeh 2009"
CURRENCY = "TWD"
SEED = 42
TEST_SIZE = 0.3


def build_uci_card(out_path=UCI_CARD_PATH, created_at: str | None = None) -> Path:
    frame = build_training_frame(load_uci())
    train, _ = stratified_split(frame, LABEL, TEST_SIZE, SEED)

    tables = fit_woe_tables(train, DEFAULT_KINDS)
    fit = fit_logit(apply_woe(train, tables), train[LABEL])
    points = build_points(fit, tables)

    return export_scorecard(
        out_path,
        fit,
        tables,
        points,
        bands=draft_bands(score_frame(train, tables, points)),
        bands_status="draft",
        model_version=MODEL_VERSION,
        purpose="pipeline-validation",
        data_source=DATA_SOURCE,
        currency=CURRENCY,
        created_at=created_at or date.today().isoformat(),
    )


def main() -> None:
    created_at = sys.argv[1] if len(sys.argv) > 1 else None
    out = build_uci_card(created_at=created_at)
    print(f"스코어카드 생성: {out}")


if __name__ == "__main__":
    main()

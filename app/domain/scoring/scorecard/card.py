"""스코어카드 파일 로드·검증. 런타임은 이 파일만 읽는다 (training, pandas, numpy 를 쓰지 않는다).

구간 경계 규칙: lower < x <= upper. None(null) 은 무한대.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

REQUIRED_FIELDS = (
    "model_version", "purpose", "data_source", "currency",
    "scale", "intercept", "variables", "bands",
)
OPTIONAL_FIELDS = ("created_at", "bands_status")
STRING_FIELDS = ("model_version", "purpose", "data_source", "currency")
SCALE_FIELDS = ("base_score", "base_good_odds", "points_per_double")

# 점수에 쓰지 않는 변수. training.scorecard.preprocess.FORBIDDEN_COLUMNS 를 소문자로 옮긴 것이며,
# 어긋나지 않는지 tests/scoring/scorecard/test_card.py 가 확인한다.
FORBIDDEN_VARIABLES = frozenset({"sex", "education", "marriage", "age", "limit_bal"})
# 연장 계열 (소액후불결제의 리볼빙·결제 이월 금지와 충돌 가능성)
FORBIDDEN_SUBSTRINGS = ("extension", "연장")

PRODUCTION = "production"


class ScoreCardError(ValueError):
    """스코어카드 파일이 규칙에 맞지 않는다."""


class ScoreCardPurposeError(ScoreCardError):
    """purpose 가 production 이 아닌 카드를 운영 모드로 로드하려 했다."""


class CurrencyMismatchError(ScoreCardError):
    """카드 통화와 입력 통화가 다르다."""


@dataclass(frozen=True)
class Scale:
    base_score: float
    base_good_odds: float
    points_per_double: float


@dataclass(frozen=True)
class BinCard:
    lower: float | None
    upper: float | None
    bad_rate: float
    woe: float
    points: float


@dataclass(frozen=True)
class VariableCard:
    beta: float
    bins: tuple[BinCard, ...]


@dataclass(frozen=True)
class Band:
    name: str
    min_score: float


@dataclass(frozen=True)
class ScoreCard:
    model_version: str
    purpose: str
    data_source: str
    currency: str
    scale: Scale
    intercept: float
    variables: dict[str, VariableCard]
    bands: tuple[Band, ...]
    created_at: str | None = None
    bands_status: str | None = None

    def assert_currency(self, currency: str) -> None:
        if currency != self.currency:
            raise CurrencyMismatchError(
                f"카드 통화({self.currency})와 입력 통화({currency})가 다르다. 금액 구간이 어긋나므로 채점하지 않는다"
            )

    def to_dict(self) -> dict:
        out = {
            "model_version": self.model_version,
            "purpose": self.purpose,
            "data_source": self.data_source,
            "currency": self.currency,
        }
        if self.created_at is not None:
            out["created_at"] = self.created_at
        out["scale"] = {
            "base_score": self.scale.base_score,
            "base_good_odds": self.scale.base_good_odds,
            "points_per_double": self.scale.points_per_double,
        }
        out["intercept"] = self.intercept
        out["variables"] = {
            name: {
                "beta": var.beta,
                "bins": [
                    {"lower": b.lower, "upper": b.upper, "bad_rate": b.bad_rate, "woe": b.woe, "points": b.points}
                    for b in var.bins
                ],
            }
            for name, var in self.variables.items()
        }
        out["bands"] = [{"name": b.name, "min_score": b.min_score} for b in self.bands]
        if self.bands_status is not None:
            out["bands_status"] = self.bands_status
        return out


# ---- 파싱 ------------------------------------------------------------------
def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(value, where: str):
    if not _is_number(value):
        raise ScoreCardError(f"{where} 은(는) 유한한 숫자여야 한다: {value!r}")
    return value


def _text(value, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScoreCardError(f"{where} 은(는) 비어 있지 않은 문자열이어야 한다: {value!r}")
    return value


def _is_forbidden(name: str) -> bool:
    low = name.lower()
    return low in FORBIDDEN_VARIABLES or any(s in low for s in FORBIDDEN_SUBSTRINGS)


def _parse_bins(name: str, raw) -> tuple[BinCard, ...]:
    if not isinstance(raw, list) or not raw:
        raise ScoreCardError(f"variables.{name}.bins 는 비어 있지 않은 목록이어야 한다")

    bins = []
    for i, item in enumerate(raw):
        where = f"variables.{name}.bins[{i}]"
        if not isinstance(item, dict):
            raise ScoreCardError(f"{where} 는 객체여야 한다")
        for key in ("lower", "upper", "bad_rate", "woe", "points"):
            if key not in item:
                raise ScoreCardError(f"{where}.{key} 가 없다")
        lower, upper = item["lower"], item["upper"]
        if lower is not None:
            _number(lower, f"{where}.lower")
        if upper is not None:
            _number(upper, f"{where}.upper")
        if lower is not None and upper is not None and not lower < upper:
            raise ScoreCardError(f"{where}: lower 는 upper 보다 작아야 한다 ({lower} >= {upper})")
        bad_rate = _number(item["bad_rate"], f"{where}.bad_rate")
        if not 0 <= bad_rate <= 1:
            raise ScoreCardError(f"{where}.bad_rate 는 0~1 이어야 한다: {bad_rate}")
        bins.append(
            BinCard(
                lower=lower,
                upper=upper,
                bad_rate=bad_rate,
                woe=_number(item["woe"], f"{where}.woe"),
                points=_number(item["points"], f"{where}.points"),
            )
        )

    if bins[0].lower is not None:
        raise ScoreCardError(f"variables.{name}: 첫 구간의 lower 는 null 이어야 한다")
    if bins[-1].upper is not None:
        raise ScoreCardError(f"variables.{name}: 마지막 구간의 upper 는 null 이어야 한다")
    for i in range(len(bins) - 1):
        prev_upper, next_lower = bins[i].upper, bins[i + 1].lower
        if prev_upper is None or next_lower is None:
            raise ScoreCardError(f"variables.{name}: 중간 구간의 경계가 null 이다 (bins[{i}], bins[{i + 1}])")
        if prev_upper > next_lower:
            raise ScoreCardError(f"variables.{name}: 구간이 겹친다 (bins[{i}].upper={prev_upper} > bins[{i + 1}].lower={next_lower})")
        if prev_upper < next_lower:
            raise ScoreCardError(f"variables.{name}: 구간 사이에 빈 곳이 있다 (bins[{i}].upper={prev_upper} < bins[{i + 1}].lower={next_lower})")
    return tuple(bins)


def _parse_variables(raw) -> dict[str, VariableCard]:
    if not isinstance(raw, dict) or not raw:
        raise ScoreCardError("variables 가 비어 있다")
    out = {}
    for name, item in raw.items():
        if _is_forbidden(name):
            raise ScoreCardError(f"점수에 쓸 수 없는 금지된 변수가 있다: {name}")
        if not isinstance(item, dict) or "beta" not in item or "bins" not in item:
            raise ScoreCardError(f"variables.{name} 에는 beta 와 bins 가 필요하다")
        beta = _number(item["beta"], f"variables.{name}.beta")
        if beta >= 0:
            raise ScoreCardError(f"variables.{name}.beta 는 음수여야 한다 (WoE 가 클수록 안전): {beta}")
        out[name] = VariableCard(beta=beta, bins=_parse_bins(name, item["bins"]))
    return out


def _parse_bands(raw) -> tuple[Band, ...]:
    if not isinstance(raw, list) or not raw:
        raise ScoreCardError("bands 는 비어 있지 않은 목록이어야 한다")
    bands = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "name" not in item or "min_score" not in item:
            raise ScoreCardError(f"bands[{i}] 에는 name 과 min_score 가 필요하다")
        bands.append(Band(_text(item["name"], f"bands[{i}].name"), _number(item["min_score"], f"bands[{i}].min_score")))
    if len({b.name for b in bands}) != len(bands):
        raise ScoreCardError("bands 의 name 이 중복된다")
    if any(a.min_score <= b.min_score for a, b in zip(bands, bands[1:])):
        raise ScoreCardError("bands 는 min_score 가 큰 등급부터 엄격히 내림차순이어야 한다")
    return tuple(bands)


def parse_scorecard(data: dict, production: bool = False) -> ScoreCard:
    if not isinstance(data, dict):
        raise ScoreCardError("스코어카드는 JSON 객체여야 한다")
    for field in REQUIRED_FIELDS:
        if field not in data:
            raise ScoreCardError(f"필수 필드가 없다: {field}")
    unknown = sorted(set(data) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
    if unknown:
        raise ScoreCardError(f"알 수 없는 필드: {', '.join(unknown)}")

    texts = {f: _text(data[f], f) for f in STRING_FIELDS}

    scale_raw = data["scale"]
    if not isinstance(scale_raw, dict):
        raise ScoreCardError("scale 은 객체여야 한다")
    for f in SCALE_FIELDS:
        if f not in scale_raw:
            raise ScoreCardError(f"scale.{f} 가 없다")
    scale = Scale(**{f: _number(scale_raw[f], f"scale.{f}") for f in SCALE_FIELDS})
    if scale.base_good_odds <= 0 or scale.points_per_double <= 0:
        raise ScoreCardError("scale.base_good_odds 와 scale.points_per_double 은 0 보다 커야 한다")

    created_at = data.get("created_at")
    if created_at is not None:
        _text(created_at, "created_at")
    bands_status = data.get("bands_status")
    if bands_status is not None:
        _text(bands_status, "bands_status")

    card = ScoreCard(
        **texts,
        scale=scale,
        intercept=_number(data["intercept"], "intercept"),
        variables=_parse_variables(data["variables"]),
        bands=_parse_bands(data["bands"]),
        created_at=created_at,
        bands_status=bands_status,
    )
    if production and card.purpose != PRODUCTION:
        raise ScoreCardPurposeError(
            f"purpose 가 '{card.purpose}' 인 카드는 운영 모드로 로드할 수 없다 (production 만 허용)"
        )
    return card


def load_scorecard(path, production: bool = False) -> ScoreCard:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise ScoreCardError(f"스코어카드 파일을 읽을 수 없다: {path} ({e})") from e
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ScoreCardError(f"스코어카드가 올바른 JSON 이 아니다: {path} ({e})") from e
    return parse_scorecard(data, production=production)

"""평가 세트 파일 형식 검증 (Phase 4). 실제 모델은 부르지 않는다."""
import json
from pathlib import Path

import pytest

from eval.evalset import EvalSetError, load_evalset

ROOT = Path(__file__).resolve().parents[2]
SETS = ROOT / "eval" / "sets"
SEED = json.loads((ROOT / "docs/llm/seed/prompts_v1.json").read_text(encoding="utf-8"))

VALIDATION_KEYS = ["validate_mood", "validate_hobby", "validate_choose"]
ALL_KEYS = VALIDATION_KEYS + ["keyword_extract"]


def write(tmp_path, data) -> Path:
    p = tmp_path / "s.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


# ---------- 실제 평가 세트 파일 ----------

def test_every_planned_key_has_an_evalset():
    assert sorted(p.stem for p in SETS.glob("*.json")) == sorted(ALL_KEYS)


@pytest.mark.parametrize("key", VALIDATION_KEYS)
def test_validation_sets_have_about_twenty_cases_with_edge_categories(key):
    s = load_evalset(SETS / f"{key}.json")
    assert s.prompt_key == key and s.kind == "validation"
    assert len(s.cases) >= 20
    cats = {c.category for c in s.cases}
    assert {"normal", "edge_nonsense", "edge_empty", "edge_long"} <= cats
    assert any(c.expected.is_valid for c in s.cases) and any(not c.expected.is_valid for c in s.cases)


def test_empty_category_really_contains_empty_or_blank_input():
    for key in VALIDATION_KEYS:
        s = load_evalset(SETS / f"{key}.json")
        assert all(not c.input.strip() for c in s.cases if c.category == "edge_empty")
        assert any(not c.input.strip() for c in s.cases if c.category == "edge_empty")
        assert all(len(c.input) >= 120 for c in s.cases if c.category == "edge_long")


@pytest.mark.parametrize("key", ["validate_mood", "validate_choose"])
def test_expected_normalized_values_are_allowed_by_the_v1_schema_enum(key):
    allowed = set(SEED[key]["response_schema"]["properties"]["normalized"]["enum"])
    for c in load_evalset(SETS / f"{key}.json").cases:
        values = c.expected.normalized_options()
        if values is not None and c.expected.is_valid:
            assert set(values) <= allowed, (c.id, values)


def test_keyword_set_has_profiles_with_contexts():
    s = load_evalset(SETS / "keyword_extract.json")
    assert s.kind == "keyword" and len(s.cases) >= 10
    assert all(c.context is not None and c.context.credit_limit > 0 for c in s.cases)
    assert any(c.context.prev_keyword for c in s.cases)
    assert any(not c.context.hobby for c in s.cases)


@pytest.mark.parametrize("key", ALL_KEYS)
def test_case_ids_are_unique(key):
    ids = [c.id for c in load_evalset(SETS / f"{key}.json").cases]
    assert len(ids) == len(set(ids))


# ---------- 로더의 형식 검사 ----------

def good_case(**kw):
    c = {"id": "c1", "category": "normal", "input": "기분 좋아", "expected": {"is_valid": True, "normalized": "positive"}}
    c.update(kw)
    return c


def test_loader_accepts_minimal_valid_file(tmp_path):
    s = load_evalset(write(tmp_path, {"prompt_key": "validate_mood", "kind": "validation", "cases": [good_case()]}))
    assert s.cases[0].expected.normalized_options() == ["positive"]


def test_loader_normalized_list_means_any_of(tmp_path):
    c = good_case(expected={"is_valid": True, "normalized": ["넷플", "넷플릭스"]})
    s = load_evalset(write(tmp_path, {"prompt_key": "validate_hobby", "kind": "validation", "cases": [c]}))
    assert s.cases[0].expected.normalized_options() == ["넷플", "넷플릭스"]


@pytest.mark.parametrize("data", [
    {"prompt_key": "validate_mood", "kind": "validation", "cases": []},                                # 빈 세트
    {"prompt_key": "validate_mood", "kind": "nope", "cases": [good_case()]},                           # 알 수 없는 kind
    {"prompt_key": "validate_mood", "kind": "validation", "cases": [good_case(), good_case()]},         # id 중복
    {"prompt_key": "validate_mood", "kind": "validation", "cases": [good_case(expected=None)]},        # expected 없음
    {"prompt_key": "validate_mood", "kind": "validation", "cases": [good_case(category="weird")]},     # 알 수 없는 category
    {"prompt_key": "validate_mood", "kind": "validation", "cases": [{"id": "c1", "category": "normal"}]},  # input 없음
    {"prompt_key": "keyword_extract", "kind": "keyword", "cases": [{"id": "k1", "category": "normal", "input": ""}]},  # context 없음
])
def test_loader_rejects_malformed_files(tmp_path, data):
    with pytest.raises(EvalSetError):
        load_evalset(write(tmp_path, data))


def test_loader_rejects_non_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json", encoding="utf-8")
    with pytest.raises(EvalSetError):
        load_evalset(p)

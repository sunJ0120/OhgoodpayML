"""docs/llm/seed/prompts_v1.json 은 Phase 3 에서 BE 마이그레이션(version 1 INSERT)의 원본이 된다.
형식이 깨지면 BE 시드가 어긋나므로 여기서 잡는다.
"""
import json
import re
from pathlib import Path

import jsonschema
import pytest

SEED = json.loads((Path(__file__).resolve().parents[2] / "docs/llm/seed/prompts_v1.json").read_text(encoding="utf-8"))

EXPECTED_KEYS = {
    "validate_mood", "validate_hobby", "validate_choose", "validate_recommendation",
    "validate_re_recommendation", "validate_default",
    "chat_hobby_check_with_hobby", "chat_hobby_check_empty", "chat_choose", "chat_fallback",
    "keyword_extract", "summary_update",
}

# ChatService / RecommendService 가 렌더링 때 넘기는 변수
ALLOWED_VARS = {
    "user_name", "hobby", "mood", "ban_clause", "avoid_note",
}


def test_seed_has_exactly_the_contract_keys():
    assert set(SEED) == EXPECTED_KEYS


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_seed_entry_is_well_formed(key):
    entry = SEED[key]
    assert entry["version"] == 1
    assert entry["content"].strip()
    assert entry["model"]
    assert entry["change_note"]
    jsonschema.Draft202012Validator.check_schema(entry["response_schema"])
    # OpenAI strict 구조화 출력 조건
    schema = entry["response_schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


@pytest.mark.parametrize("key", sorted(EXPECTED_KEYS))
def test_seed_placeholders_are_known_variables(key):
    used = set(re.findall(r"\{\{\s*(\w+)\s*\}\}", SEED[key]["content"]))
    assert used <= ALLOWED_VARS, used - ALLOWED_VARS


@pytest.mark.parametrize("key", sorted(k for k in EXPECTED_KEYS if k.startswith("validate_")))
def test_validation_prompts_no_longer_use_text_protocol(key):
    content = SEED[key]["content"]
    # 예전 출력 프로토콜 줄("- VALID: …", "- INVALID: …")이 남아 있으면 안 된다.
    # ("…해당하면 VALID:" 같은 판단 기준 소제목은 허용)
    assert not re.search(r"^\s*-\s*(IN)?VALID\b[^\n]*:", content, re.M)
    assert "is_valid" in content

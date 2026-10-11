"""비교 표 출력과 CLI 안전장치(과금 방지)."""
import asyncio
import json

import pytest

from app.llm.models import PromptSpec
from app.llm.runner import LlmRunner
from eval import run_eval
from eval.evalset import EvalCase, EvalSet, Expected
from eval.metrics import evaluate, format_comparison
from eval_helpers import ScriptedClient, StepClock

SCHEMA = {"type": "object",
          "properties": {"is_valid": {"type": "boolean"}, "normalized": {"type": "string"}},
          "required": ["is_valid", "normalized"], "additionalProperties": False}


def small_set() -> EvalSet:
    return EvalSet(prompt_key="validate_mood", kind="validation", cases=[
        EvalCase(id=f"c{i}", category="normal" if i < 3 else "edge_nonsense", input="x",
                 expected=Expected(is_valid=True, normalized="positive")) for i in range(4)])


def prompt(version: int) -> PromptSpec:
    return PromptSpec(promptKey="validate_mood", promptVersion=version, content="규칙", responseSchema=SCHEMA, model="m")


def report(version, outputs, label):
    return asyncio.run(evaluate(small_set(), prompt(version), LlmRunner(ScriptedClient(*outputs), clock=StepClock(0.1)),
                                label=label, concurrency=1))


GOOD = {"is_valid": True, "normalized": "positive"}
BAD = {"is_valid": True, "normalized": "negative"}


def test_comparison_table_has_exact_percentages_and_deltas():
    a = report(1, [GOOD, GOOD, BAD, "junk"], "v1")          # 스키마 3/4=75%, 일치 2/4=50%
    b = report(2, [GOOD, GOOD, GOOD, GOOD], "v2")           # 스키마 4/4=100%, 일치 4/4=100%

    table = format_comparison([a, b])

    assert "| 스키마 통과율 | 75.0% (3/4) | 100.0% (4/4) | +25.0%p |" in table
    assert "| 기대 결과 일치율 | 50.0% (2/4) | 100.0% (4/4) | +50.0%p |" in table
    assert "| 일치율 [normal] | 66.7% (2/3) | 100.0% (3/3) | +33.3%p |" in table
    assert "| 일치율 [edge_nonsense] | 0.0% (0/1) | 100.0% (1/1) | +100.0%p |" in table
    assert "| 평균 지연(ms) | 100 | 100 | +0 |" in table
    assert "SCHEMA_INVALID×1" in table
    assert table.splitlines()[0].startswith("| 지표 | v1 | v2 | Δ |")


def test_single_report_has_no_delta_column():
    a = report(1, [GOOD, GOOD, BAD, GOOD], "v1")
    table = format_comparison([a])
    assert "| 지표 | v1 |" in table.splitlines()[0]
    assert "Δ" not in table


# ---------- CLI: 실수로 과금되지 않게 ----------

@pytest.fixture()
def no_real_client(monkeypatch):
    created = []

    class Boom:
        def __init__(self, *a, **k):
            created.append(1)
            raise AssertionError("실제 OpenAI 클라이언트가 만들어지면 안 된다")

    monkeypatch.setattr(run_eval, "OpenAiLlmClient", Boom)
    return created


def test_cli_without_real_flag_only_prints_plan(capsys, no_real_client):
    code = run_eval.main(["--key", "validate_mood", "--a", "seed"])

    out = capsys.readouterr().out
    assert code == 0 and no_real_client == []
    assert "호출 계획" in out and "20" in out and "--real" in out


def test_cli_real_flag_requires_env_confirmation(capsys, monkeypatch, no_real_client):
    monkeypatch.delenv("EVAL_ALLOW_REAL", raising=False)

    code = run_eval.main(["--key", "validate_mood", "--a", "seed", "--real"])

    assert code == 2 and no_real_client == []
    assert "EVAL_ALLOW_REAL" in capsys.readouterr().err


def test_cli_runs_with_injected_client_and_writes_results(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_ALLOW_REAL", "1")
    client = ScriptedClient(fallback=lambda system, user: {"is_valid": True, "normalized": "positive"})

    code = run_eval.main(["--key", "validate_mood", "--a", "seed", "--real", "--results-dir", str(tmp_path)],
                         client_factory=lambda: client)

    out = capsys.readouterr().out
    assert code == 0
    assert "| 스키마 통과율 | 100.0% (20/20) |" in out
    files = list(tmp_path.glob("validate_mood_*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text(encoding="utf-8"))
    assert saved["prompt_key"] == "validate_mood" and len(saved["reports"][0]["results"]) == 20


def test_cli_candidate_file_inherits_schema_from_seed(tmp_path):
    cand = tmp_path / "cand.json"
    cand.write_text(json.dumps({"version": 2, "content": "새 규칙", "model": "gpt-4o-mini"}), encoding="utf-8")

    spec = run_eval.load_prompt("validate_mood", str(cand))

    assert spec.prompt_version == 2 and spec.content == "새 규칙"
    assert spec.response_schema["required"] == ["is_valid", "normalized"]


def test_cli_seed_source_is_v1():
    spec = run_eval.load_prompt("validate_mood", "seed")
    assert spec.prompt_version == 1 and "분류기" in spec.content

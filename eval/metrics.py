"""평가 실행과 지표 계산, 비교 표 출력.

실제 모델 호출 여부는 넘겨받은 LlmRunner(클라이언트)가 정한다. 이 모듈은 어떤 클라이언트든 같은 방식으로 계산한다.
"""
import asyncio
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from app.domain.recommend.recommend_service import RecommendService
from app.llm.models import PromptSpec
from app.llm.runner import LlmRunner
from eval.evalset import EvalCase, EvalSet


@dataclass
class CaseResult:
    case_id: str
    category: str
    schema_valid: bool
    correct: bool
    detail: str = ""
    error_code: Optional[str] = None
    llm_calls: int = 1
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class EvalReport:
    label: str
    prompt_key: str
    prompt_version: int
    model: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def schema_ok(self) -> int:
        return sum(r.schema_valid for r in self.results)

    @property
    def correct(self) -> int:
        return sum(r.correct for r in self.results)

    @property
    def schema_pass_rate(self) -> float:
        return self.schema_ok / self.total if self.total else 0.0

    @property
    def correct_rate(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def errors(self) -> Counter:
        return Counter(r.error_code for r in self.results if r.error_code)

    @property
    def tokens_in(self) -> int:
        return sum(r.tokens_in for r in self.results)

    @property
    def tokens_out(self) -> int:
        return sum(r.tokens_out for r in self.results)

    @property
    def llm_calls(self) -> int:
        return sum(r.llm_calls for r in self.results)

    @property
    def avg_latency_ms(self) -> int:
        return round(sum(r.latency_ms for r in self.results) / self.total) if self.total else 0

    def by_category(self) -> dict[str, tuple[int, int]]:
        """카테고리 → (정답 수, 전체 수). 처음 등장한 순서를 유지한다."""
        out: dict[str, list[int]] = {}
        for r in self.results:
            c = out.setdefault(r.category, [0, 0])
            c[0] += r.correct
            c[1] += 1
        return {k: (v[0], v[1]) for k, v in out.items()}

    def to_dict(self) -> dict:
        return {
            "label": self.label, "prompt_key": self.prompt_key, "prompt_version": self.prompt_version,
            "model": self.model,
            "summary": {"total": self.total, "schema_ok": self.schema_ok, "correct": self.correct,
                        "schema_pass_rate": self.schema_pass_rate, "correct_rate": self.correct_rate,
                        "avg_latency_ms": self.avg_latency_ms, "tokens_in": self.tokens_in,
                        "tokens_out": self.tokens_out, "llm_calls": self.llm_calls,
                        "errors": dict(self.errors)},
            "results": [r.__dict__ for r in self.results],
        }


# ---------------------------------------------------------------- 실행

async def evaluate(evalset: EvalSet, prompt: PromptSpec, runner: LlmRunner, *,
                   label: str, concurrency: int = 4) -> EvalReport:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(case: EvalCase) -> CaseResult:
        async with sem:
            if evalset.kind == "keyword":
                return await _eval_keyword(case, prompt, runner)
            return await _eval_validation(case, prompt, runner)

    results = await asyncio.gather(*(one(c) for c in evalset.cases))   # gather 는 입력 순서를 유지한다
    return EvalReport(label=label, prompt_key=evalset.prompt_key, prompt_version=prompt.prompt_version,
                      model=prompt.model or "default", results=list(results))


async def _eval_validation(case: EvalCase, prompt: PromptSpec, runner: LlmRunner) -> CaseResult:
    res = await runner.run("main", prompt, {}, case.input)
    m = res.meta
    base = dict(case_id=case.id, category=case.category, llm_calls=1, latency_ms=m.latency_ms,
                tokens_in=m.tokens_in, tokens_out=m.tokens_out)
    if not res.ok:
        return CaseResult(schema_valid=False, correct=False, detail=m.error_code or "", error_code=m.error_code, **base)

    exp = case.expected
    got_valid, got_norm = res.result["is_valid"], res.result["normalized"]
    if got_valid != exp.is_valid:
        return CaseResult(schema_valid=True, correct=False, detail=f"is_valid {got_valid} != {exp.is_valid}", **base)
    options = exp.normalized_options()
    if exp.is_valid and options is not None and got_norm not in options:
        return CaseResult(schema_valid=True, correct=False, detail=f"normalized {got_norm!r} not in {options}", **base)
    return CaseResult(schema_valid=True, correct=True, **base)


async def _eval_keyword(case: EvalCase, prompt: PromptSpec, runner: LlmRunner) -> CaseResult:
    ctx = case.context
    outcome = await RecommendService(runner, None).generate_keywords(
        prompt, user_name="평가", hobby=ctx.hobby, mood=ctx.mood,
        credit_limit=ctx.credit_limit, balance=ctx.balance, prev_keyword=ctx.prev_keyword)
    calls = outcome.calls
    base = dict(case_id=case.id, category=case.category, llm_calls=len(calls),
                latency_ms=sum(c.latency_ms for c in calls),
                tokens_in=sum(c.tokens_in for c in calls), tokens_out=sum(c.tokens_out for c in calls))

    if outcome.keyword is None:
        code = next((c.error_code for c in reversed(calls) if c.error_code), None)
        return CaseResult(schema_valid=False, correct=False, detail=code or "", error_code=code, **base)

    problems = []
    if outcome.fallback:
        problems.append("fallback: 재시도 후에도 금지어와 겹쳐 안전 기본값 사용")
    lo, hi = (int(x) for x in outcome.price_range.split("-"))
    cap = min(ctx.credit_limit, ctx.balance)
    if not outcome.fallback and hi > cap:
        problems.append(f"price: max {hi} > min(한도, 잔액) {cap}")
    if not outcome.keyword.strip():
        problems.append("keyword 가 비어 있음")
    norm = RecommendService._normalize_kw
    banned = {norm(w) for w in (ctx.hobby, ctx.prev_keyword) if w}
    if not outcome.fallback and norm(outcome.keyword) in banned:
        problems.append(f"ban: {outcome.keyword!r} 가 금지어와 겹침")
    return CaseResult(schema_valid=True, correct=not problems, detail="; ".join(problems), **base)


# ---------------------------------------------------------------- 표

def _pct(n: int, d: int) -> str:
    return f"{n / d * 100:.1f}% ({n}/{d})" if d else "-"


def _delta_pct(a: tuple[int, int], b: tuple[int, int]) -> str:
    if not a[1] or not b[1]:
        return "-"
    return f"{(b[0] / b[1] - a[0] / a[1]) * 100:+.1f}%p"


def format_comparison(reports: list[EvalReport]) -> str:
    """1개면 단일 표, 2개 이상이면 마지막 − 처음을 Δ 열에 보여준다."""
    first, last = reports[0], reports[-1]
    compare = len(reports) > 1
    head = ["지표"] + [r.label for r in reports] + (["Δ"] if compare else [])
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]

    def row(name, cells, delta=""):
        lines.append("| " + " | ".join([name, *cells] + ([delta] if compare else [])) + " |")

    row("프롬프트 버전", [f"{r.prompt_key} v{r.prompt_version}" for r in reports])
    row("스키마 통과율", [_pct(r.schema_ok, r.total) for r in reports],
        _delta_pct((first.schema_ok, first.total), (last.schema_ok, last.total)))
    row("기대 결과 일치율", [_pct(r.correct, r.total) for r in reports],
        _delta_pct((first.correct, first.total), (last.correct, last.total)))

    cats = list(dict.fromkeys(c for r in reports for c in r.by_category()))
    for cat in cats:
        cells = [_pct(*r.by_category().get(cat, (0, 0))) for r in reports]
        row(f"일치율 [{cat}]", cells,
            _delta_pct(first.by_category().get(cat, (0, 0)), last.by_category().get(cat, (0, 0))))

    row("평균 지연(ms)", [str(r.avg_latency_ms) for r in reports], f"{last.avg_latency_ms - first.avg_latency_ms:+d}")
    row("토큰 in/out", [f"{r.tokens_in}/{r.tokens_out}" for r in reports])
    row("LLM 호출 수", [str(r.llm_calls) for r in reports], f"{last.llm_calls - first.llm_calls:+d}")
    row("오류 코드", [", ".join(f"{k}×{v}" for k, v in r.errors.items()) or "-" for r in reports])
    return "\n".join(lines)

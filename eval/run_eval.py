"""프롬프트 평가 CLI (CI 밖에서 수동으로만 실행한다).

    python -m eval.run_eval --key validate_mood --a seed --b eval/candidates/mood_v2.json --real

- --real 이 없으면 호출 계획만 출력하고 아무것도 호출하지 않는다.
- --real 을 쓰려면 환경변수 EVAL_ALLOW_REAL=1 도 필요하다 (실수로 과금되는 것을 막는 이중 안전장치).
- --a / --b 는 'seed'(docs/llm/seed/prompts_v1.json 의 v1) 또는 후보 프롬프트 JSON 파일:
    {"version": 2, "content": "...", "response_schema": {...}(생략 시 seed 것), "model": "..."(생략 시 seed 것)}
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from app.llm.client import LlmClient, OpenAiLlmClient
from app.llm.models import PromptSpec
from app.llm.runner import LlmRunner
from eval.evalset import EvalSetError, load_evalset
from eval.metrics import EvalReport, evaluate, format_comparison

ROOT = Path(__file__).resolve().parents[1]
SETS_DIR = ROOT / "eval" / "sets"
SEED_PATH = ROOT / "docs" / "llm" / "seed" / "prompts_v1.json"
DEFAULT_RESULTS_DIR = ROOT / "eval" / "results"


def load_prompt(key: str, source: str) -> PromptSpec:
    seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    if key not in seed:
        raise SystemExit(f"seed 에 없는 프롬프트 키: {key}")
    base = seed[key]
    if source == "seed":
        data = {"version": base["version"]}
    else:
        data = json.loads(Path(source).read_text(encoding="utf-8"))
        if "content" not in data:
            raise SystemExit(f"{source}: content 가 필요하다")
    return PromptSpec(
        promptKey=key,
        promptVersion=data.get("version", base["version"] + 1),
        content=data.get("content", base["content"]),
        responseSchema=data.get("response_schema", base["response_schema"]),
        model=data.get("model", base["model"]),
    )


def _label(source: str, spec: PromptSpec) -> str:
    return f"{'seed' if source == 'seed' else Path(source).stem} v{spec.prompt_version}"


def main(argv: Optional[list[str]] = None, client_factory: Optional[Callable[[], LlmClient]] = None) -> int:
    p = argparse.ArgumentParser(prog="eval.run_eval", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--key", required=True, help="평가할 프롬프트 키 (eval/sets/<key>.json)")
    p.add_argument("--a", default="seed", help="기준 프롬프트: 'seed' 또는 JSON 파일 (기본 seed=v1)")
    p.add_argument("--b", default=None, help="비교할 후보 프롬프트 JSON 파일")
    p.add_argument("--real", action="store_true", help="실제 모델을 호출한다 (과금 발생). EVAL_ALLOW_REAL=1 필요")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    args = p.parse_args(argv)

    try:
        evalset = load_evalset(SETS_DIR / f"{args.key}.json")
    except EvalSetError as e:
        print(f"평가 세트 오류: {e}", file=sys.stderr)
        return 2

    sources = [args.a] + ([args.b] if args.b else [])
    prompts = [(s, load_prompt(args.key, s)) for s in sources]

    n_cases = len(evalset.cases)
    per_case = "최대 2회(금지어 재시도 포함)" if evalset.kind == "keyword" else "1회"
    total = n_cases * len(prompts)
    models = sorted({spec.model or "default" for _, spec in prompts})
    print(f"호출 계획: key={args.key}, 케이스 {n_cases}개 × 프롬프트 {len(prompts)}개 = LLM 호출 {total}회"
          f" (케이스당 {per_case}), 모델 {', '.join(models)}")
    for s, spec in prompts:
        print(f"  - {_label(s, spec)}")

    if not args.real:
        print("실제 호출은 하지 않았습니다. 실행하려면 --real 과 환경변수 EVAL_ALLOW_REAL=1 이 필요합니다.")
        return 0
    if os.environ.get("EVAL_ALLOW_REAL") != "1":
        print("실제 모델 호출은 과금이 발생합니다. 확인했다면 EVAL_ALLOW_REAL=1 을 설정하고 다시 실행하세요.", file=sys.stderr)
        return 2

    runner = LlmRunner((client_factory or OpenAiLlmClient)())
    reports: list[EvalReport] = []
    for s, spec in prompts:
        reports.append(asyncio.run(evaluate(evalset, spec, runner, label=_label(s, spec), concurrency=args.concurrency)))

    print()
    print(format_comparison(reports))

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.key}_{datetime.now():%Y%m%d-%H%M%S}.json"
    path.write_text(json.dumps({"prompt_key": args.key, "created_at": datetime.now().isoformat(timespec="seconds"),
                                "reports": [r.to_dict() for r in reports]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

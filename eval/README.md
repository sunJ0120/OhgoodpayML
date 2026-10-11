# 프롬프트 평가 (CI 밖, 수동 실행)

새 프롬프트 버전을 올리기 전에, 고정 입력 세트로 이전 버전과 비교한다. 응답이 매번 달라지고 **과금이 발생**하므로 CI에는 넣지 않는다.
계산 로직(`eval/metrics.py`)과 세트 형식은 가짜 클라이언트로 `tests/eval/`에서 CI가 검증한다.

## 평가 세트 (`eval/sets/<prompt_key>.json`)

| 키 | 케이스 | 기대 결과 |
|---|---|---|
| `validate_mood` | 20 | `is_valid` + 기분 enum(`positive`/`neutral`/`negative`) |
| `validate_hobby` | 20 | `is_valid` + 축약 취미(여러 정답 허용, 생략 가능) |
| `validate_choose` | 20 | `is_valid` + `recommend` |
| `keyword_extract` | 14 | 정답 문자열이 아니라 **제약**: 금지어 미사용, 가격 상한(min(한도, 잔액)) 이내, 안전 기본값 폴백 아님 |

카테고리: `normal`, `edge_nonsense`(엉뚱한 답), `edge_empty`(빈 답), `edge_long`(120자 이상 긴 답), keyword는 `edge_low_balance` / `edge_no_hobby` / `edge_prev_keyword`.

## 실행

```powershell
# 1) 호출 계획만 출력 (아무것도 호출하지 않음)
.\.venv\Scripts\python.exe -m eval.run_eval --key validate_mood

# 2) 실제 모델 호출 — 과금 발생. 환경변수로 한 번 더 확인한다.
$env:EVAL_ALLOW_REAL = "1"
$env:OPENAI_API_KEY = "..."
.\.venv\Scripts\python.exe -m eval.run_eval --key validate_mood --a seed --real

# 3) 후보 버전과 비교 (seed v1 vs 후보)
.\.venv\Scripts\python.exe -m eval.run_eval --key validate_mood --a seed --b eval/candidates/validate_mood_v2.json --real
```

후보 프롬프트 파일: `{"version": 2, "content": "...", "response_schema": {...}, "model": "..."}`
(`response_schema`, `model`을 생략하면 seed v1 것을 쓴다.) 결과는 `eval/results/`에 JSON으로 저장된다(git 제외).

## 지표

- **스키마 통과율**: 응답이 응답 스키마를 만족한 비율 (타임아웃·LLM 오류 포함해 분모는 전체 케이스)
- **기대 결과 일치율**: 스키마 통과 + 기대 결과 일치. 스키마 실패는 불일치로 센다
- 카테고리별 일치율, 평균 지연, 토큰 in/out, LLM 호출 수, 오류 코드

## 새 프롬프트 버전을 올리는 절차

1. (BE) 새 버전을 INSERT 하는 **새** Flyway 마이그레이션 작성 (기존 파일은 수정하지 않는다)
2. (ML) 후보 내용을 JSON 파일로 두고 위 3)으로 이전 버전과 비교
3. 지표가 나빠지지 않으면 배포, 이후 `llm_call_log`의 `prompt_version`별 `valid`/`error_code` 비율로 확인

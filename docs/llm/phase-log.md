---
title: Phase별 작업 기록
parent: LLM 개선 ① 작업 기록
nav_order: 1
---

# Phase별 작업 기록

각 Phase가 끝날 때마다 결과를 남긴다. 테스트 수는 그 시점의 전체 기준이다. 개발 방식은 TDD(테스트 작성 → 실패 확인 → 최소 구현)이고, 실제 LLM을 부르는 테스트는 CI에 넣지 않았다.

| Phase | 레포 | 내용 | 상태 | 테스트 |
|---|---|---|---|---|
| 0 | ML+BE | AS-IS 조사(코드 수정 없음) | 완료 | — |
| 1 | 문서 | 계약 확정(`contract.md`), 결정 3건 + 질문 5건 | 완료 | — |
| 2 | ML | 호출기·구조화 출력·프롬프트 상수 제거·v1 시드 | 완료 | ML 324 통과 (이후 과금 차단 추가) |
| 3 | BE(+ML 문서) | Flyway/MySQL, 프롬프트 저장소, 호출 기록, 흐름 복원, 직접 호출 제거 | 완료 | BE 66 중 59 통과(7은 이전부터 실패), 신규 44 전부 통과 |
| 4 | ML | 평가 세트 4종, 평가 스크립트(과금 이중 안전장치) | 완료 (실모델 평가는 하지 않기로 함) | ML 361 통과 (`scoring` 255 + `llm` 70 + `eval` 36) |
| 5 | ML+BE | 문서 | 완료 | 변경 없음 |

---

## Phase 0. AS-IS 조사

### 발견
- BE 채팅은 Spring AI로 **OpenAI를 직접 호출**하고 있었고(`bb6cae7`에서 FastAPI 호출을 대체), ML `/ml/chat`은 호출하는 곳이 없었다.
- BE에 Flyway·MySQL이 없고 DB는 H2 인메모리였다. 프롬프트는 전부 코드 상수였다.
- `mood_check`는 입력을 무시하고 고정 문장을 내도록 한 프롬프트로 LLM을 부르고 있었다.
- 응답은 전부 문자열 접두어 파싱이었고, 실패하면 조용히 기본값/위장 응답이 나갔다.
- "인기 상품 배치(TOP 50/100)"는 코드에서 찾지 못했다.

### 결정
채팅 경로를 `/ml/chat`으로 통일(D1), Flyway+MySQL 도입(D2), `mood_check` 고정 문장(D3).

---

## Phase 1. 계약 확정

### 바꾼 파일
`docs/llm/contract.md` (신규)

### 내용
프롬프트 키 12개, `POST /ml/chat`·`/ml/chat/validation` 요청·응답, 오류 코드, 데이터 모델, 개인정보 원칙(원문 미저장).
지시서와 달라진 점: 단일 `meta` → `calls[]` 배열 + `prompts` 슬롯(Q4), 이전 키워드 `prevKeyword`(Q5).

---

## Phase 2. ML 레포

### 테스트
- 구현 전: `ModuleNotFoundError: app.llm`으로 전체 실패 → 구현 후 통과
- 실패 1건은 테스트 인자 오용(구현은 정상)이라 테스트를 고침. 시드 형식 테스트의 오탐 6건은 정규식을 좁힘

### 바꾼 파일
| 파일 | 변경 |
|---|---|
| `app/llm/{models,client,runner,deps}.py` | 프롬프트/호출 메타 모델, OpenAI `json_schema(strict)` 클라이언트, 템플릿 렌더링·호출·재검증 러너, DI |
| `app/domain/chat/chat_service.py`, `recommend/*_service.py` | 프롬프트를 요청으로 받아 호출 순서만 담당. 키워드 금지어 재시도·폴백 유지 |
| `app/schemas/chat/*` | `prompts`, `prevKeyword` 요청 / `keyword`, `result`, `calls`, `normalized` 응답 |
| `app/routers/chat.py` | 의존성 주입, 검증 응답에 `normalized`·`calls` |
| `app/services/narratives/{chat,validation}_prompter.py` | **삭제** (프롬프트 상수 제거) |
| `docs/llm/seed/prompts_v1.json` | 기존 프롬프트 12개를 v1로 보존(출력 형식만 JSON용으로 수정) |
| `requirements.txt` | `jsonschema` |

### 판단이 필요했던 것 → 결과
- `validate_choose`의 `normalized`가 `recommend`뿐 → D8로 유지
- 실제 OpenAI 미검증 → D13으로 실모델은 하지 않음

---

## Phase 3. BE 레포

### 테스트
- BE 신규 44개 전부 통과: 흐름 20, 에러 처리 3, 저장소 계약 3, 프롬프트 서비스 2, ML 클라이언트 6, **실제 MySQL 8.4 마이그레이션 7**, 설정 바인딩 3
- 전체 66개 중 실패 7개는 Phase 3 이전부터(`PayImmediately*` 6: 데이터 의존, `JwtTests` 1: 약한 키)

### 구현 중 드러난 문제와 처리
| 문제 | 처리 |
|---|---|
| Testcontainers 1.21.2가 Docker 29를 인식 못 함 | 1.21.4로 상향 (D11) |
| `@Table(schema="llm")`이 MySQL에서 무시되어 JPA가 테이블을 못 찾음 | 같은 DB에 테이블 배치 (D4) |
| `FastApiConfig`가 바인딩되지 않아 `localhost:80`으로 호출 | `spring.config.import`로 수정 + 테스트 (D10) |
| 슬라이스 테스트에 `JPAQueryFactory` 빈 없음 | 테스트에서 `QuerydslConfig` import |

### 바꾼 파일(요지)
`llm/*`(엔티티·저장소·DTO·서비스·ML 클라이언트), `chat/flow/ChatFlow`, `chat/session/*`, `ChatService` 재작성, Flyway 2개, `application.yml`/`fastapi.yml`/`build.gradle`/`docker-compose.yml`, 삭제: `LlmApiClient`, `PromptProvider`, `RecommendationService`, 네이버 직접 호출 코드, Spring AI 의존성.

### 수동 확인 (BE + MySQL 8.4 + Redis + ML)
전체 흐름, 무효 입력 2회 후 강제 진행, v2 추가 즉시 반영, 호출 기록을 확인했다. **OpenAI 호환 스텁**을 썼고, 실제 OpenAI 키는 크레딧이 없어 429로 거절됐다(과금 없음).
수동 확인 뒤 정리에서 **다른 프로젝트의 Docker 볼륨 2개를 잘못 삭제**했다(최종 정리 7장).

---

## Phase 4. 평가 세트 (ML)

### 테스트
- 36개 신규(세트 형식·계산·표·CLI 안전장치). 구현 전 수집 오류로 실패 확인
- 형식 테스트가 `validate_choose`의 "긴 답" 케이스(95자)가 기준(120자) 미달임을 잡아 데이터를 고침

### 바꾼 파일
`eval/{evalset,metrics,run_eval}.py`, `eval/sets/*.json`(mood 20, hobby 20, choose 20, keyword 14), `eval/README.md`, `tests/eval/*`, `tests/conftest.py`(전체 테스트에서 실제 OpenAI 차단), `recommend_service.py`(`fallback` 필드)

### 과금 방지
기본은 호출 계획만 출력. `--real` + `EVAL_ALLOW_REAL=1`이 모두 있어야 호출. **v1 실모델 평가는 하지 않기로 결정**(D13).

---

## Phase 5. 문서

`docs/llm/{index,overview,phase-log}.md`(ML), `docs/llm/README.md`(BE) 작성. 용어, AS-IS→TO-BE, 변경 절차, 결정 기록, 한계, 후속 작업을 정리했다.

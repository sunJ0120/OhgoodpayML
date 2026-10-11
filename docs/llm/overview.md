---
title: 최종 정리
parent: LLM 개선 ① 작업 기록
nav_order: 2
---

# LLM 개선 ① — 최종 정리

작성일 2026-10-11. 대상: OhgoodpayML(FastAPI), OhgoodpayBE(Spring Boot).

## 1. 한 줄 요약

채팅의 LLM 호출 경로를 **BE → ML(`/ml/chat`) 한 갈래로 통일**하고, 프롬프트는 BE의 DB(`prompt_version`)에서 버전으로 관리해 요청에 실어 보내며, ML은 **구조화 출력(JSON 스키마)으로 호출·재검증**한 뒤 호출 메타데이터를 돌려주고, BE가 그것을 `llm_call_log`에 저장한다.

## 2. 용어 정의

| 용어 | 정의 |
|---|---|
| **프롬프트 키** | 대화 단계별 프롬프트의 이름. 12개: `validate_{mood,hobby,choose,recommendation,re_recommendation,default}`, `chat_{hobby_check_with_hobby,hobby_check_empty,choose,fallback}`, `keyword_extract`, `summary_update` |
| **프롬프트 버전** | 같은 키 안에서 1부터 증가하는 정수. **한 번 저장한 버전은 수정하지 않는다** |
| **활성 버전** | 키별 `MAX(version)`. 되돌릴 때는 옛 내용을 복사해 **새 버전으로 추가**한다 |
| **템플릿** | 프롬프트 `content`. `{{user_name}}`, `{{hobby}}`, `{{mood}}`, `{{ban_clause}}`, `{{avoid_note}}` 자리표시자를 ML이 치환한다 |
| **응답 스키마** | 프롬프트 버전마다 짝지어진 JSON 스키마. OpenAI `json_schema`(strict)로 강제하고 ML이 `jsonschema`로 한 번 더 검증한다 |
| **슬롯** | 한 요청 안의 LLM 호출 역할. `main`(본 응답·검증·키워드), `summary`(대화 요약 갱신) |
| **호출 기록** | LLM 호출 1건의 메타데이터(`llm_call_log` 1행): 키·버전, 모델, 지연, 토큰, 검증 통과 여부, 오류 코드. **사용자 원문은 저장하지 않는다** |
| **오류 코드** | `TIMEOUT`, `SCHEMA_INVALID`, `LLM_ERROR`, `TEMPLATE_ERROR`(ML), `ML_ERROR`(BE가 ML에 닿지 못함) |
| **평가 세트** | 키별 고정 입력과 기대 결과. 새 버전을 올리기 전에 이전 버전과 비교한다(CI 밖, 수동) |

## 3. AS-IS → TO-BE

### 3-1. 호출 흐름

```
[BE] 현재 단계(flow) 확인 → 입력 검증을 위해 validate_* 활성 프롬프트 조회
  → POST /ml/chat/validation { prompts.main, inputMessage, flow, ... }
[ML] 템플릿 렌더링 → LLM(json_schema strict) → 스키마 재검증 → { isValid, normalized, calls[] }
[BE] 호출 기록 저장(llm_call_log) → 무효면 표준 안내문(2회까지), 유효하면 값 저장
  → 다음 단계 chat_* / keyword_extract + summary_update 활성 프롬프트 조회
  → POST /ml/chat { prompts.main, prompts.summary, mood, hobby, summary, prevKeyword, ... }
[ML] (추천 단계는 키워드 생성 → 금지어 검사 → 네이버 검색) → { message, products, keyword, summary, calls[] }
[BE] 호출 기록 저장 → 실패 시 errorCode 를 그대로 올림(상태는 진행하지 않음) / 성공 시 Redis 상태 갱신
```

`mood_check` 질문은 LLM 없이 BE가 고정 문장으로 낸다(ML 호출·기록 없음).

### 3-2. Phase 0 조사 결과와 이번 변경

| # | 항목 | AS-IS | TO-BE |
|---|---|---|---|
| 1 | 프롬프트 위치 | ML·BE 코드의 상수·f-string. 키별로 분산 | BE DB `prompt_version`. 코드 상수 제거(테스트로 검사) |
| 2 | 대화 단계 | ML `Flow` enum. BE는 Spring AI로 단계 없이 직접 호출 | BE `ChatFlow` 상태기계 + ML 호출 (원래 구조 복원) |
| 3 | 고정 문장 단계의 LLM 호출 | `mood_check`가 "입력 무시, 정해진 문장만" 프롬프트로 LLM 호출 | BE 코드 고정 문장, LLM 미호출 |
| 4 | 응답 파싱 | `VALID:`/`INVALID:`, `키워드:`/`가격대:`, `SEARCH_KEYWORD:` 문자열 파싱, 실패 시 조용히 기본값 | JSON 스키마 강제 + 재검증, 실패는 `errorCode` |
| 5 | 모델·클라이언트 | `gpt-4o-mini`, ML AsyncOpenAI·동기 OpenAI 2종, BE Spring AI | 모델은 프롬프트 버전이 정함(기본 `gpt-4o-mini`). BE Spring AI 제거 |
| 6 | Spring → FastAPI 계약 | 채팅 호출 DTO 없음(BE는 미호출) | `prompts` 슬롯, `prevKeyword` 요청 / `keyword`, `result`, `calls[]` 응답 ([계약]({{ site.baseurl }}/llm/contract)) |
| 7 | DB 마이그레이션 | 없음(H2 인메모리 + `ddl-auto: update`) | Flyway + MySQL. 프롬프트·기록 테이블만 Flyway 관리 |
| 8 | 대화 로그 | Redis 맥락만(1시간) | Redis 세션 상태 + 영구 `llm_call_log`(메타데이터만) |

### 3-3. 후속 작업용 조사 (Phase 0 2-2) — 현재 상태

| # | 항목 | 현재 상태 |
|---|---|---|
| 9 | 타임아웃 | BE→ML: connect 2s / read 25s(`fastapi.yml`, 이번에 실제로 바인딩되도록 수정). ML→OpenAI: 채팅 경로 **설정 없음**(SDK 기본), `call_llm_json`(대시보드 조언)만 20s. ML→네이버: 10s |
| 10 | 실패 분기·폴백 | LLM 실패: 이제 `errorCode`로 전달(위장 응답 제거). **네이버 실패: ML이 가짜 상품 10개("OO 관련 상품 n")를 정상 응답처럼 반환 — 그대로 남아 있음** |
| 11 | 인기 상품 배치 | BE `refactor/chatbot`·ML `main` 코드에 **없음**. "TOP 50/100" 기억은 이 코드에서 확인 못 함(다른 브랜치일 수 있으나 열어보지 않음) |
| 12 | 키워드 축소 | 예전 BE는 "3어절 이상이면 첫·끝 단어만 남기기 → LLM 재생성 최대 2회". **BE 직접 호출 제거로 함께 사라짐.** 지금은 ML의 금지어 충돌 1회 재시도 + 안전 기본값뿐 |
| 13 | Redis TTL | 세션 상태 1시간(`CacheSpec`). 만료되면 `start`부터 다시(고정 mood 질문). 고객 정보는 캐시 없이 DB 직접 조회 |

## 4. 프롬프트를 바꾸는 절차

1. **BE**: 새 버전을 `INSERT`하는 **새** Flyway 파일을 추가한다 (`src/main/resources/db/migration/V{YYYYMMDDHHMM}__설명.sql`).
   이미 적용된 파일은 절대 수정하지 않는다. 되돌릴 때도 옛 내용을 복사해 새 버전으로 추가한다.
   `response_schema`를 바꾸면 ML의 소비 코드(`result["…"]` 키)도 함께 바뀌어야 한다.
2. **ML**: 후보 내용을 JSON 파일로 두고 `python -m eval.run_eval --key <키> --a seed --b <후보.json>`으로 이전 버전과 비교한다 (`eval/README.md`).
   실모델 호출은 과금이 발생하므로 `--real`과 `EVAL_ALLOW_REAL=1`이 둘 다 있어야 한다. **이번 작업에서는 실행하지 않았다.**
3. **배포**: BE 배포 시 Flyway가 새 버전을 적용한다. 활성 버전 조회에 캐시가 없어 적용 즉시 반영된다.
4. **확인**: `llm_call_log`에서 `prompt_key`/`prompt_version`별 `valid`·`error_code` 비율을 본다.

```sql
SELECT prompt_key, prompt_version, COUNT(*) AS n, SUM(valid) AS ok,
       ROUND(AVG(latency_ms)) AS avg_ms, SUM(error_code IS NOT NULL) AS errors
FROM llm_call_log WHERE created_at >= NOW() - INTERVAL 1 DAY
GROUP BY prompt_key, prompt_version ORDER BY prompt_key, prompt_version;
```

## 5. 결정 기록

| # | 결정 | 이유·메모 |
|---|---|---|
| D1 | 채팅 LLM 호출은 BE 직접 호출(Spring AI)을 없애고 **ML `/ml/chat`으로 통일** | 사용자 결정. 원래 구조였고 `bb6cae7`에서 Spring AI로 바뀐 것. 프롬프트·기록을 한 경로에서 일관되게 다루기 위함 |
| D2 | BE에 **Flyway + MySQL 도입** | 사용자 결정. H2 인메모리는 재시작 시 프롬프트·기록이 사라짐 |
| D3 | `mood_check`는 **LLM 없이 BE 고정 문장** | 사용자 결정. 기존 프롬프트가 입력을 무시하고 정해진 문장만 내도록 되어 있었음 |
| D4 | 별도 `llm` 스키마를 **쓰지 않고 같은 DB에 테이블을 둔다** | 구현 중 변경. MySQL에서 schema=database인데 Hibernate MySQL 방언이 `@Table(schema)`를 무시해, Flyway는 `llm` DB에 만들고 JPA는 앱 DB에서 찾아 실패(재현됨). 별도 DB는 `CREATE DATABASE` 권한도 필요 |
| D5 | `prompts` 슬롯 + `calls[]` 배열(지시서의 단일 `meta` 대신) | 사용자 승인(Q4). `/ml/chat` 한 번에 LLM이 2~3회 호출됨 |
| D6 | 이전 키워드는 요약 정규식 대신 `prevKeyword` 필드 | 사용자 승인(Q5). 요약 "맨 끝 `키워드:`" 규칙 제거 |
| D7 | 기존 JPA 테이블은 Flyway로 옮기지 않음(`ddl-auto` 유지), `llm_call_log` 정리·`dash_advice`는 후속 | 사용자가 추천대로 진행(Q1~Q3) |
| D8 | `validate_choose`의 `normalized`는 `recommend`뿐 | 기존 프롬프트가 "추천 수락"만 VALID로 취급. 대시보드 선택 판별은 없었음 |
| D9 | `customerId`를 채팅 요청에서 **필수**로 | 이전 Spring AI 버전은 이름을 하드코딩. 단계 상태기계는 고객 정보가 필요 |
| D10 | `FastApiConfig`를 `spring.config.import: classpath:fastapi.yml`로 바인딩 | `@PropertySource`는 YAML을 못 읽어 base-url이 비고 `localhost:80`으로 호출됐음(수동 확인에서 발견). 대시보드 클라이언트도 같은 영향이었음 |
| D11 | Testcontainers를 1.21.4로 상향 | Boot 3.5.3 기본 1.21.2가 Docker Engine 29(최소 API 1.44)를 인식 못 함 |
| D12 | 검증 `normalized` 값을 세션에 저장(기분 enum, 축약 취미) | 원문 대신 정규화 값을 쓰면 키워드 프롬프트 입력이 안정됨 |
| D13 | **실모델 평가는 하지 않는다** | 사용자 결정(2026-10-11): 목적은 구조 개선이고 모델 성능 측정이 아님 |

## 6. 한계

- **실제 모델로 검증되지 않았다.** 기준선(v1)의 스키마 통과율·일치율 수치가 없다. OpenAI `strict` 모드가 12개 스키마를 모두 받아들이는지도 실제 API로는 확인하지 못했다(요청 형식은 스텁 서버로 `json_schema`·`strict=true`·`additionalProperties=false`까지만 확인).
- 수동 두 서버 확인은 **OpenAI 호환 스텁**을 ML 뒤에 붙여 수행했다. BE·MySQL·Redis·ML 경로와 호출 기록·버전 반영은 실제로 동작했지만 모델 응답은 고정값이다.
- **평가 세트가 작다**(키당 14~20건). 큰 회귀를 거르는 용도이지 통계적 결론을 내는 용도가 아니다. `validate_hobby`의 기대값은 여러 정답 허용·미검사 항목이 있다.
- **실모델 평가가 CI 밖**이라 프롬프트 변경의 품질 회귀는 사람이 스크립트를 돌려야 잡힌다.
- v1 프롬프트의 기분 enum(`positive/neutral/negative`) 규칙은 기존 프롬프트 끝에 붙인 한 줄이라 약하다.
- 활성 프롬프트 조회에 **캐시가 없다**(요청당 최대 2회 조회). 호출 기록 저장 실패는 대화를 막지 않도록 **로그만 남기고 삼킨다**.
- 상품은 3개만 반환하고 캐시하지 않는다. 이미지 프록시(CORS 우회)는 쓰지 않는다(이전 Spring AI 버전과 동일).
- `llm_call_log` **보관 기간 정리 작업이 없다**(후속).
- 두 레포 모두 작업 시점에 **커밋되지 않았다.** BE 전체 테스트 66개 중 실패 7개(`PayImmediately*` 6, `JwtTests` 1)는 이번 작업 이전부터의 문제다.

## 7. 후속 작업

| 작업 | 내용 | 비고 |
|---|---|---|
| **② 타임아웃·폴백** | ML→OpenAI 채팅 경로 타임아웃 설정, **네이버 실패 시 가짜 상품 반환 제거/정책 재정의**, BE가 `errorCode`별로 어떻게 응답할지(재시도·안내 문구) | 3-3의 9·10번 |
| **③ 인기 상품 배치** | 코드에서 찾지 못함. 기억의 TOP 50/100이 어느 브랜치/계획인지 확인 필요 | 3-3의 11번 |
| **④ 키워드 축소** | 직접 호출 제거로 사라진 축소 로직을 어디(ML/BE)에 어떻게 둘지 재설계. 위치 기반(첫·끝 단어)은 의미가 깨질 수 있음 | 3-3의 12번 |
| `dash_advice` 이관 | 대시보드 조언 프롬프트(`DashboardAdvicePrompt`, `call_llm_json`)를 같은 방식으로 | `/dash/advice` 경로, 키 추가 필요 |
| `llm_call_log` 정리 | 90일 보관 후 삭제 배치 | 계약 7장 |
| 기존 JPA 테이블 Flyway 이관 | 지금은 `ddl-auto: update` | D7 |
| 실모델 평가 | 필요해지면 `eval/README.md` 절차로 v1 기준선 측정 | D13으로 이번엔 제외 |
| 로컬 환경 | `redis-docker/redis.conf` 없음(compose redis 빌드 실패), `gradle-wrapper.jar` 없음(`.gitignore`의 `*.jar`) | 환경 정리 |
| **볼륨 삭제 사건** | 작업 중 정리 명령이 의도보다 넓어 다른 프로젝트의 미사용 Docker 볼륨 2개(`concert-ticketing-service_mysql_data`, `ohgoodfood-be_mysql_data`)를 삭제함. 복구 불가 | 해당 프로젝트에서 필요한 데이터였는지 확인 필요 |

## 8. 검증 현황

| 항목 | 결과 |
|---|---|
| ML 테스트 | 361 통과 = 기존 `tests/scoring` 255 + 신규 `tests/llm` 70(호출기·API·시드·과금 차단) + `tests/eval` 36(평가 세트·계산·CLI) |
| BE 테스트 | 66개 중 59 통과, 7 실패(이전부터). 신규 44(흐름 20, 마이그레이션 7 *실제 MySQL 8.4*, ML 클라이언트 6, 기타) 전부 통과 |
| 실제 서버 확인 | BE + MySQL 8.4 + Redis + ML(스텁 LLM): 전체 흐름·무효 입력 2회 후 강제 진행·v2 프롬프트 반영·호출 기록 확인 |
| 실모델 | 실행하지 않음 (D13) |

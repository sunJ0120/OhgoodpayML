---
title: BE ↔ ML 계약
parent: LLM 개선 ① 작업 기록
nav_order: 3
---

# LLM 개선 ① — BE ↔ ML 계약

> 상태: **Phase 1에서 확정, Phase 2~3 구현 중 일부 갱신** (6장 데이터 모델의 스키마 결정 등). 최종 결정 이력은 [최종 정리]({{ site.baseurl }}/llm/overview) 5장.
> 대상: OhgoodpayBE (Spring Boot), OhgoodpayML (FastAPI)

## 0. 확정된 결정

| # | 결정 | 근거 |
|---|---|---|
| D1 | 채팅 LLM 호출은 **BE 직접 호출(Spring AI)을 없애고 ML `/ml/chat`으로 통일**한다 | 사용자 결정. 원래 구조였음 (`bb6cae7`에서 FastAPI→Spring AI로 바뀜) |
| D2 | BE에 **Flyway + MySQL을 도입**한다 | 사용자 결정. 현재 H2 인메모리라 재시작 시 프롬프트·기록이 사라짐 |
| D3 | `mood_check`는 **LLM 호출 없이 BE 코드에서 고정 문장 출력**한다 | 사용자 결정. 현재 프롬프트가 "입력 무시, 정해진 문장만" 이었음 |

## 1. 용어

| 용어 | 정의 |
|---|---|
| 프롬프트 키 | 단계별 프롬프트 이름 (3장) |
| 프롬프트 버전 | 키 안에서 1부터 증가하는 정수. 저장 후 수정하지 않는다 |
| 활성 버전 | 키별 `MAX(version)`. 되돌릴 때는 옛 내용을 **새 버전으로 추가** |
| 응답 스키마 | 프롬프트 버전과 짝지어진 JSON 스키마. LLM 응답을 강제·검증 |
| 호출 기록 | LLM 호출 1건의 메타데이터 (`llm_call_log` 1행) |
| 평가 세트 | 키별 고정 입력·기대 결과 (Phase 4) |

## 2. 책임 경계

```
BE   : 대화 단계(flow) 상태기계, Redis 맥락, 프롬프트 조회(DB), 호출 기록 저장(DB), 프론트 통신
ML   : 템플릿 렌더링, LLM 호출(구조화 출력), 스키마 재검증, 메타데이터 생성, 네이버 쇼핑 호출
```

- ML은 DB에 접속하지 않고 프롬프트를 저장하지 않는다.
- BE의 `chat` 패키지는 다음을 제거한다: `LlmApiClient`(Spring AI 직접 호출), `PromptProvider`, `RecommendationService`의 문자열 파싱(`SEARCH_KEYWORD:`, `MAX_PRICE:`), 키워드 축소·재생성 로직(후속 작업 ④에서 ML/BE 어디에 둘지 재결정).
- BE는 원래 구조(`03dc557^`의 `FlowProcessor` / `FlowConfiguration` / `ChatCacheService`)를 **참고해 복원**한다. 그대로 복사하지 않고 테스트부터 쓴다.
- `spring-ai-*` 의존성은 채팅에서 쓰지 않게 되면 제거한다.

## 3. 프롬프트 키

단계(flow) 전이는 원래 구조 기준: `start → mood_check → hobby_check → choose → recommendation → re-recommendation(반복)`.

| 키 | 용도 | 스키마(result) | 비고 |
|---|---|---|---|
| `validate_mood` | mood_check 입력 검증 | `{is_valid: bool, normalized: string}` | `normalized`는 `positive` / `neutral` / `negative` (무효면 `""`) |
| `validate_hobby` | hobby_check 입력 검증 | `{is_valid, normalized}` | `normalized`는 2~4어절 명사형 취미, 없다고 답하면 `취미 없음` |
| `validate_choose` | choose 입력 검증 | `{is_valid, normalized}` | 기존 프롬프트가 "추천 수락"만 VALID로 취급 → `normalized`는 `recommend` (대시보드 선택 판별은 현재 없음) |
| `validate_recommendation` | recommendation 입력 검증 | `{is_valid, normalized}` | `recommend`(현재 추천 수락) / `other`(다른 상품 요청) |
| `validate_re_recommendation` | re-recommendation 입력 검증 | `{is_valid, normalized}` | `other` |
| `validate_default` | 정의되지 않은 flow | `{is_valid, normalized}` | `""` |
| `chat_hobby_check_with_hobby` | 기존 취미가 있을 때 응답 | `{message: string}` | 변수: `user_name`, `hobby` |
| `chat_hobby_check_empty` | 취미가 없을 때 응답 | `{message}` | 변수: `user_name` |
| `chat_choose` | 추천 vs 대시보드 선택 유도 | `{message}` | 변수: `user_name` |
| `chat_fallback` | 허용되지 않은 flow | `{message}` | 현재 코드의 `else` 분기. `mood_check` 프롬프트로 대체하던 것을 고정 문장으로 |
| `keyword_extract` | 추천 키워드·가격대 | `{keyword: string, price_min: int, price_max: int}` | 변수: `hobby`, `mood`, `credit_limit`, `balance`, `ban_clause`, `avoid_note` |
| `summary_update` | 대화 요약 갱신 | `{summary: string}` | 변수: 기존 요약, 신규 대화, `flow`, `keyword` |

- `chat_mood_check`는 **키가 없다.** D3에 따라 BE 코드의 고정 문장이다 (아래 5장).
- 대시보드 조언(`dash_advice`)은 **이번 범위에서 제외**한다. 호출 경로가 달라(`/dash/advice`, BE가 스냅샷만 전달) 채팅 완료 후 같은 방식으로 후속 적용한다. → **확인 필요 (결정 Q3)**
- `validate_*`는 기존 `VALID:` / `INVALID:` 문자열 파싱을 JSON 스키마로 대체한다. 기존 `_canonical_invalid` 문구는 **BE/ML 코드의 표준 안내문**으로 유지한다 (LLM이 쓴 문구를 쓰지 않음).
- `summary_update`의 "맨 끝 `키워드: XXX`" 규칙은 `keyword_extract`가 구조화되면서 요약 문자열에 의존할 이유가 없어진다. 이전 키워드는 **BE가 Redis에 별도 필드로 저장**해 요청 변수로 전달한다 (`prev_keyword`). 요약 본문에서 정규식으로 꺼내는 코드를 제거한다.

## 4. 요청·응답 계약

### 4-1. 설계 조정 (지시서 3-1 대비)

실제 코드는 `/ml/chat` 한 번에 **LLM 호출이 최대 2~3회** 일어난다 (본 응답 + 요약 갱신, 추천 단계는 키워드 + 요약, 금지어 충돌 시 키워드 1회 재시도). 그래서 지시서의 단일 `meta`를 다음처럼 바꾼다.

1. 요청에 프롬프트를 **이름 있는 슬롯(`prompts`)** 으로 싣는다.
2. 프롬프트 본문은 `{{변수}}` 자리표시자가 있는 **템플릿**이다. 렌더링은 ML이 한다 (`variables`).
3. 응답의 `meta`는 **`calls` 배열**이다. 호출 1건당 1개.

### 4-2. `POST /ml/chat`

요청 (camelCase, 기존 `BasicChatRequest` 별칭 방식 유지):

```json
{
  "sessionId": "s-123",
  "flow": "choose",
  "inputMessage": "요즘 러닝에 빠졌어",
  "customerInfo": { "customerId": 1, "name": "오선정", "creditLimit": 300000 },
  "mood": "positive",
  "hobby": "러닝",
  "balance": 120000,
  "summary": "",
  "prevKeyword": null,
  "prompts": {
    "main":    { "promptKey": "chat_choose",    "promptVersion": 1, "content": "...{{user_name}}...", "responseSchema": { "type": "object", "properties": { "message": { "type": "string" } }, "required": ["message"], "additionalProperties": false }, "model": "gpt-4o-mini" },
    "summary": { "promptKey": "summary_update", "promptVersion": 1, "content": "...", "responseSchema": { } , "model": "gpt-4o-mini" }
  }
}
```

- `flow`가 `recommendation` / `re-recommendation`이면 `prompts.main` 자리에 `keyword_extract`를 싣는다.
- 요청에 `variables` 필드는 **없다.** ML이 요청 필드에서 직접 조립해 템플릿에 치환한다. 사용 가능한 변수는 아래뿐이고, 템플릿이 다른 이름을 쓰면 `TEMPLATE_ERROR`다.
  - 일반 단계(`chat_*`): `user_name`(=`customerInfo.name`), `hobby`
  - `keyword_extract`: `user_name`, `hobby`, `mood`, `ban_clause`, `avoid_note`
  - `summary_update`, `validate_*`: 변수 없음 (입력은 user 메시지로만 전달)
- `ban_clause`(금지 키워드 안내)와 `avoid_note`(금지어 충돌 시 재시도 안내)는 **ML이 만드는 문자열 조각**이다. 취미·`prevKeyword`가 있으면 채우고 없으면 빈 문자열이다.
- 프롬프트 `content`는 **시스템 메시지**이고, 사용자 데이터(입력 문장, 기존 요약, 취미·기분·한도)는 ML이 user 메시지로 조립한다. 지시문은 모두 프롬프트(DB)에 있고, 코드에는 데이터 블록 형식만 남는다.
- `prompts.*` 중 하나라도 필수 필드(`promptKey`, `promptVersion`, `content`, `responseSchema`)가 없으면 **422**.

응답:

```json
{
  "sessionId": "s-123",
  "flow": "choose",
  "message": "오 새 취미 멋지다 😆 🎁 추천? 📊 대시보드?",
  "newHobby": "요즘 러닝에 빠졌어",
  "products": [],
  "summary": "...",
  "keyword": null,
  "result": { "main": { "message": "..." }, "summary": { "summary": "..." } },
  "calls": [
    { "slot": "main",    "promptKey": "chat_choose",    "promptVersion": 1, "model": "gpt-4o-mini",
      "latencyMs": 812, "tokensIn": 640, "tokensOut": 41, "valid": true, "errorCode": null },
    { "slot": "summary", "promptKey": "summary_update", "promptVersion": 1, "model": "gpt-4o-mini",
      "latencyMs": 530, "tokensIn": 410, "tokensOut": 88, "valid": true, "errorCode": null }
  ]
}
```

- 기존 필드(`message`, `newHobby`, `products`, `summary`, `flow`)는 **유지**해서 프론트·BE DTO 변경을 최소화한다. `keyword`, `result`, `calls`가 추가된다.
- `keyword`: 추천 단계에서 **실제로 검색에 쓴 키워드**다 (금지어 충돌 시 대체된 안전 키워드 포함). BE는 이 값을 Redis에 저장했다가 다음 요청의 `prevKeyword`로 보낸다. 추천 단계가 아니거나 키워드 호출이 실패하면 `null`.
- 추천 단계의 `message`는 성공 시 `""`(기존 동작과 동일), 키워드 호출 실패 시 `null`이다.
- **`main` 호출이 `valid=false`면**: HTTP 200으로 응답하되 `message`는 `null`, 실패 호출의 `errorCode`를 `calls`에 담는다. 지금처럼 "죄송해요…"를 정상 메시지로 위장하지 않는다. BE는 `errorCode`를 `LlmServerException(errorCode)`로 올린다 (폴백 정책은 작업 ②).
- **`summary` 호출이 실패하면**: 기존 요약을 그대로 돌려주고 `calls`에 실패를 기록한다 (현재 동작 유지, 대화 자체는 막지 않음).
- `errorCode` 값: `TIMEOUT`, `SCHEMA_INVALID`, `LLM_ERROR`, `TEMPLATE_ERROR`(템플릿 변수 누락 — 프롬프트 등록 실수이며 LLM은 호출하지 않음). 이번 작업에서 타임아웃 **값**은 바꾸지 않는다.
- 요청에 `prompts.main` 필수 필드(`promptKey`, `promptVersion`, `content`, `responseSchema`)가 없으면 HTTP 422. `prompts.summary`는 생략할 수 있고, 생략하면 요약 갱신 호출을 하지 않는다.
- 요약 갱신은 본 응답이 성공한 경우에만 호출한다.

### 4-3. `POST /ml/chat/validation`

요청에 `prompts.main`(키 `validate_*`)을 싣고, 응답은 다음과 같다.

```json
{
  "customerId": 1, "sessionId": "s-123", "flow": "mood_check", "inputMessage": "...",
  "isValid": true, "normalized": "positive", "message": "",
  "calls": [ { "slot": "main", "promptKey": "validate_mood", "promptVersion": 1, "...": "..." } ]
}
```

- `isValid=false`일 때 `message`는 코드의 표준 안내문(`_canonical_invalid`). LLM 문구를 쓰지 않는다.
- LLM 호출 자체가 실패(`valid=false`)하면 `isValid=false`로 위장하지 않고 `errorCode`를 `calls`에 담아 BE가 구분할 수 있게 한다. 지금은 LLM 오류도 "다시 입력해줘"로 보이는 문제가 있다.
- 기분 저장: 현재 BE는 `validated.getInputMessage()`(원문)를 mood로 저장한다. 이후 **`normalized`(enum)를 저장**한다. 키워드 프롬프트의 `mood`도 enum이 된다.

## 5. 고정 문장 단계 (D3)

- `mood_check`(= `start` → `mood_check` 전이 시 질문)는 BE가 코드 상수로 응답하고 ML을 부르지 않는다.
  - 문장: `"추천 플로우 시작할게, {name}야! 오늘 기분은 어때? (좋아/보통/별로) 😊"` (기존 옵션 1/2 중 하나로 고정. 최종 문구는 구현 시 확정)
  - LLM 호출이 없으므로 `llm_call_log`에 기록하지 않는다. 요약 갱신도 건너뛴다.
- `chat_hobby_check_empty`(첫 취미 수집)도 거의 고정 문장이지만 **기분 공감 한 줄이 입력에 따라 달라지므로 LLM 유지**한다. 필요하면 후속에서 재검토.

## 6. 데이터 모델 (BE · MySQL)

> **변경 (Phase 3 구현 중 확정)**: 별도 `llm` 스키마를 만들지 않고 **애플리케이션 DB에 테이블을 함께 둔다.**
> 이유: MySQL에서 schema는 database와 같고, Hibernate MySQL 방언은 `@Table(schema=…)`를 무시한다.
> 그 결과 Flyway는 `llm` DB에 테이블을 만들지만 JPA는 애플리케이션 DB에서 찾아 실패했다(실제 재현). 별도 DB는 `CREATE DATABASE` 권한도 요구한다.

- 테이블 `prompt_version`, `llm_call_log`는 지시서 3-2와 같다. 아래만 추가했다.
  - `prompt_version.response_schema`는 MySQL `JSON` 타입.
  - `llm_call_log.slot`(varchar): 한 요청의 `main`/`summary` 구분.
  - `llm_call_log.request_id`(varchar): 같은 `/ml/chat` 응답에서 나온 호출들을 묶음.
  - 인덱스: `prompt_version(prompt_key, version)` 유니크, `llm_call_log(created_at)`, `(prompt_key, prompt_version)`, `(conversation_id)`.
- `llm_call_log.flow_step`: 검증 호출은 **사용자가 답한 단계**, 채팅 호출은 **생성한 다음 단계**의 flow 값이다. 같은 값이라도 `prompt_key`(`validate_*` vs `chat_*`)로 구분한다.
- 프롬프트 저장소는 `JpaRepository`가 아니라 `Repository`를 상속하고 **조회 메서드만** 둔다. 엔티티는 `@Immutable`이고 setter가 없다. 새 버전은 Flyway INSERT로만 추가한다.
- Flyway 파일: `src/main/resources/db/migration/V{YYYYMMDDHHMM}__설명.sql`. DDL(`V202610111000`)과 프롬프트 v1 시드(`V202610111010`)는 파일을 나눴다.
  시드는 ML 레포 `docs/llm/seed/prompts_v1.json`에서 생성했다 (이미 적용된 마이그레이션은 수정하지 않는다).
- **이미 JPA 테이블이 있는 DB라서 `spring.flyway.baseline-on-migrate=true`가 필요하다.** (히스토리 테이블이 없는 비어 있지 않은 DB에서 첫 실행이 실패하는 것을 막는다. Testcontainers 테스트가 이 경로를 검증한다.)
- 테스트 DB: 실제 MySQL 8.4(Testcontainers). 레거시 `@SpringBootTest`/`@DataJpaTest`는 `src/test/resources/application.properties`로 H2를 쓴다. Docker Engine 29 호환을 위해 Testcontainers를 1.21.4로 올렸다(Boot 3.5.3 기본 1.21.2는 Docker 29를 인식하지 못함).
- 배포 시 필요한 것: `DB_URL`, `DB_USERNAME`, `DB_PASSWORD` 환경변수. DB 계정에 `CREATE`/`INSERT`/`SELECT`/`ALTER`(Flyway 히스토리 테이블 포함) 권한.

## 7. 개인정보와 보관

- **사용자 원문은 `llm_call_log`에 저장하지 않는다.** 저장하는 건 메타데이터뿐이다 (키, 버전, 모델, 지연, 토큰, 검증 여부, 오류 코드).
- 보관 기간: `llm_call_log` **90일** 후 삭제하는 정리 작업은 이번 범위에서는 만들지 않고, 문서의 후속 작업으로 남긴다. (→ Q2)
- BE 로그의 `System.out.println(">>> DEBUG OUT ..." + json)`처럼 요청 본문을 찍는 코드는 채팅 경로에서는 만들지 않는다.

## 8. 프롬프트 조회 캐시

- **처음에는 캐시하지 않는다.** 요청당 `MAX(version)` 조회 1~2회는 `(prompt_key, version)` 인덱스로 충분하다. 병목이 확인되면 TTL 캐시를 추가한다 (버전은 추가만 되므로 TTL 만료로 충분).

## 9. 질문과 답변 (모두 정리됨)

Q1~Q3은 추천대로 진행(이번 범위 밖), Q4·Q5는 승인되어 이 문서의 4장에 반영되었다.

| Q | 질문 | 추천 |
|---|---|---|
| Q1 | MySQL 전환 시 기존 JPA 테이블(고객, 결제, 쇼츠 등)도 Flyway로 옮길까? | 이번엔 `prompt_version`/`llm_call_log`만 Flyway로 관리. 기존 테이블은 건드리지 않고 `ddl-auto`를 유지 (범위 밖) |
| Q2 | `llm_call_log` 90일 보관 정리를 이번에 만들까? | 후속으로 |
| Q3 | `dash_advice`를 이번 범위에 넣을까? | 후속으로 (채팅 완료 후) |
| Q4 | 요청·응답에 `prompts` 슬롯과 `calls` 배열을 쓰는 설계 조정(4-1)을 승인하는가? | 승인 (단일 `meta`로는 한 요청의 2~3회 호출을 기록할 수 없음) |
| Q5 | 요청 필드에서 `prevKeyword`를 새로 추가해 요약 문자열 정규식 의존을 없애는 것을 승인하는가? | 승인 |

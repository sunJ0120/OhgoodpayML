---
title: LLM 개선 ① 작업 기록
nav_order: 4
has_children: true
has_toc: false
---

# LLM 개선 ① — 프롬프트 버전 관리 · 호출 기록 · 구조화 출력

프롬프트를 코드에서 꺼내 **버전으로 관리**하고, 모든 LLM 호출에 **어떤 프롬프트 버전으로 응답했는지 기록**하고, 응답 형식을 **스키마로 강제·검증**하는 구조 개선의 기록이다.

- [Phase별 작업 기록]({{ site.baseurl }}/llm/phase-log) — 각 Phase 종료 시 결과, 바꾼 파일, 판단이 필요했던 지점
- [최종 정리]({{ site.baseurl }}/llm/overview) — 용어, AS-IS → TO-BE, 프롬프트를 바꾸는 절차, 결정 기록, 한계, 후속 작업
- [BE ↔ ML 계약]({{ site.baseurl }}/llm/contract) — 요청·응답 필드, 프롬프트 키, 오류 코드, 데이터 모델
- 평가 세트 사용법: 레포의 `eval/README.md`
- BE 쪽 정리: OhgoodpayBE 레포의 `docs/llm/README.md`

> **목적은 구조 개선이다.** 프롬프트 문구의 품질을 올리거나 실제 모델로 성능을 재는 작업은 하지 않았다.
> v1 프롬프트는 기존 코드의 문구를 그대로 옮겼고(출력 형식만 JSON 스키마용으로 수정), **실제 모델 평가는 실행하지 않았다.**

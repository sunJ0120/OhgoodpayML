---
title: 오굿점수 v2 작업 기록
nav_order: 3
has_children: true
has_toc: false
---

# 오굿점수 v2 — 로지스틱 회귀 스코어카드 전환

오굿점수를 **사람이 정한 가중치**에서 **데이터로 추정한 배점표**로 바꾸는 작업의 기록이다.

- [Phase별 작업 기록]({{ site.baseurl }}/scoring/phase-log) — 각 Phase 종료 시 결과, 바꾼 파일, 판단이 필요한 지점을 남긴다
- [오굿점수 v2 스코어카드]({{ site.baseurl }}/scoring/scorecard-v2) — 최종 정리 문서 (용어, AS-IS/TO-BE, 데이터 결정 기록, 결과, 한계, 결정이 필요한 항목, 후속 과제)

> **운영 `/score`는 아직 교체하지 않았다.** 새 엔진과 학습 파이프라인을 만들고 UCI 데이터로 파이프라인이 맞게 동작하는지 검증하는 것이 이번 범위다.

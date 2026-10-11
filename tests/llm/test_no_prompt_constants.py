"""프롬프트 본문이 ML 코드에 상수로 남아 있지 않은지 검사한다 (프롬프트는 BE가 DB에서 내려준다).

대시보드 조언(dash_advice)은 이번 범위에서 제외하므로 검사하지 않는다.
"""
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"

REMOVED_FILES = [
    "services/narratives/chat_prompter.py",
    "services/narratives/validation_prompter.py",
]

# 이관된 v1 프롬프트의 고유 문구
MARKERS = [
    "【IDENTITY】",
    "OVERRIDE RULE",
    "당신은 한국어 사용자 입력을 검증하는",
    "너는 쇼핑 추천 전문가",
    "대화 요약 전문가",
    "UNIVERSAL CONVERSION RULES",
]


def test_prompter_modules_are_removed():
    for rel in REMOVED_FILES:
        assert not (APP / rel).exists(), rel


def test_no_prompt_text_in_app_code():
    offenders = []
    for path in APP.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for m in MARKERS:
            if m in text:
                offenders.append(f"{path.relative_to(APP)}: {m}")
    assert offenders == []

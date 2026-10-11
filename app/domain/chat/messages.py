"""사용자에게 보여주는 표준 안내문. LLM이 쓴 문구가 아니라 코드의 고정 문구를 쓴다."""

_CANONICAL_INVALID = {
    "mood_check": "기분을 파악하지 못했어ㅠㅠ 다시 입력해줘!",
    "hobby_check": "취미/관심사를 알아듣지 못했어ㅠㅠ 다시 입력해줘!",
    "choose": "선택을 알아듣지 못했어ㅠㅠ 둘 중 하나를 선택해줘!",
    "recommendation": "추천을 알아듣지 못했어ㅠㅠ 둘 중 하나를 선택해줘!",
    "re-recommendation": "재추천 의도를 알아듣지 못했어ㅠㅠ \"다른 거/마음에 안 들어\" 같이 말해줘!",
}
_DEFAULT_INVALID = "입력을 알아듣지 못했어ㅠㅠ 다시 입력해줘!"


def canonical_invalid(flow: str) -> str:
    return _CANONICAL_INVALID.get(flow, _DEFAULT_INVALID)

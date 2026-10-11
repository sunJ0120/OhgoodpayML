"""부실비와 WoE.

    WoE_k = ln( (정상_k / 정상_전체) / (부실_k / 부실_전체) )
          = ln( 구간 정상 오즈 / 전체 정상 오즈 )

WoE 가 양수이면 평균보다 안전하고 음수이면 위험하다.
구간에 정상 또는 부실이 0건이면 두 값 모두에 0.5 를 더해 계산한다 (ln(0) 방지).
"""
import math


def bad_rate(good: float, bad: float) -> float:
    return bad / (good + bad)


def _corrected(good: float, bad: float) -> tuple[float, float]:
    if good == 0 or bad == 0:
        return good + 0.5, bad + 0.5
    return good, bad


def woe(good: float, bad: float, good_total: float, bad_total: float) -> float:
    g, b = _corrected(good, bad)
    return math.log((g / good_total) / (b / bad_total))


def woe_by_odds(good: float, bad: float, good_total: float, bad_total: float) -> float:
    """동치 표현. woe() 와 같은 값이어야 한다."""
    g, b = _corrected(good, bad)
    return math.log((g / b) / (good_total / bad_total))

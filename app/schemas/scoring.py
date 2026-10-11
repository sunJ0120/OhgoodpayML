from enum import Enum
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Optional

# profile 
class ScoreProfile(str, Enum):
    baseline = "baseline"
    conservative = "conservative"
    growth = "growth"

# 네 dataclass와 1:1 매핑
class InputFeaturesIn(BaseModel):
    # camelCase -> snake_case 매핑 (alias)
    extension_this_month: bool = Field(False, alias="extensionThisMonth")
    auto_extension_this_month: bool = Field(False, alias="autoExtensionThisMonth")
    auto_extension_cnt_12m: int = Field(0, ge=0, alias="autoExtensionCnt12m")
    grade_point: int = Field(0, ge=0, le=200, alias="gradePoint")  
    is_blocked: bool = Field(False, alias="blocked")
    payment_cnt_12m: int = Field(0, ge=0, alias="paymentCnt12m")
    payment_amount_12m: float = Field(0.0, ge=0, alias="paymentAmount12m")
    current_cycle_spend: float = Field(0.0, ge=0, alias="currentCycleSpend")

    model_config = ConfigDict(
        populate_by_name=True,  # alias 채우기 허용
        extra="forbid",         # 오타/미매핑 키가 오면 422로 즉시 알림
    )
# /score 전용 입력. 키는 도메인 InputFeatures 와 동일한 snake_case.
# 모르는 키나 누락된 키를 기본값으로 덮지 않고 422 로 거절한다.
class ScoreRequest(BaseModel):
    extension_this_month: bool
    auto_extension_this_month: bool
    auto_extension_cnt_12m: int
    grade_point: int
    is_blocked: bool
    payment_cnt_12m: int
    payment_amount_12m: float
    current_cycle_spend: float

    model_config = ConfigDict(extra="forbid")

class ReasonOut(BaseModel):
    code: str
    label: str
    contribution: float
    detail: str

class ScoreResultOut(BaseModel):
    score: int
    band: str
    risk: float
    grade_name: str
    limit: int
    point_percent: float
    est_rewards_this_cycle: int
    to_next_grade_points: int
    top_negative: List[ReasonOut]
    top_positive: Optional[ReasonOut]

class SayMyNameIn(InputFeaturesIn):  # ← 기존 점수 입력 스키마 상속
    customer_id: Optional[str] = Field(default=None, alias="customerId")
    username: Optional[str] = None
    name: Optional[str] = None
    grade: Optional[str] = None
    ohgood_score: Optional[int] = Field(default=None, alias="ohgoodScore")

    # camelCase/snake_case 모두 허용
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    # 스프링이 숫자로 보낼 수도 있어 문자열로 정규화
    @field_validator("customer_id", mode="before")
    def _id_to_str(cls, v):
        return None if v is None else str(v)

class SayMyNameOut(BaseModel):
    message: str
    sessionId: str
    ttlSeconds: int
    score: int
    userId: str
    gradeName: str
    gradeLimit: int
    pointPercent: float
"""평가 세트 파일 형식과 로더.

평가 세트 = 프롬프트 키별 고정 입력과 기대 결과. 새 프롬프트 버전을 올리기 전에 돌려 이전 버전과 비교한다.
파일 위치: eval/sets/<prompt_key>.json
"""
import json
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, ValidationError, model_validator

CATEGORIES = {
    "normal", "edge_nonsense", "edge_empty", "edge_long",
    "edge_low_balance", "edge_no_hobby", "edge_prev_keyword",
}


class EvalSetError(Exception):
    """평가 세트 파일 형식 오류"""


class Expected(BaseModel):
    """validate_* 의 기대 결과. normalized 가 None 이면 값은 검사하지 않고, 리스트면 그중 하나면 정답(any-of)이다."""
    is_valid: bool
    normalized: Union[str, list[str], None] = None

    def normalized_options(self) -> Optional[list[str]]:
        if self.normalized is None:
            return None
        return [self.normalized] if isinstance(self.normalized, str) else list(self.normalized)


class KeywordContext(BaseModel):
    """keyword_extract 평가 입력(고객 상황)"""
    hobby: str = ""
    mood: str = ""
    credit_limit: int
    balance: int
    prev_keyword: Optional[str] = None


class EvalCase(BaseModel):
    id: str
    category: str
    input: str
    expected: Optional[Expected] = None
    context: Optional[KeywordContext] = None

    @model_validator(mode="after")
    def _check_category(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"알 수 없는 category: {self.category}")
        return self


class EvalSet(BaseModel):
    prompt_key: str
    kind: Literal["validation", "keyword"]
    cases: list[EvalCase]

    @model_validator(mode="after")
    def _check_cases(self):
        if not self.cases:
            raise ValueError("cases 가 비어 있다")
        ids = [c.id for c in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case id 가 중복된다")
        for c in self.cases:
            if self.kind == "validation" and c.expected is None:
                raise ValueError(f"{c.id}: validation 케이스에는 expected 가 필요하다")
            if self.kind == "keyword" and c.context is None:
                raise ValueError(f"{c.id}: keyword 케이스에는 context 가 필요하다")
        return self


def load_evalset(path: Union[str, Path]) -> EvalSet:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return EvalSet.model_validate(raw)
    except (json.JSONDecodeError, ValidationError, OSError) as e:
        raise EvalSetError(f"{path}: {e}") from e

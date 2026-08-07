from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence


@dataclass(frozen=True)
class ActivitySpec:
    code: int
    internal_label: str
    prompt_label: str
    prompt_note: str
    route_policy: str
    infra_type: Optional[int] = None


LLM_DEFAULT_STAY_MINUTES = 60
LLM_DEFAULT_STAY_TICKS = max(1, int(round(LLM_DEFAULT_STAY_MINUTES / 5.0)))


HOME = 1
WORK = 2
EAT_MEAL = 3
EDUCATION = 4
RECREATIONAL = 5
SHOPPING = 6
CARE = 7
COMMUNITY = 8
OTHER = 9
SOCIAL_VISIT = 10


_ACTIVITY_SPECS: Sequence[ActivitySpec] = (
    ActivitySpec(
        code=HOME,
        internal_label="home",
        prompt_label="Home",
        prompt_note=(
            "Home activities, including staying at home for personal routines."
        ),
        route_policy="own_home",
    ),
    ActivitySpec(
        code=WORK,
        internal_label="work",
        prompt_label="Work",
        prompt_note="Work and work-related destinations.",
        route_policy="assigned_work",
    ),
    ActivitySpec(
        code=EAT_MEAL,
        internal_label="eat_meal",
        prompt_label="Eat Meal",
        prompt_note=(
            "Going out to eat (restaurant/cafe/food pickup destination)."
        ),
        route_policy="sample_infra_type",
        infra_type=3,
    ),
    ActivitySpec(
        code=EDUCATION,
        internal_label="education",
        prompt_label="Education",
        prompt_note=(
            "Education-related destinations, including school and "
            "daycare/child-care attendance."
        ),
        route_policy="education_special",
        infra_type=4,
    ),
    ActivitySpec(
        code=RECREATIONAL,
        internal_label="recreational",
        prompt_label="Recreational",
        prompt_note="Recreation, leisure, and exercise destinations.",
        route_policy="sample_infra_type",
        infra_type=5,
    ),
    ActivitySpec(
        code=SHOPPING,
        internal_label="shopping",
        prompt_label="Shopping",
        prompt_note="Buying goods and services.",
        route_policy="sample_infra_type",
        infra_type=6,
    ),
    ActivitySpec(
        code=CARE,
        internal_label="care",
        prompt_label="Care",
        prompt_note="Health care and adult-care destinations.",
        route_policy="sample_infra_type",
        infra_type=7,
    ),
    ActivitySpec(
        code=COMMUNITY,
        internal_label="community",
        prompt_label="Community",
        prompt_note="Volunteer, religious, and community activities.",
        route_policy="sample_infra_type",
        infra_type=8,
    ),
    ActivitySpec(
        code=OTHER,
        internal_label="other",
        prompt_label="Other",
        prompt_note=(
            "Other/general purposes not covered by the categories above."
        ),
        route_policy="sample_infra_type",
        infra_type=9,
    ),
    ActivitySpec(
        code=SOCIAL_VISIT,
        internal_label="social_visit",
        prompt_label="Social Visit",
        prompt_note=(
            "Visiting friends or relatives (often at another residence)."
        ),
        route_policy="residential_not_own_home",
        infra_type=1,
    ),
)


ACTIVITY_SPECS_BY_CODE: Dict[int, ActivitySpec] = {
    spec.code: spec for spec in _ACTIVITY_SPECS
}
ACTIVITY_CODES = tuple(spec.code for spec in _ACTIVITY_SPECS)
VALID_ACTIVITY_TYPES = frozenset(ACTIVITY_CODES)
INTERNAL_LABELS_BY_CODE: Dict[int, str] = {
    spec.code: spec.internal_label for spec in _ACTIVITY_SPECS
}
PROMPT_LABELS_BY_CODE: Dict[int, str] = {
    spec.code: spec.prompt_label for spec in _ACTIVITY_SPECS
}
ROUTE_POLICY_BY_CODE: Dict[int, str] = {
    spec.code: spec.route_policy for spec in _ACTIVITY_SPECS
}
INFRA_TYPE_BY_CODE: Dict[int, Optional[int]] = {
    spec.code: spec.infra_type for spec in _ACTIVITY_SPECS
}
PLANNER_ACTIVITY_NAME_TO_CODE: Dict[str, int] = {
    spec.internal_label: spec.code for spec in _ACTIVITY_SPECS
}


def get_activity_spec(code: int) -> Optional[ActivitySpec]:
    try:
        return ACTIVITY_SPECS_BY_CODE.get(int(code))
    except Exception:
        return None


def activity_internal_label(code: int, default: str = "unknown") -> str:
    spec = get_activity_spec(code)
    return spec.internal_label if spec is not None else default


def activity_prompt_label(code: int, default: str = "Unknown") -> str:
    spec = get_activity_spec(code)
    return spec.prompt_label if spec is not None else default


def activity_prompt_note(code: int, default: str = "") -> str:
    spec = get_activity_spec(code)
    return spec.prompt_note if spec is not None else default


def activity_route_policy(code: int) -> Optional[str]:
    spec = get_activity_spec(code)
    return spec.route_policy if spec is not None else None


def activity_route_infra_type(code: int) -> Optional[int]:
    spec = get_activity_spec(code)
    return spec.infra_type if spec is not None else None


def prompt_activity_vocab_text() -> str:
    return ", ".join(spec.prompt_label for spec in _ACTIVITY_SPECS)


def planner_activity_vocab_text() -> str:
    return ", ".join(spec.internal_label for spec in _ACTIVITY_SPECS)


def planner_activity_name_to_code() -> Dict[str, int]:
    return dict(PLANNER_ACTIVITY_NAME_TO_CODE)


def format_activity_codes_block(indent: int = 4) -> str:
    prefix = " " * max(0, int(indent))
    lines = [
        f"{prefix}{spec.code}: {spec.prompt_label}" for spec in _ACTIVITY_SPECS
    ]
    return "\n".join(lines)


def format_activity_notes_block(
    indent: int = 4,
    codes: Optional[Sequence[int]] = None,
) -> str:
    prefix = " " * max(0, int(indent))
    allowed = None if codes is None else {int(code) for code in codes}
    lines = []
    for spec in _ACTIVITY_SPECS:
        if allowed is not None and spec.code not in allowed:
            continue
        lines.append(f"{prefix}- {spec.prompt_label}: {spec.prompt_note}")
    return "\n".join(lines)

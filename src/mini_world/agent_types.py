from __future__ import annotations

from typing import (
    Iterable,
    # Optional
)

import pandas as pd

WORKER = 1
STUDENT = 2
HOMEMAKER = 3

VALID_AGENT_TYPES = frozenset({WORKER, STUDENT, HOMEMAKER})

ROLE_LABEL_BY_TYPE = {
    WORKER: "worker",
    STUDENT: "student",
    HOMEMAKER: "homemaker",
}

COHORT_LABEL_BY_TYPE = {
    WORKER: "worker",
    STUDENT: "student",
    HOMEMAKER: "homemaker",
}

AGENT_TYPE_BY_COHORT_LABEL = {
    "worker": WORKER,
    "student": STUDENT,
    "homemaker": HOMEMAKER,
}


def parse_agent_type(value, context: str = "agent_type") -> int:
    try:
        agent_type = int(value)
    except Exception as exc:
        raise ValueError(
            f"{context}: expected integer in {sorted(VALID_AGENT_TYPES)}, "
            f"got {value!r}"
        ) from exc
    if agent_type not in VALID_AGENT_TYPES:
        raise ValueError(
            f"{context}: invalid value {agent_type}; expected one of "
            f"{sorted(VALID_AGENT_TYPES)}"
        )
    return agent_type


def role_label_for_agent_type(value, context: str = "agent_type") -> str:
    return ROLE_LABEL_BY_TYPE[parse_agent_type(value, context=context)]


def cohort_label_for_agent_type(value, context: str = "agent_type") -> str:
    return COHORT_LABEL_BY_TYPE[parse_agent_type(value, context=context)]


def parse_cohort_label(value: str) -> int:
    label = str(value or "").strip().lower()
    if label not in AGENT_TYPE_BY_COHORT_LABEL:
        raise ValueError(
            f"Unsupported cohort '{value}'. "
            f"Expected one of {sorted(AGENT_TYPE_BY_COHORT_LABEL)}."
        )
    return AGENT_TYPE_BY_COHORT_LABEL[label]


def validate_agent_type_iterable(
    values: Iterable,
    context: str = "agent_type",
) -> None:
    invalid = set()
    for raw in values:
        if pd.isna(raw):
            continue
        try:
            val = int(raw)
        except Exception:
            invalid.add(str(raw))
            continue
        if val not in VALID_AGENT_TYPES:
            invalid.add(val)
    if invalid:
        raise ValueError(
            f"{context}: invalid values {invalid}; expected only "
            f"{sorted(VALID_AGENT_TYPES)}"
        )


def validate_agent_type_series(
    series: pd.Series,
    context: str = "agent_type",
    allow_na: bool = True,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if not allow_na and numeric.isna().any():
        bad_idx = numeric[numeric.isna()].index[:10].tolist()
        raise ValueError(
            f"{context}: contains non-numeric values at {bad_idx}"
        )
    invalid_mask = numeric.notna() & ~numeric.isin(list(VALID_AGENT_TYPES))
    if invalid_mask.any():
        invalid_vals = sorted(
            numeric[invalid_mask].astype(int).unique().tolist()
        )
        raise ValueError(
            f"{context}: invalid values {invalid_vals}; expected only "
            f"{sorted(VALID_AGENT_TYPES)}"
        )
    return numeric.astype("Int64")


def validated_agent_type(value, context: str = "agent_type") -> int:
    return parse_agent_type(value, context=context)

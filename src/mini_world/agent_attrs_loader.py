"""Load and validate ``input_agent_attrs.csv`` for the current schema.

This loader is intentionally aligned to the newer attr sidecar format used by
``Inputs/SF_agents_10K``. The previous implementation is kept in
``agent_attrs_loader_legacy.py`` for reference only.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, Iterable, List

import pandas as pd


EXPECTED_COLUMNS = [
    "agent_id",
    "household_income_band",
    "school_type",
    "employment_status",
    "work_schedule_type",
    "subtype_hint",
    "worker_type",
    "household_size",
    "household_vehicle_count",
    "work_building_tag",
    "occupation",
    "household_adults",
    "household_children",
    "household_elder",
]

KNOWN_COLUMNS = set(EXPECTED_COLUMNS) | {"city_name"}

DEFAULT_AGENT_ATTRS: Dict[str, Any] = {
    "employment_status": "unknown",
    "work_schedule_type": "unknown",
    "school_type": "unknown",
    "subtype_hint": "unknown",
    "worker_type": "unknown",
    "household_income_band": "unknown",
    "household_size": "unknown",
    "household_vehicle_count": "unknown",
    "work_building_tag": "unknown",
    "occupation": "unknown",
    "household_adults": "unknown",
    "household_children": "unknown",
    "household_elder": "unknown",
    "city_name": "unknown",
}

ENUM_COLUMNS = {
    "employment_status": {"worker", "student", "non_worker"},
    "work_schedule_type": {"full_time", "not_applicable"},
    "school_type": {
        "college",
        "elementary_school",
        "high_school",
        "kindergarten",
        "middle_school",
        "not_in_school",
    },
    "subtype_hint": {
        "student_college",
        "student_elementary_school",
        "student_high_school",
        "student_kindergarten",
        "student_middle_school",
        "unknown",
        "worker_full_time",
    },
    "worker_type": {"full_time", "not_worker"},
    "household_income_band": {
        "$10,000 to $14,999",
        "$100,000 to $149,999",
        "$15,000 to $24,999",
        "$150,000 to $199,999",
        "$200,000 or more",
        "$25,000 to $34,999",
        "$35,000 to $49,999",
        "$50,000 to $74,999",
        "$75,000 to $99,999",
        "Less than $10,000",
    },
    "household_size": {
        "1-person",
        "2-person",
        "3-person",
        "4-or-more-person",
    },
    "household_vehicle_count": {
        "1 vehicle",
        "2 vehicles",
        "3 vehicles",
        "4 or more vehicles",
        "No vehicle",
    },
    "household_adults": {"Couple", "One or more", "Single"},
    "household_children": {
        "6 to 17 yo",
        "No children",
        "Under 6 yo",
        "Under 6 yo and 6 to 17yo",
    },
}

BOOL_STRING_COLUMNS = {"household_elder"}
TEXT_COLUMNS = {"work_building_tag", "occupation", "city_name"}


@dataclass
class AgentAttrsLoadResult:
    enabled: bool
    sidecar_path: str
    rows_loaded: int
    matched_agents: int
    unmatched_rows: int
    agents_using_defaults: int
    unknown_columns: List[str]
    attrs_by_agent_id: Dict[int, Dict[str, Any]]

    def summary_line(self) -> str:
        if not self.enabled:
            return (
                "Agent attrs sidecar not found; using defaults for all agents "
                f"(defaults={self.agents_using_defaults})."
            )
        unknown_cols = (
            ", ".join(self.unknown_columns) if self.unknown_columns else "none"
        )
        return (
            "Agent attrs sidecar loaded "
            f"(path={self.sidecar_path}, rows={self.rows_loaded}, "
            f"matched={self.matched_agents}, unmatched={self.unmatched_rows}, "
            f"defaults={self.agents_using_defaults}, "
            f"unknown_columns={unknown_cols})."
        )


def default_agent_attrs() -> Dict[str, Any]:
    return dict(DEFAULT_AGENT_ATTRS)


def load_optional_agent_attrs(
    input_data_folder: str,
    agent_ids: Iterable[int],
) -> AgentAttrsLoadResult:
    target_agent_ids = {int(v) for v in agent_ids}
    sidecar_path = os.path.join(input_data_folder, "input_agent_attrs.csv")
    if not os.path.exists(sidecar_path):
        return AgentAttrsLoadResult(
            enabled=False,
            sidecar_path=sidecar_path,
            rows_loaded=0,
            matched_agents=0,
            unmatched_rows=0,
            agents_using_defaults=len(target_agent_ids),
            unknown_columns=[],
            attrs_by_agent_id={},
        )

    df = pd.read_csv(sidecar_path, dtype=str, keep_default_na=False)
    rows_loaded = len(df)
    _validate_required_columns(df, sidecar_path)
    df = _validate_agent_id_column(df, sidecar_path)

    unknown_columns = sorted(c for c in df.columns if c not in KNOWN_COLUMNS)
    _validate_enum_columns(df, sidecar_path)
    _validate_bool_string_columns(df, sidecar_path)
    _validate_text_columns(df, sidecar_path)

    attrs_by_agent_id: Dict[int, Dict[str, Any]] = {}
    unmatched_rows = 0

    for row in df.to_dict(orient="records"):
        agent_id = int(row["agent_id"])
        if agent_id not in target_agent_ids:
            unmatched_rows += 1
            continue
        attrs = default_agent_attrs()
        attrs.update(_row_to_attrs(row))
        attrs_by_agent_id[agent_id] = attrs

    matched_agents = len(attrs_by_agent_id)
    return AgentAttrsLoadResult(
        enabled=True,
        sidecar_path=sidecar_path,
        rows_loaded=rows_loaded,
        matched_agents=matched_agents,
        unmatched_rows=unmatched_rows,
        agents_using_defaults=max(0, len(target_agent_ids) - matched_agents),
        unknown_columns=unknown_columns,
        attrs_by_agent_id=attrs_by_agent_id,
    )


def _validate_required_columns(df: pd.DataFrame, sidecar_path: str) -> None:
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"{sidecar_path}: missing required columns {missing}."
        )


def _validate_agent_id_column(
    df: pd.DataFrame,
    sidecar_path: str,
) -> pd.DataFrame:
    raw = df["agent_id"].astype(str).str.strip()
    if (raw == "").any():
        bad_rows = (raw == "").to_numpy().nonzero()[0][:5].tolist()
        raise ValueError(
            f"{sidecar_path}: empty agent_id values at rows {bad_rows}."
        )
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.isna().any():
        bad_vals = raw[numeric.isna()].head(5).tolist()
        raise ValueError(
            f"{sidecar_path}: non-numeric agent_id values {bad_vals}."
        )
    if (numeric % 1 != 0).any():
        bad_vals = raw[(numeric % 1 != 0)].head(5).tolist()
        raise ValueError(
            f"{sidecar_path}: non-integer agent_id values {bad_vals}."
        )
    ids = numeric.astype(int)
    if (ids <= 0).any():
        bad_vals = ids[ids <= 0].head(5).tolist()
        raise ValueError(
            f"{sidecar_path}: agent_id values must be > 0. Got {bad_vals}."
        )
    if ids.duplicated().any():
        dupes = sorted(ids[ids.duplicated()].unique().tolist())[:10]
        raise ValueError(
            f"{sidecar_path}: duplicate agent_id rows found {dupes}."
        )
    checked = df.copy()
    checked["agent_id"] = ids
    return checked


def _validate_enum_columns(df: pd.DataFrame, sidecar_path: str) -> None:
    for col, allowed in ENUM_COLUMNS.items():
        vals = df[col].astype(str).str.strip()
        invalid = vals[(vals != "") & (~vals.isin(allowed))]
        if not invalid.empty:
            bad_vals = sorted(set(invalid.tolist()))[:10]
            raise ValueError(
                f"{sidecar_path}: invalid values for '{col}': {bad_vals}. "
                f"Allowed={sorted(allowed)}"
            )


def _normalize_bool_string(value: Any) -> str:
    raw = str(value).strip()
    lowered = raw.lower()
    if lowered == "true":
        return "True"
    if lowered == "false":
        return "False"
    return raw


def _validate_bool_string_columns(df: pd.DataFrame, sidecar_path: str) -> None:
    for col in BOOL_STRING_COLUMNS:
        vals = df[col].astype(str).str.strip()
        normalized = vals.map(_normalize_bool_string)
        invalid = normalized[(normalized != "") & (~normalized.isin({"True", "False"}))]
        if not invalid.empty:
            bad_vals = sorted(set(invalid.tolist()))[:10]
            raise ValueError(
                f"{sidecar_path}: invalid values for '{col}': {bad_vals}. "
                "Allowed=['False', 'True']"
            )


def _validate_text_columns(df: pd.DataFrame, sidecar_path: str) -> None:
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            continue
        vals = df[col].astype(str).str.strip()
        if col in {"work_building_tag", "occupation"} and (vals == "").any():
            bad_rows = (vals == "").to_numpy().nonzero()[0][:10].tolist()
            raise ValueError(
                f"{sidecar_path}: empty values for '{col}' at rows {bad_rows}."
            )


def _row_to_attrs(row: Dict[str, Any]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for col in KNOWN_COLUMNS:
        if col == "agent_id" or col not in row:
            continue
        raw = str(row[col]).strip()
        if not raw:
            continue
        if col in BOOL_STRING_COLUMNS:
            attrs[col] = _normalize_bool_string(raw)
        else:
            attrs[col] = raw
    return attrs

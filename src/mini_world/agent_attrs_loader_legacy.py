"""Load and validate optional ``input_agent_attrs.csv`` sidecar data.

This module is imported by the simulation runtime. It performs schema checks,
normalizes legacy values, and returns per-agent attribute dictionaries keyed by
``agent_id``. It is not intended to be run as a standalone CLI script.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Dict, Iterable, List

import pandas as pd


DEFAULT_AGENT_ATTRS: Dict[str, Any] = {
    "employment_status": "unknown",
    "work_schedule_type": "unknown",
    "school_type": "unknown",
    "household_vehicle_count": None,
    "household_income_band": "unknown",
    "household_lifecycle_type": "unknown",
    "subtype_hint": "unknown",
    "worker_type": "unknown",
    "occupation_group_soc": "unknown",
    "industry_group_naics": "unknown",
    "commute_mode": "unknown",
    "commute_time_band": "unknown",
    "household_size": None,
    "city_name": "unknown",
    "attrs_source": "runtime_default",
    # Legacy fallback retained for backward compatibility.
    "nhts_hh_cbsa_code": "0",
}

INCOME_BAND_LABEL_VALUES = {
    "under 10k",
    "10k to 14k",
    "15k to 24k",
    "25k to 34k",
    "35k to 49k",
    "50k to 74k",
    "75k to 99k",
    "100k to 124k",
    "125k to 149k",
    "150k to 199k",
    "200k or more",
}

LIFECYCLE_LABEL_VALUES = {
    "one adult no children",
    "two or more adults no children",
    "one adult youngest child age 0 to 5",
    "two or more adults youngest child age 0 to 5",
    "one adult youngest child age 6 to 15",
    "two or more adults youngest child age 6 to 15",
    "one adult youngest child age 16 to 21",
    "two or more adults youngest child age 16 to 21",
    "one adult retired no children",
    "two or more adults retired no children",
}

LEGACY_INCOME_TO_LABEL = {
    "inc_01": "under 10k",
    "inc_02": "10k to 14k",
    "inc_03": "15k to 24k",
    "inc_04": "25k to 34k",
    "inc_05": "35k to 49k",
    "inc_06": "50k to 74k",
    "inc_07": "75k to 99k",
    "inc_08": "100k to 124k",
    "inc_09": "125k to 149k",
    "inc_10": "150k to 199k",
    "inc_11": "200k or more",
}

LEGACY_LIFECYCLE_TO_LABEL = {
    "lifecycle_1": "one adult no children",
    "lifecycle_2": "two or more adults no children",
    "lifecycle_3": "one adult youngest child age 0 to 5",
    "lifecycle_4": "two or more adults youngest child age 0 to 5",
    "lifecycle_5": "one adult youngest child age 6 to 15",
    "lifecycle_6": "two or more adults youngest child age 6 to 15",
    "lifecycle_7": "one adult youngest child age 16 to 21",
    "lifecycle_8": "two or more adults youngest child age 16 to 21",
    "lifecycle_9": "one adult retired no children",
    "lifecycle_10": "two or more adults retired no children",
}

ENUM_COLUMNS = {
    "employment_status": {"worker", "non_worker", "student", "unknown"},
    "work_schedule_type": {
        "full_time",
        "part_time",
        "irregular",
        "not_applicable",
        "unknown",
    },
    "school_type": {
        "k12",
        "college",
        "other_school",
        "not_in_school",
        "unknown",
    },
    "household_income_band": (
        INCOME_BAND_LABEL_VALUES
        | set(LEGACY_INCOME_TO_LABEL.keys())
        | {"unknown"}
    ),
    "household_lifecycle_type": (
        LIFECYCLE_LABEL_VALUES
        | set(LEGACY_LIFECYCLE_TO_LABEL.keys())
        | {"unknown"}
    ),
    "subtype_hint": {
        "worker_ft",
        "worker_pt",
        "worker_generic",
        "student_k12",
        "student_college",
        "student_generic",
        "unknown",
    },
    "worker_type": {"full_time", "part_time", "not_worker", "unknown"},
    "commute_mode": {
        "drive",
        "transit",
        "bike",
        "walk",
        "motorcycle",
        "work_from_home",
        "other",
        "unknown",
    },
    "commute_time_band": {
        "0_14",
        "15_29",
        "30_44",
        "45_59",
        "60_plus",
        "unknown",
    },
}

INT_COLUMNS = {
    "household_vehicle_count",
    "household_size",
}

LEGACY_INT_COLUMNS = {
    "nhts_prmact_code",
    "nhts_worker_code",
    "nhts_wkftpt_code",
    "nhts_schtyp_code",
    "nhts_hhvehcnt_code",
    "nhts_hhfaminc_code",
    "nhts_lif_cyc_code",
}

KNOWN_COLUMNS = {"agent_id", "attrs_source", "city_name", "nhts_hh_cbsa_code"}
KNOWN_COLUMNS.update(ENUM_COLUMNS.keys())
KNOWN_COLUMNS.update(INT_COLUMNS)
KNOWN_COLUMNS.update(LEGACY_INT_COLUMNS)
KNOWN_COLUMNS.update({"occupation_group_soc", "industry_group_naics"})


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
                "Agent attrs sidecar not found; using legacy defaults for all "
                f"agents (defaults={self.agents_using_defaults})."
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
    df = _validate_agent_id_column(df, sidecar_path)

    unknown_columns = sorted(c for c in df.columns if c not in KNOWN_COLUMNS)
    _validate_enum_columns(df, sidecar_path)
    _validate_integer_columns(df, sidecar_path)

    attrs_by_agent_id: Dict[int, Dict[str, Any]] = {}
    unmatched_rows = 0

    for row in df.to_dict(orient="records"):
        agent_id = int(row["agent_id"])
        if agent_id not in target_agent_ids:
            unmatched_rows += 1
            continue
        attrs = default_agent_attrs()
        attrs.update(_row_to_attrs(row))
        if attrs["attrs_source"] == "runtime_default":
            attrs["attrs_source"] = "input_agent_attrs_csv"
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


def _normalize_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _validate_agent_id_column(
    df: pd.DataFrame, sidecar_path: str
) -> pd.DataFrame:
    if "agent_id" not in df.columns:
        raise ValueError(
            f"{sidecar_path}: missing required column 'agent_id'."
        )
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
        if col not in df.columns:
            continue
        vals = df[col].map(_normalize_str)
        invalid = vals[(vals != "") & (~vals.isin(allowed))]
        if not invalid.empty:
            bad_vals = sorted(set(invalid.tolist()))[:10]
            raise ValueError(
                f"{sidecar_path}: invalid values for '{col}': {bad_vals}. "
                f"Allowed={sorted(allowed)}"
            )


def _validate_integer_columns(df: pd.DataFrame, sidecar_path: str) -> None:
    for col in (INT_COLUMNS | LEGACY_INT_COLUMNS):
        if col not in df.columns:
            continue
        vals = df[col].astype(str).str.strip()
        non_empty = vals != ""
        parsed = pd.to_numeric(vals.where(non_empty, None), errors="coerce")
        invalid = non_empty & parsed.isna()
        if invalid.any():
            bad_vals = vals[invalid].head(10).tolist()
            raise ValueError(
                f"{sidecar_path}: non-integer values for '{col}': {bad_vals}."
            )
        parsed_non_empty = parsed[non_empty].astype(int)
        if col == "household_vehicle_count":
            if (parsed_non_empty < 0).any():
                bad_vals = parsed_non_empty[
                    parsed_non_empty < 0
                ].head(10).tolist()
                raise ValueError(
                    f"{sidecar_path}: '{col}' must be >= 0. Got {bad_vals}."
                )
        if col == "household_size":
            if (parsed_non_empty < 1).any():
                bad_vals = parsed_non_empty[
                    parsed_non_empty < 1
                ].head(10).tolist()
                raise ValueError(
                    f"{sidecar_path}: '{col}' must be >= 1. Got {bad_vals}."
                )
        if col == "nhts_lif_cyc_code":
            # Allow 0 (unknown sentinel) and legacy negatives.
            invalid_lif = (
                (parsed_non_empty > 0)
                & ((parsed_non_empty < 1) | (parsed_non_empty > 10))
            )
            if invalid_lif.any():
                bad_vals = parsed_non_empty[invalid_lif].head(10).tolist()
                raise ValueError(
                    f"{sidecar_path}: '{col}' must be 0 or in [1,10]. "
                    f"Got {bad_vals}."
                )


def _row_to_attrs(row: Dict[str, Any]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    for col in KNOWN_COLUMNS:
        if col == "agent_id" or col not in row:
            continue
        if col in ENUM_COLUMNS:
            normalized = _normalize_str(row[col])
            if col == "household_income_band":
                normalized = LEGACY_INCOME_TO_LABEL.get(normalized, normalized)
            elif col == "household_lifecycle_type":
                normalized = LEGACY_LIFECYCLE_TO_LABEL.get(
                    normalized, normalized
                )
            if normalized:
                attrs[col] = normalized
            continue
        if col in (INT_COLUMNS | LEGACY_INT_COLUMNS):
            raw = str(row[col]).strip()
            if raw:
                parsed = int(raw)
                if col in LEGACY_INT_COLUMNS and parsed < 0:
                    parsed = 0
                attrs[col] = parsed
            continue
        if col in {
            "attrs_source",
            "city_name",
            "nhts_hh_cbsa_code",
            "occupation_group_soc",
            "industry_group_naics",
        }:
            raw = str(row[col]).strip()
            if raw:
                attrs[col] = raw
            continue
    return attrs

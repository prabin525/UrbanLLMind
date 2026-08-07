#!/usr/bin/env python3
"""Generate prompt-ready per-agent attributes from NHTS person marginals.

This script reads a simulation ``input_agents.txt`` file, samples NHTS
``perpub.csv`` rows conditioned on ``agent_type`` and target CBSA, and writes
``input_agent_attrs.csv`` compatible with MMv4's optional sidecar loader.

Run:
    python3 scripts/generate_input_agent_attrs_from_nhts.py \
        --agent-file InputDataSample_10/input_agents.txt \
        --perpub dataset/NHTS_2017_csv/perpub.csv \
        --cbsa 41860 \
        --out InputDataSample_10/input_agent_attrs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mini_world.agent_types import (  # noqa: E402
    HOMEMAKER,
    STUDENT,
    WORKER,
    parse_agent_type,
    validate_agent_type_series,
)


OUTPUT_COLUMNS = [
    "agent_id",
    "employment_status",
    "work_schedule_type",
    "school_type",
    "household_vehicle_count",
    "household_income_band",
    "household_lifecycle_type",
    "subtype_hint",
    "attrs_source",
    "nhts_prmact_code",
    "nhts_worker_code",
    "nhts_wkftpt_code",
    "nhts_schtyp_code",
    "nhts_hhvehcnt_code",
    "nhts_hhfaminc_code",
    "nhts_lif_cyc_code",
    "nhts_hh_cbsa_code",
]

PERPUB_BASE_COLUMNS = [
    "HH_CBSA",
    "PRMACT",
    "WORKER",
    "WKFTPT",
    "SCHTYP",
    "HHVEHCNT",
    "HHFAMINC",
    "LIF_CYC",
]

PERPUB_AGE_COLUMN_CANDIDATES = ("R_AGE_IMP", "R_AGE")
PERPUB_SEX_COLUMN_CANDIDATES = ("R_SEX_IMP", "R_SEX")

INCOME_BAND_LABELS = {
    # NHTS 2017 HHFAMINC codebook.
    1: "under 10k",
    2: "10k to 14k",
    3: "15k to 24k",
    4: "25k to 34k",
    5: "35k to 49k",
    6: "50k to 74k",
    7: "75k to 99k",
    8: "100k to 124k",
    9: "125k to 149k",
    10: "150k to 199k",
    11: "200k or more",
}

LIFECYCLE_LABELS = {
    # NHTS 2017 LIF_CYC codebook.
    1: "one adult no children",
    2: "two or more adults no children",
    3: "one adult youngest child age 0 to 5",
    4: "two or more adults youngest child age 0 to 5",
    5: "one adult youngest child age 6 to 15",
    6: "two or more adults youngest child age 6 to 15",
    7: "one adult youngest child age 16 to 21",
    8: "two or more adults youngest child age 16 to 21",
    9: "one adult retired no children",
    10: "two or more adults retired no children",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate input_agent_attrs.csv from NHTS person marginals, "
            "conditioned on simulation agent_type."
        )
    )
    parser.add_argument(
        "--agent-file",
        required=True,
        help="Path to input_agents.txt (tab-separated).",
    )
    parser.add_argument(
        "--perpub",
        default="dataset/NHTS_2017_csv/perpub.csv",
        help="Path to NHTS perpub.csv.",
    )
    parser.add_argument(
        "--cbsa",
        default="41860",
        help="CBSA code to filter NHTS rows (default SF=41860).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Output CSV path. Default: <agent-file-dir>/input_agent_attrs.csv"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for sampling.",
    )
    parser.add_argument(
        "--allow-student-worker-overlap",
        action="store_true",
        help=(
            "If set, student pool includes SCHTYP rows even when WORKER==1. "
            "Default excludes overlap to match realism contract."
        ),
    )
    parser.add_argument(
        "--min-cell-size",
        type=int,
        default=20,
        help=(
            "Preferred minimum candidate row count for a matching stratum "
            "before relaxing constraints (default: 20). If no stratum meets "
            "this threshold, the first non-empty candidate is used."
        ),
    )
    return parser.parse_args()


def _norm_code(value) -> int:
    if pd.isna(value):
        return 0
    try:
        v = int(float(value))
    except Exception:
        return 0
    return v if v > 0 else 0


def _norm_positive_int(value) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        v = int(float(value))
    except Exception:
        return None
    return v if v > 0 else None


def _norm_sim_gender(value) -> str:
    try:
        v = int(float(value))
    except Exception:
        return "unknown"
    if v == 0:
        return "male"
    if v == 1:
        return "female"
    return "unknown"


def _norm_nhts_sex(value) -> str:
    try:
        v = int(float(value))
    except Exception:
        return "unknown"
    if v == 1:
        return "male"
    if v == 2:
        return "female"
    return "unknown"


def _adult_age_bin(age: Optional[int]) -> str:
    if age is None:
        return "unknown"
    if age <= 24:
        return "18_24"
    if age <= 34:
        return "25_34"
    if age <= 44:
        return "35_44"
    if age <= 54:
        return "45_54"
    if age <= 64:
        return "55_64"
    return "65_plus"


def _student_age_bin(age: Optional[int]) -> str:
    if age is None:
        return "unknown"
    if age <= 12:
        return "5_12"
    if age <= 17:
        return "13_17"
    if age <= 22:
        return "18_22"
    if age <= 29:
        return "23_29"
    return "30_plus"


def _target_age_bin(agent_type: int, age_value) -> str:
    agent_type = parse_agent_type(
        agent_type,
        context="_target_age_bin(agent_type)",
    )
    age = _norm_positive_int(age_value)
    if agent_type == STUDENT:
        return _student_age_bin(age)
    # Workers and homemakers use adult bins.
    return _adult_age_bin(age)


def _nhts_age_bin_columns(agent_type: int) -> str:
    agent_type = parse_agent_type(
        agent_type,
        context="_nhts_age_bin_columns(agent_type)",
    )
    if agent_type == STUDENT:
        return "__match_age_bin_student"
    return "__match_age_bin_adult"


def _income_band(code: int) -> str:
    return INCOME_BAND_LABELS.get(code, "unknown")


def _lifecycle_band(code: int) -> str:
    return LIFECYCLE_LABELS.get(code, "unknown")


def _work_schedule_type(code: int) -> str:
    if code == 1:
        return "full_time"
    if code == 2:
        return "part_time"
    if code > 2:
        return "irregular"
    return "unknown"


def _school_type(code: int) -> str:
    if code == 1:
        return "k12"
    if code == 2:
        return "college"
    if code == 3:
        return "other_school"
    return "not_in_school"


def _employment_status(worker_code: int, schtyp_code: int) -> str:
    if worker_code == 1:
        return "worker"
    if schtyp_code in {1, 2, 3}:
        return "student"
    if worker_code == 2:
        return "non_worker"
    return "unknown"


def _subtype_hint(agent_type: int, wkftpt_code: int, schtyp_code: int) -> str:
    agent_type = parse_agent_type(
        agent_type,
        context="_subtype_hint(agent_type)",
    )
    if agent_type == WORKER:
        if wkftpt_code == 1:
            return "worker_ft"
        if wkftpt_code == 2:
            return "worker_pt"
        return "worker_generic"
    if agent_type == STUDENT:
        if schtyp_code == 1:
            return "student_k12"
        if schtyp_code == 2:
            return "student_college"
        return "student_generic"
    return "homemaker"


def _pool_for_agent_type(
    agent_type: int,
    worker_pool: pd.DataFrame,
    student_pool: pd.DataFrame,
    homemaker_pool: pd.DataFrame,
) -> pd.DataFrame:
    agent_type = parse_agent_type(
        agent_type,
        context="_pool_for_agent_type(agent_type)",
    )
    if agent_type == WORKER and len(worker_pool) > 0:
        return worker_pool
    if agent_type == STUDENT and len(student_pool) > 0:
        return student_pool
    if agent_type == HOMEMAKER and len(homemaker_pool) > 0:
        return homemaker_pool
    return pd.DataFrame(columns=worker_pool.columns)


def _build_type_pools(
    base_pool: pd.DataFrame,
    allow_student_worker_overlap: bool,
) -> Dict[str, pd.DataFrame]:
    worker_pool = base_pool[base_pool["WORKER"] == 1].copy()
    student_pool = base_pool[base_pool["SCHTYP"].isin([1, 2, 3])].copy()
    if not allow_student_worker_overlap:
        student_pool = student_pool[student_pool["WORKER"] != 1].copy()
    homemaker_pool = base_pool[
        (base_pool["WORKER"] == 2) & (~base_pool["SCHTYP"].isin([1, 2, 3]))
    ].copy()
    return {
        "all": base_pool,
        "worker": worker_pool,
        "student": student_pool,
        "homemaker": homemaker_pool,
    }


def _sample_row(pool: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    idx = int(rng.integers(0, len(pool)))
    return pool.iloc[idx]


def _sample_row_with_backoff(
    sf_type_pool: pd.DataFrame,
    us_type_pool: pd.DataFrame,
    sf_fallback_pool: pd.DataFrame,
    us_fallback_pool: pd.DataFrame,
    agent_type: int,
    sim_gender_value,
    sim_age_value,
    rng: np.random.Generator,
    min_cell_size: int,
) -> Tuple[pd.Series, str]:
    target_sex = _norm_sim_gender(sim_gender_value)
    target_age_bin = _target_age_bin(agent_type, sim_age_value)
    age_bin_col = _nhts_age_bin_columns(agent_type)

    def _filter(
        pool: pd.DataFrame,
        require_sex: bool,
        require_age_bin: bool,
    ) -> pd.DataFrame:
        out = pool
        if require_sex:
            if target_sex == "unknown" or "__match_sex" not in out.columns:
                return out.iloc[0:0]
            out = out[out["__match_sex"] == target_sex]
        if require_age_bin:
            if target_age_bin == "unknown" or age_bin_col not in out.columns:
                return out.iloc[0:0]
            out = out[out[age_bin_col] == target_age_bin]
        return out

    candidates = [
        ("sf:type+sex+age_bin", _filter(sf_type_pool, True, True)),
        ("us:type+sex+age_bin", _filter(us_type_pool, True, True)),
        ("sf:type+age_bin", _filter(sf_type_pool, False, True)),
        ("us:type+age_bin", _filter(us_type_pool, False, True)),
        ("sf:type+sex", _filter(sf_type_pool, True, False)),
        ("us:type+sex", _filter(us_type_pool, True, False)),
        ("sf:type_only", sf_type_pool),
        ("us:type_only", us_type_pool),
        ("sf:cbsa_fallback", sf_fallback_pool),
        ("us:all_fallback", us_fallback_pool),
    ]
    first_non_empty: Optional[Tuple[str, pd.DataFrame]] = None
    for label, candidate_pool in candidates:
        candidate_size = len(candidate_pool)
        if candidate_size == 0:
            continue
        if first_non_empty is None:
            first_non_empty = (label, candidate_pool)
        if candidate_size >= min_cell_size:
            return _sample_row(candidate_pool, rng), label

    if first_non_empty is not None:
        label, pool = first_non_empty
        return _sample_row(pool, rng), f"{label}:below_min"

    # Should never happen if a fallback pool is non-empty, but keep an explicit
    # error to avoid silent failures if upstream assumptions break.
    raise ValueError("No available NHTS rows to sample from.")


def _build_attr_row(
    agent_id: int,
    agent_type: int,
    sampled: pd.Series,
    attrs_source: str,
) -> Dict:
    agent_type = parse_agent_type(
        agent_type,
        context=f"_build_attr_row(agent_id={agent_id}) agent_type",
    )
    prmact = _norm_code(sampled.get("PRMACT"))
    worker = _norm_code(sampled.get("WORKER"))
    wkftpt = _norm_code(sampled.get("WKFTPT"))
    schtyp = _norm_code(sampled.get("SCHTYP"))
    hhveh = _norm_code(sampled.get("HHVEHCNT"))
    hhinc = _norm_code(sampled.get("HHFAMINC"))
    lifcyc = _norm_code(sampled.get("LIF_CYC"))
    cbsa = _norm_code(sampled.get("HH_CBSA"))

    row = {
        "agent_id": int(agent_id),
        "employment_status": _employment_status(worker, schtyp),
        "work_schedule_type": _work_schedule_type(wkftpt),
        "school_type": _school_type(schtyp),
        "household_vehicle_count": hhveh,
        "household_income_band": _income_band(hhinc),
        "household_lifecycle_type": _lifecycle_band(lifcyc),
        "subtype_hint": _subtype_hint(agent_type, wkftpt, schtyp),
        "attrs_source": attrs_source,
        "nhts_prmact_code": prmact,
        "nhts_worker_code": worker,
        "nhts_wkftpt_code": wkftpt,
        "nhts_schtyp_code": schtyp,
        "nhts_hhvehcnt_code": hhveh,
        "nhts_hhfaminc_code": hhinc,
        "nhts_lif_cyc_code": lifcyc,
        "nhts_hh_cbsa_code": str(cbsa if cbsa > 0 else 0),
    }

    if agent_type == HOMEMAKER:
        # Homemaker cohort in current simulation inputs.
        # Keep household / traceability fields from the sampled NHTS row,
        # but force semantic fields to the intended runtime meaning.
        row["employment_status"] = "non_worker"
        row["work_schedule_type"] = "not_applicable"
        row["school_type"] = "not_in_school"
        row["subtype_hint"] = "homemaker"

    if row["employment_status"] == "worker":
        row["school_type"] = "not_in_school"
    if row["employment_status"] == "student":
        row["work_schedule_type"] = "not_applicable"

    return row


def load_inputs(
    agent_file: Path,
    perpub_path: Path,
    cbsa: str,
    allow_student_worker_overlap: bool,
) -> Tuple[
    pd.DataFrame,
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, Optional[str]],
]:
    agents = pd.read_csv(agent_file, sep="\t")
    required = {"agent_id", "agent_type"}
    missing = required - set(agents.columns)
    if missing:
        raise ValueError(
            f"{agent_file}: missing required columns {sorted(missing)}."
        )
    if agents["agent_id"].duplicated().any():
        dupes = sorted(
            agents.loc[agents["agent_id"].duplicated(), "agent_id"]
            .unique()
            .tolist()
        )[:10]
        raise ValueError(
            f"{agent_file}: duplicate agent_id values {dupes}."
        )
    agents["agent_type"] = validate_agent_type_series(
        agents["agent_type"],
        context=f"{agent_file} agent_type",
        allow_na=False,
    )

    perpub_header_cols = pd.read_csv(perpub_path, nrows=0).columns.tolist()
    age_col = next(
        (c for c in PERPUB_AGE_COLUMN_CANDIDATES if c in perpub_header_cols),
        None,
    )
    sex_col = next(
        (c for c in PERPUB_SEX_COLUMN_CANDIDATES if c in perpub_header_cols),
        None,
    )
    perpub_usecols: List[str] = list(PERPUB_BASE_COLUMNS)
    if age_col and age_col not in perpub_usecols:
        perpub_usecols.append(age_col)
    if sex_col and sex_col not in perpub_usecols:
        perpub_usecols.append(sex_col)

    perpub = pd.read_csv(perpub_path, usecols=perpub_usecols, low_memory=False)
    cbsa_series = perpub["HH_CBSA"].astype(str).str.strip()
    sf = perpub[cbsa_series == str(cbsa).strip()].copy()
    if sf.empty:
        raise ValueError(
            f"No perpub rows found for HH_CBSA={cbsa} in {perpub_path}."
        )

    for col in PERPUB_BASE_COLUMNS:
        if col == "HH_CBSA":
            continue
        perpub[col] = pd.to_numeric(perpub[col], errors="coerce")
        sf[col] = pd.to_numeric(sf[col], errors="coerce")

    if sex_col and sex_col in perpub.columns:
        perpub["__match_sex"] = perpub[sex_col].map(_norm_nhts_sex)
        sf["__match_sex"] = sf[sex_col].map(_norm_nhts_sex)
    else:
        perpub["__match_sex"] = "unknown"
        sf["__match_sex"] = "unknown"

    if age_col and age_col in perpub.columns:
        perpub_age_numeric = pd.to_numeric(perpub[age_col], errors="coerce")
        perpub["__match_age_bin_adult"] = perpub_age_numeric.map(
            lambda v: _adult_age_bin(_norm_positive_int(v))
        )
        perpub["__match_age_bin_student"] = perpub_age_numeric.map(
            lambda v: _student_age_bin(_norm_positive_int(v))
        )

        sf_age_numeric = pd.to_numeric(sf[age_col], errors="coerce")
        sf["__match_age_bin_adult"] = sf_age_numeric.map(
            lambda v: _adult_age_bin(_norm_positive_int(v))
        )
        sf["__match_age_bin_student"] = sf_age_numeric.map(
            lambda v: _student_age_bin(_norm_positive_int(v))
        )
    else:
        perpub["__match_age_bin_adult"] = "unknown"
        perpub["__match_age_bin_student"] = "unknown"
        sf["__match_age_bin_adult"] = "unknown"
        sf["__match_age_bin_student"] = "unknown"

    sf_pools = _build_type_pools(
        base_pool=sf,
        allow_student_worker_overlap=allow_student_worker_overlap,
    )
    us_pools = _build_type_pools(
        base_pool=perpub,
        allow_student_worker_overlap=allow_student_worker_overlap,
    )
    match_meta = {"age_col": age_col, "sex_col": sex_col}
    return agents, sf_pools, us_pools, match_meta


def main() -> None:
    args = parse_args()
    agent_file = Path(args.agent_file)
    perpub_path = Path(args.perpub)
    if args.out is None:
        out_path = agent_file.parent / "input_agent_attrs.csv"
    else:
        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    agents, sf_pools, us_pools, match_meta = load_inputs(
        agent_file=agent_file,
        perpub_path=perpub_path,
        cbsa=str(args.cbsa),
        allow_student_worker_overlap=args.allow_student_worker_overlap,
    )
    rng = np.random.default_rng(args.seed)
    attrs_source = f"nhts_marginal_sampler_cbsa_{args.cbsa}"

    rows = []
    type_pool_missing_count = 0
    match_strategy_counts: Dict[str, int] = {}
    missing_gender_count = 0
    missing_age_bin_count = 0
    has_gender_col = "gender" in agents.columns
    has_age_col = "age" in agents.columns
    min_cell_size = max(1, int(args.min_cell_size))
    for _, agent in agents.iterrows():
        agent_id = int(agent["agent_id"])
        agent_type = parse_agent_type(
            agent["agent_type"],
            context=f"agent_id={agent_id} agent_type",
        )
        sim_gender_value = agent["gender"] if has_gender_col else None
        sim_age_value = agent["age"] if has_age_col else None
        if _norm_sim_gender(sim_gender_value) == "unknown":
            missing_gender_count += 1
        if _target_age_bin(agent_type, sim_age_value) == "unknown":
            missing_age_bin_count += 1
        sf_type_pool = _pool_for_agent_type(
            agent_type=agent_type,
            worker_pool=sf_pools["worker"],
            student_pool=sf_pools["student"],
            homemaker_pool=sf_pools["homemaker"],
        )
        us_type_pool = _pool_for_agent_type(
            agent_type=agent_type,
            worker_pool=us_pools["worker"],
            student_pool=us_pools["student"],
            homemaker_pool=us_pools["homemaker"],
        )
        if len(sf_type_pool) == 0 and len(us_type_pool) == 0:
            type_pool_missing_count += 1
        sampled, match_strategy = _sample_row_with_backoff(
            sf_type_pool=sf_type_pool,
            us_type_pool=us_type_pool,
            sf_fallback_pool=sf_pools["all"],
            us_fallback_pool=us_pools["all"],
            agent_type=agent_type,
            sim_gender_value=sim_gender_value,
            sim_age_value=sim_age_value,
            rng=rng,
            min_cell_size=min_cell_size,
        )
        match_strategy_counts[match_strategy] = (
            match_strategy_counts.get(match_strategy, 0) + 1
        )
        rows.append(
            _build_attr_row(
                agent_id=agent_id,
                agent_type=agent_type,
                sampled=sampled,
                attrs_source=attrs_source,
            )
        )

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Agents processed: {len(out_df)}")
    print("SF pool sizes:", {
        "worker": len(sf_pools["worker"]),
        "student": len(sf_pools["student"]),
        "homemaker": len(sf_pools["homemaker"]),
    })
    print("US pool sizes:", {
        "worker": len(us_pools["worker"]),
        "student": len(us_pools["student"]),
        "homemaker": len(us_pools["homemaker"]),
    })
    print(f"Configured min_cell_size: {min_cell_size}")
    print(
        f"Agents with missing type pools in SF+US: {type_pool_missing_count}"
    )
    print(
        "Matching columns:",
        {
            "sex_col": match_meta.get("sex_col"),
            "age_col": match_meta.get("age_col"),
            "input_has_gender": has_gender_col,
            "input_has_age": has_age_col,
        },
    )
    print(
        "Target demographic availability:",
        {
            "missing_gender": missing_gender_count,
            "missing_age_bin": missing_age_bin_count,
        },
    )
    print("Backoff usage:", match_strategy_counts)
    print(
        "Subtype counts:",
        out_df["subtype_hint"].value_counts(dropna=False).to_dict(),
    )
    print(
        "Employment counts:",
        out_df["employment_status"].value_counts(dropna=False).to_dict(),
    )


if __name__ == "__main__":
    main()

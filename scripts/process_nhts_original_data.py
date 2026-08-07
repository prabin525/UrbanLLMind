#!/usr/bin/env python3
"""Canonical NHTS processor for MMv4 evaluation artifacts.

Generates canonical processed files under dataset/NHTS_2017_csv/processed_data:
- all (US-wide)
- sf  (CBSA-specific, with temporal stay windows)

Key guarantees:
- Preserves legacy core columns used by eval utilities.
- Appends extensible class fields for cohort slicing.
- Applies robust chain construction with continuity diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    # Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

# import numpy as np
import pandas as pd


# Legacy/core schema expected by existing eval loaders.
CORE_COLUMNS: List[str] = [
    "hhid",
    "person",
    "loc_type",
    "travel_time",
    "TDAYDATE",
    "sex",
    "age",
    "location",
    "sex_orig",
]

# Appended cohort/class fields.
CLASS_COLUMNS: List[str] = [
    "nhts_worker_code",
    "nhts_schtyp_code",
    "nhts_prmact_code",
    "person_class",
    "agent_type",
]

# SF-only temporal fields consumed by temporal eval metrics.
SF_TEMPORAL_COLUMNS: List[str] = ["user_id", "date", "combined_time"]


@dataclass(frozen=True)
class ClassRule:
    name: str
    agent_type: int


CLASS_RULES: Sequence[ClassRule] = (
    ClassRule(name="worker", agent_type=1),
    ClassRule(name="student", agent_type=2),
    ClassRule(name="homemaker", agent_type=3),
    ClassRule(name="unclassified", agent_type=0),
)


def _to_int_code(value: Any) -> int:
    if pd.isna(value):
        return 0
    try:
        out = int(float(value))
    except Exception:
        return 0
    if out < 0:
        return 0
    return out


def _safe_int(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _classify_person(
    worker_code: int,
    schtyp_code: int,
    context: str = "",
) -> Tuple[str, int]:
    if worker_code == 1:
        return CLASS_RULES[0].name, CLASS_RULES[0].agent_type
    if schtyp_code in {1, 2, 3} and worker_code != 1:
        return CLASS_RULES[1].name, CLASS_RULES[1].agent_type
    if worker_code == 2 and schtyp_code not in {1, 2, 3}:
        return CLASS_RULES[2].name, CLASS_RULES[2].agent_type
    # NHTS source has a small number of rows with ambiguous WORKER/SCHTYP
    # combinations (e.g. 0/0). Keep these rows in global evaluations as a
    # separate unclassified cohort (agent_type=0).
    return CLASS_RULES[3].name, CLASS_RULES[3].agent_type


def _hhmm_to_minutes(value: Any) -> Optional[int]:
    iv = _safe_int(value)
    if iv is None:
        return None
    if iv < 0:
        return None
    hour = iv // 100
    minute = iv % 100
    if hour == 24 and minute == 0:
        return 24 * 60
    if hour > 24 or minute > 59:
        return None
    return hour * 60 + minute


def _format_ampm(minutes: int) -> str:
    m = max(0, min(int(minutes), 24 * 60 - 1))
    hour24 = m // 60
    minute = m % 60
    ampm = "AM" if hour24 < 12 else "PM"
    hour12 = hour24 % 12
    if hour12 == 0:
        hour12 = 12
    return f"{hour12:02d}:{minute:02d} {ampm}"


def _format_range(start_min: int, end_min: int) -> str:
    s = max(0, min(int(start_min), 24 * 60 - 1))
    e = max(0, min(int(end_min), 24 * 60 - 1))
    if e < s:
        e = s
    return f"{_format_ampm(s)}-{_format_ampm(e)}"


def _build_loc_chain(
    from_vals: Sequence[int], to_vals: Sequence[int]
) -> Optional[List[int]]:
    if not from_vals or not to_vals:
        return None
    if len(from_vals) != len(to_vals):
        return None

    # Fix 1: keep both endpoints for single-trip days.
    if len(from_vals) == 1:
        return [int(from_vals[0]), int(to_vals[0])]

    chain: List[int] = [int(from_vals[0])]
    for i in range(len(from_vals)):
        src = int(from_vals[i])
        dst = int(to_vals[i])
        if i > 0:
            prev_dst = int(to_vals[i - 1])
            # Fix 2: continuity mismatch gets skipped by caller,
            # not hard-failed.
            if prev_dst != src:
                return None
            chain.append(src)
        if i == len(from_vals) - 1:
            chain.append(dst)
    return chain


def _derive_date_str(tdaydate: Any, travday: Any) -> Optional[str]:
    month_key = str(_safe_int(tdaydate) or "")
    day = _safe_int(travday)
    if len(month_key) != 6 or day is None or day <= 0:
        return None
    try:
        year = int(month_key[:4])
        month = int(month_key[4:6])
        dt = pd.Timestamp(year=year, month=month, day=int(day))
    except Exception:
        return None
    return dt.strftime("%Y-%m-%d")


def _build_combined_time(
    group_sorted: pd.DataFrame, expected_len: int
) -> Optional[List[str]]:
    starts = group_sorted["STRTTIME"].map(_hhmm_to_minutes).tolist()
    ends = group_sorted["ENDTIME"].map(_hhmm_to_minutes).tolist()
    if any(v is None for v in starts) or any(v is None for v in ends):
        return None

    starts_i = [int(v) for v in starts if v is not None]
    ends_i = [int(v) for v in ends if v is not None]
    if not starts_i or not ends_i:
        return None

    segments: List[Tuple[int, int]] = []

    # Pre-first-trip stay.
    pre_start = 0
    pre_end = max(pre_start, min(starts_i[0], 24 * 60 - 1))
    segments.append((pre_start, pre_end))

    # Intermediate destination dwells.
    for i in range(len(starts_i) - 1):
        s = max(0, min(ends_i[i], 24 * 60 - 1))
        e = max(s, min(starts_i[i + 1], 24 * 60 - 1))
        segments.append((s, e))

    # Final stay to end-of-day.
    last_s = max(0, min(ends_i[-1], 24 * 60 - 1))
    last_e = 24 * 60 - 1
    if last_e < last_s:
        last_e = last_s
    segments.append((last_s, last_e))

    combined = [_format_range(s, e) for s, e in segments]
    if len(combined) != expected_len:
        return None
    return combined


def _filter_cbsa(df: pd.DataFrame, cbsa: str) -> pd.DataFrame:
    if cbsa.upper() == "ALL":
        return df.copy()
    return df[df["HH_CBSA"].astype(str) == str(cbsa)].copy()


def _sex_label(v: Any) -> str:
    code = _safe_int(v)
    return "male" if code == 1 else "female"


def _load_inputs(
    perpub_path: Path, trippub_path: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_cols = ["HOUSEID", "PERSONID", "HH_CBSA", "WORKER", "SCHTYP", "PRMACT"]
    trip_cols = [
        "HOUSEID",
        "PERSONID",
        "HH_CBSA",
        "TDAYDATE",
        "TRAVDAY",
        "TDTRPNUM",
        "WHYFROM",
        "WHYTO",
        "STRTTIME",
        "ENDTIME",
        "TRVLCMIN",
        "R_SEX",
        "R_AGE",
    ]

    perpub = pd.read_csv(
        perpub_path, usecols=per_cols, dtype={"HH_CBSA": str}, low_memory=False
    )
    trippub = pd.read_csv(
        trippub_path,
        usecols=trip_cols,
        dtype={"HH_CBSA": str},
        low_memory=False,
    )

    # Keep one person row for person-level class attributes.
    perpub = perpub.drop_duplicates(
        subset=["HOUSEID", "PERSONID"], keep="first"
    ).copy()
    return perpub, trippub


@dataclass
class BuildDiagnostics:
    person_days_total: int = 0
    person_days_kept: int = 0
    person_days_dropped_continuity: int = 0
    person_days_dropped_missing_chain: int = 0
    person_days_dropped_missing_time: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "person_days_total": int(self.person_days_total),
            "person_days_kept": int(self.person_days_kept),
            "person_days_dropped_continuity": int(
                self.person_days_dropped_continuity
            ),
            "person_days_dropped_missing_chain": int(
                self.person_days_dropped_missing_chain
            ),
            "person_days_dropped_missing_time": int(
                self.person_days_dropped_missing_time
            ),
        }


def _build_processed(
    perpub: pd.DataFrame,
    trippub: pd.DataFrame,
    cbsa: str,
    include_temporal_fields: bool,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    per = _filter_cbsa(perpub, cbsa)
    trips = _filter_cbsa(trippub, cbsa)

    merged = trips.merge(
        per[["HOUSEID", "PERSONID", "WORKER", "SCHTYP", "PRMACT"]],
        on=["HOUSEID", "PERSONID"],
        how="left",
    )

    merged = merged.dropna(
        subset=[
            "HOUSEID",
            "PERSONID",
            "TDAYDATE",
            "TDTRPNUM",
            "WHYFROM",
            "WHYTO",
        ]
    ).copy()
    merged["TDTRPNUM"] = pd.to_numeric(merged["TDTRPNUM"], errors="coerce")
    merged = merged.dropna(subset=["TDTRPNUM"]).copy()
    merged = merged.sort_values(
        ["HOUSEID", "PERSONID", "TDAYDATE", "TDTRPNUM"]
    ).copy()

    diag = BuildDiagnostics()
    records: List[Dict[str, Any]] = []

    group_cols = ["HOUSEID", "PERSONID", "TDAYDATE"]
    grouped = merged.groupby(group_cols, sort=False)
    for (house_id, person_id, tdaydate), g in grouped:
        diag.person_days_total += 1
        g = g.sort_values("TDTRPNUM").copy()

        from_vals = [_safe_int(v) for v in g["WHYFROM"].tolist()]
        to_vals = [_safe_int(v) for v in g["WHYTO"].tolist()]
        if any(v is None for v in from_vals) or any(
            v is None for v in to_vals
        ):
            diag.person_days_dropped_missing_chain += 1
            continue

        chain = _build_loc_chain(
            from_vals=[int(v) for v in from_vals if v is not None],
            to_vals=[int(v) for v in to_vals if v is not None],
        )
        if chain is None:
            diag.person_days_dropped_continuity += 1
            continue

        worker_code = _to_int_code(g["WORKER"].iloc[0])
        schtyp_code = _to_int_code(g["SCHTYP"].iloc[0])
        prmact_code = _to_int_code(g["PRMACT"].iloc[0])
        person_class, agent_type = _classify_person(
            worker_code,
            schtyp_code,
            context=(
                f"HOUSEID={house_id}, PERSONID={person_id}, "
                f"TDAYDATE={tdaydate}"
            ),
        )

        travel_minutes = (
            pd.to_numeric(g["TRVLCMIN"], errors="coerce").fillna(0.0).sum()
        )
        sex_orig = _safe_int(g["R_SEX"].iloc[0])
        age_val = _safe_int(g["R_AGE"].iloc[0])

        tdaydate_val = (
            _safe_int(tdaydate)
            if _safe_int(tdaydate) is not None
            else tdaydate
        )
        record: Dict[str, Any] = {
            "hhid": int(house_id),
            "person": int(person_id),
            "loc_type": chain,
            "travel_time": float(travel_minutes) * 60.0,
            "TDAYDATE": tdaydate_val,
            "sex": _sex_label(sex_orig),
            "age": age_val if age_val is not None else 0,
            "location": int(len(chain)),
            "sex_orig": sex_orig if sex_orig is not None else 0,
            "nhts_worker_code": worker_code,
            "nhts_schtyp_code": schtyp_code,
            "nhts_prmact_code": prmact_code,
            "person_class": person_class,
            "agent_type": int(agent_type),
        }

        if include_temporal_fields:
            combined_time = _build_combined_time(g, expected_len=len(chain))
            if combined_time is None:
                diag.person_days_dropped_missing_time += 1
                continue
            record["user_id"] = f"{int(house_id)}{int(person_id)}"
            record["date"] = _derive_date_str(tdaydate, g["TRAVDAY"].iloc[0])
            record["combined_time"] = combined_time

        records.append(record)
        diag.person_days_kept += 1

    df = pd.DataFrame(records)

    ordered_cols = (
        CORE_COLUMNS
        + CLASS_COLUMNS
        + (SF_TEMPORAL_COLUMNS if include_temporal_fields else [])
    )
    # Keep column order stable and append any future additions at end.
    extra_cols = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + extra_cols]
    return df, diag.to_dict()


def _snapshot_table(df: pd.DataFrame) -> Dict[str, Any]:
    if df is None or df.empty:
        return {
            "rows": 0,
            "unique_persons": 0,
            "unique_person_days": 0,
        }

    keys_person = set(zip(df["hhid"].astype(int), df["person"].astype(int)))
    keys_day = set(
        zip(
            df["hhid"].astype(int),
            df["person"].astype(int),
            df["TDAYDATE"].astype(str),
        )
    )
    return {
        "rows": int(len(df)),
        "unique_persons": int(len(keys_person)),
        "unique_person_days": int(len(keys_day)),
    }


def _safe_read_pickle(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_pickle(path)
    except Exception:
        return None


def _core_equality_counts(
    old_df: pd.DataFrame, new_df: pd.DataFrame
) -> Dict[str, Any]:
    if old_df is None or new_df is None or old_df.empty or new_df.empty:
        return {"matched_rows": 0, "by_column": {}}

    key_cols = ["hhid", "person", "TDAYDATE"]
    if not all(c in old_df.columns for c in key_cols) or not all(
        c in new_df.columns for c in key_cols
    ):
        return {"matched_rows": 0, "by_column": {}}

    left = old_df[
        key_cols + [c for c in CORE_COLUMNS if c not in key_cols]
    ].copy()
    right = new_df[
        key_cols + [c for c in CORE_COLUMNS if c not in key_cols]
    ].copy()

    merged = left.merge(
        right, on=key_cols, how="inner", suffixes=("_old", "_new")
    )
    result: Dict[str, Any] = {
        "matched_rows": int(len(merged)),
        "by_column": {},
    }
    if merged.empty:
        return result

    for col in [c for c in CORE_COLUMNS if c not in key_cols]:
        old_col = f"{col}_old"
        new_col = f"{col}_new"
        if old_col not in merged.columns or new_col not in merged.columns:
            continue
        if col == "loc_type":
            equal_mask = merged[old_col].apply(tuple) == merged[new_col].apply(
                tuple
            )
        else:
            equal_mask = merged[old_col] == merged[new_col]
        eq_count = int(equal_mask.sum())
        result["by_column"][col] = {
            "equal_rows": eq_count,
            "total_rows": int(len(merged)),
            "equal_pct": round(100.0 * eq_count / max(1, len(merged)), 3),
        }
    return result


def _key_overlap(
    old_df: pd.DataFrame, new_df: pd.DataFrame
) -> Dict[str, float]:
    if old_df is None or new_df is None or old_df.empty or new_df.empty:
        return {
            "overlap_count": 0,
            "overlap_pct_of_old": 0.0,
            "overlap_pct_of_new": 0.0,
        }

    old_keys = set(
        zip(
            old_df["hhid"].astype(int),
            old_df["person"].astype(int),
            old_df["TDAYDATE"].astype(str),
        )
    )
    new_keys = set(
        zip(
            new_df["hhid"].astype(int),
            new_df["person"].astype(int),
            new_df["TDAYDATE"].astype(str),
        )
    )
    inter = old_keys.intersection(new_keys)
    return {
        "overlap_count": int(len(inter)),
        "overlap_pct_of_old": round(
            100.0 * len(inter) / max(1, len(old_keys)), 3
        ),
        "overlap_pct_of_new": round(
            100.0 * len(inter) / max(1, len(new_keys)), 3
        ),
    }


def _print_parity_report(
    name: str, old_df: Optional[pd.DataFrame], new_df: pd.DataFrame
) -> None:
    old_snapshot = (
        _snapshot_table(old_df)
        if old_df is not None
        else {"rows": 0, "unique_persons": 0, "unique_person_days": 0}
    )
    new_snapshot = _snapshot_table(new_df)
    overlap = (
        _key_overlap(old_df, new_df)
        if old_df is not None
        else {
            "overlap_count": 0,
            "overlap_pct_of_old": 0.0,
            "overlap_pct_of_new": 0.0,
        }
    )
    eq = (
        _core_equality_counts(old_df, new_df)
        if old_df is not None
        else {"matched_rows": 0, "by_column": {}}
    )

    print("=" * 100)
    print(f"PARITY REPORT :: {name}")
    old_unique_persons = old_snapshot["unique_persons"]
    old_unique_days = old_snapshot["unique_person_days"]
    new_unique_persons = new_snapshot["unique_persons"]
    new_unique_days = new_snapshot["unique_person_days"]
    print(
        f"old rows={old_snapshot['rows']} "
        f"unique_persons={old_unique_persons} "
        f"unique_person_days={old_unique_days}"
    )
    print(
        f"new rows={new_snapshot['rows']} "
        f"unique_persons={new_unique_persons} "
        f"unique_person_days={new_unique_days}"
    )
    old_overlap_pct = overlap["overlap_pct_of_old"]
    new_overlap_pct = overlap["overlap_pct_of_new"]
    print(
        "key overlap "
        f"count={overlap['overlap_count']} "
        f"old%={old_overlap_pct} new%={new_overlap_pct}"
    )
    print(f"core matched rows on key={eq.get('matched_rows', 0)}")
    for col, stats in eq.get("by_column", {}).items():
        equal_rows = stats["equal_rows"]
        total_rows = stats["total_rows"]
        equal_pct = stats["equal_pct"]
        print(
            f"  {col}: equal_rows={equal_rows} / {total_rows} "
            f"({equal_pct}%)"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical MMv4 processed NHTS files."
    )
    parser.add_argument(
        "--perpub",
        default="dataset/NHTS_2017_csv/perpub.csv",
        help="Path to NHTS perpub.csv",
    )
    parser.add_argument(
        "--trippub",
        default="dataset/NHTS_2017_csv/trippub.csv",
        help="Path to NHTS trippub.csv",
    )
    parser.add_argument(
        "--processed-dir",
        default="dataset/NHTS_2017_csv/processed_data",
        help="Output directory for processed pickles.",
    )
    parser.add_argument(
        "--sf-cbsa", default="41860", help="CBSA code for SF artifact."
    )
    parser.add_argument(
        "--generate",
        choices=["all", "sf", "both"],
        default="both",
        help="Which artifact(s) to generate.",
    )
    parser.add_argument(
        "--all-name", default="all", help="Output filename for all artifact."
    )
    parser.add_argument(
        "--sf-name", default="sf", help="Output filename for SF artifact."
    )
    parser.add_argument(
        "--keep-sf-new",
        action="store_true",
        help="Do not remove processed_data/sf_new after generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    processed_dir = Path(args.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    perpub, trippub = _load_inputs(Path(args.perpub), Path(args.trippub))
    print(f"Loaded perpub rows={len(perpub)} trippub rows={len(trippub)}")

    if args.generate in {"all", "both"}:
        out_all = processed_dir / args.all_name
        old_all = _safe_read_pickle(out_all)
        df_all, diag_all = _build_processed(
            perpub=perpub,
            trippub=trippub,
            cbsa="ALL",
            include_temporal_fields=False,
        )
        print(f"Built all rows={len(df_all)} diag={diag_all}")
        df_all.to_pickle(out_all)
        _print_parity_report("all", old_all, df_all)

    if args.generate in {"sf", "both"}:
        out_sf = processed_dir / args.sf_name
        old_sf = _safe_read_pickle(out_sf)
        df_sf, diag_sf = _build_processed(
            perpub=perpub,
            trippub=trippub,
            cbsa=str(args.sf_cbsa),
            include_temporal_fields=True,
        )
        print(f"Built sf rows={len(df_sf)} diag={diag_sf}")
        df_sf.to_pickle(out_sf)
        _print_parity_report("sf", old_sf, df_sf)

    sf_new_path = processed_dir / "sf_new"
    if sf_new_path.exists() and not args.keep_sf_new:
        sf_new_path.unlink()
        print(f"Removed retired artifact: {sf_new_path}")

    print("Done.")


if __name__ == "__main__":
    main()

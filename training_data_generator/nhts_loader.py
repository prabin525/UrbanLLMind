from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import datetime
from os import PathLike
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import pandas as pd

from analysis.constants import NHTS_WHYTO_TO_NEW_ACTION

from .config import GeneratorConfig
from .role_and_attrs import age_from_row, derive_prompt_attrs


PERPUB_COLUMNS = [
    "HOUSEID",
    "PERSONID",
    "TRAVDAY",
    "TDAYDATE",
    "HH_CBSA",
    "R_AGE",
    "R_AGE_IMP",
    "R_SEX",
    "R_SEX_IMP",
    "WORKER",
    "SCHTYP",
    "PRMACT",
    "WKFTPT",
    "HHVEHCNT",
    "HHFAMINC",
    "LIF_CYC",
    "WRKTRANS",
    "TIMETOWK",
    "WRK_HOME",
    "OCCAT",
    "HHSIZE",
]

TRIPPUB_COLUMNS = [
    "HOUSEID",
    "PERSONID",
    "TRAVDAY",
    "TDAYDATE",
    "HH_CBSA",
    "TDTRPNUM",
    "WHYFROM",
    "WHYTO",
    "STRTTIME",
    "ENDTIME",
]

NHTS_TRAVDAY_TO_NAME = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday",
}

NHTS_TRAVDAY_TO_PYTHON_WEEKDAY = {
    1: 6,
    2: 0,
    3: 1,
    4: 2,
    5: 3,
    6: 4,
    7: 5,
}


@dataclass(frozen=True)
class SampledPersonDay:
    sample_index: int
    sample_day_id: str
    house_id: int
    person_id: int
    person_key: str
    cbsa_code: str
    cbsa_title: str
    prompt_location_name: str
    date_str: str
    day_of_week: str
    age: int
    gender: str
    agent_role: str
    attrs: Dict[str, Any]
    dwell_blocks: List[Dict[str, int]]


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _normalize_code(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _hhmm_to_minutes(value: Any) -> int | None:
    ivalue = _safe_int(value, default=-1)
    if ivalue < 0:
        return None
    hour = ivalue // 100
    minute = ivalue % 100
    if hour == 24 and minute == 0:
        return 1440
    if hour > 24 or minute > 59:
        return None
    return hour * 60 + minute


def _derive_date_str(tdaydate: Any, travday: Any) -> str | None:
    month_key = str(_safe_int(tdaydate, default=-1))
    travday_code = _safe_int(travday, default=-1)
    if len(month_key) != 6 or travday_code <= 0:
        return None
    year = int(month_key[:4])
    month = int(month_key[4:6])
    try:
        if travday_code in NHTS_TRAVDAY_TO_PYTHON_WEEKDAY:
            first_of_month = datetime(year=year, month=month, day=1)
            target_weekday = NHTS_TRAVDAY_TO_PYTHON_WEEKDAY[travday_code]
            offset = (target_weekday - first_of_month.weekday()) % 7
            parsed = datetime(year=year, month=month, day=1 + offset)
        else:
            max_day = monthrange(year, month)[1]
            if travday_code > max_day:
                return None
            parsed = datetime(year=year, month=month, day=travday_code)
    except Exception:
        return None
    return parsed.strftime("%Y-%m-%d")


def _derive_day_of_week(date_str: str, travday: Any) -> str | None:
    travday_code = _safe_int(travday, default=-1)
    if travday_code in NHTS_TRAVDAY_TO_NAME:
        return NHTS_TRAVDAY_TO_NAME[travday_code]
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%A")
    except Exception:
        return None


def _metro_title_to_location_name(cbsa_title: str) -> str:
    title = str(cbsa_title or "").strip()
    if not title:
        return "San Francisco, CA"
    region, separator, suffix = title.partition(",")
    primary_city = region.split("-", 1)[0].strip()
    if not primary_city:
        return "San Francisco, CA"
    if not separator:
        return primary_city
    primary_state = suffix.strip().split("-", 1)[0].strip()
    if not primary_state:
        return primary_city
    return f"{primary_city}, {primary_state}"


def _load_cities_info(path: str | PathLike[str]) -> Dict[str, str]:
    frame = pd.read_csv(path, low_memory=False)
    mapping: Dict[str, str] = {}
    for row in frame.to_dict(orient="records"):
        cbsa_code = _normalize_code(row.get("CBSA"))
        title = str(row.get("CBSA_Title", "")).strip()
        if cbsa_code and title:
            mapping[cbsa_code] = title
    return mapping


def _build_trip_group_lookup(trippub: pd.DataFrame) -> Dict[tuple[int, int, int, int], List[int]]:
    groups = trippub.groupby(
        ["HOUSEID", "PERSONID", "TDAYDATE", "TRAVDAY"],
        sort=False,
    ).indices
    lookup: Dict[tuple[int, int, int, int], List[int]] = {}
    for key, indices in groups.items():
        lookup[tuple(int(value) for value in key)] = list(indices)
    return lookup


def _map_nhts_activity(raw_code: Any) -> int:
    code = _safe_int(raw_code, default=97)
    return int(NHTS_WHYTO_TO_NEW_ACTION.get(code, 9))


def _reconstruct_dwell_blocks(trips: pd.DataFrame) -> List[Dict[str, int]] | None:
    if trips.empty:
        return None

    ordered = trips.sort_values("TDTRPNUM")
    from_to = ordered[["WHYFROM", "WHYTO"]].to_numpy(dtype=float)
    chain_raw: List[int] = []
    for idx, pair in enumerate(from_to):
        source = int(pair[0])
        target = int(pair[1])
        if idx == 0:
            chain_raw.append(source)
            if len(from_to) == 1:
                chain_raw.append(target)
            continue
        prev_target = int(from_to[idx - 1][1])
        if prev_target != source:
            return None
        chain_raw.append(source)
        if idx == len(from_to) - 1:
            chain_raw.append(target)

    starts: List[int] = []
    ends: List[int] = []
    prev_end = -1
    for row in ordered.itertuples(index=False):
        start_minute = _hhmm_to_minutes(getattr(row, "STRTTIME"))
        end_minute = _hhmm_to_minutes(getattr(row, "ENDTIME"))
        if start_minute is None or end_minute is None:
            return None
        if end_minute < start_minute:
            end_minute += 1440
        if start_minute < prev_end:
            return None
        starts.append(start_minute)
        ends.append(end_minute)
        prev_end = end_minute

    blocks: List[Dict[str, int]] = []
    pre_first = max(0, min(starts[0], 1440))
    if pre_first > 0:
        blocks.append(
            {
                "start_minute": 0,
                "end_minute": pre_first,
                "activity_type": _map_nhts_activity(chain_raw[0]),
            }
        )

    for idx in range(len(starts)):
        seg_start = max(0, min(ends[idx], 1440))
        next_start = starts[idx + 1] if idx + 1 < len(starts) else 1440
        seg_end = max(0, min(next_start, 1440))
        if seg_end <= seg_start:
            continue
        blocks.append(
            {
                "start_minute": seg_start,
                "end_minute": seg_end,
                "activity_type": _map_nhts_activity(chain_raw[idx + 1]),
            }
        )

    return blocks or None


def _build_sample(
    *,
    sample_index: int,
    perpub_row: Mapping[str, Any],
    trip_rows: pd.DataFrame,
    config: GeneratorConfig,
    city_lookup: Mapping[str, str],
) -> SampledPersonDay | None:
    date_str = _derive_date_str(
        perpub_row.get("TDAYDATE"),
        perpub_row.get("TRAVDAY"),
    )
    if date_str is None and not trip_rows.empty:
        first_trip = trip_rows.iloc[0].to_dict()
        date_str = _derive_date_str(
            first_trip.get("TDAYDATE"),
            first_trip.get("TRAVDAY"),
        )
    if date_str is None:
        return None

    dwell_blocks = _reconstruct_dwell_blocks(trip_rows)
    if not dwell_blocks:
        return None

    cbsa_code = _normalize_code(perpub_row.get("HH_CBSA"))
    cbsa_title = city_lookup.get(cbsa_code, "Unknown metro")
    prompt_location_name = (
        config.location_name_override
        or _metro_title_to_location_name(cbsa_title)
    )

    attrs = derive_prompt_attrs(
        dict(perpub_row),
        prompt_location_name=prompt_location_name,
        cbsa_code=cbsa_code,
    )
    if (
        config.enable_min_age_filter
        and int(attrs["age"]) < int(config.min_age)
    ):
        return None

    house_id = _safe_int(perpub_row.get("HOUSEID"), default=0)
    person_id = _safe_int(perpub_row.get("PERSONID"), default=0)
    day_of_week = _derive_day_of_week(date_str, perpub_row.get("TRAVDAY"))
    if day_of_week is None:
        return None
    sample_day_id = (
        f"{house_id}:{person_id}:"
        f"{_normalize_code(perpub_row.get('TDAYDATE'))}:"
        f"{_normalize_code(perpub_row.get('TRAVDAY'))}"
    )
    return SampledPersonDay(
        sample_index=sample_index,
        sample_day_id=sample_day_id,
        house_id=house_id,
        person_id=person_id,
        person_key=f"{house_id}:{person_id}",
        cbsa_code=cbsa_code,
        cbsa_title=cbsa_title,
        prompt_location_name=prompt_location_name,
        date_str=date_str,
        day_of_week=day_of_week,
        age=int(attrs["age"]),
        gender=str(attrs["gender"]),
        agent_role=str(attrs["role"]),
        attrs=attrs,
        dwell_blocks=dwell_blocks,
    )


def load_sampled_person_days(config: GeneratorConfig) -> List[SampledPersonDay]:
    perpub = pd.read_csv(config.perpub_path, usecols=PERPUB_COLUMNS, low_memory=False)
    trippub = pd.read_csv(config.trippub_path, usecols=TRIPPUB_COLUMNS, low_memory=False)

    numeric_columns = [
        "HOUSEID",
        "PERSONID",
        "TRAVDAY",
        "TDAYDATE",
        "WORKER",
        "SCHTYP",
        "PRMACT",
        "WKFTPT",
        "HHVEHCNT",
        "HHFAMINC",
        "LIF_CYC",
        "WRKTRANS",
        "TIMETOWK",
        "WRK_HOME",
        "OCCAT",
        "HHSIZE",
        "R_AGE",
        "R_AGE_IMP",
        "R_SEX",
        "R_SEX_IMP",
    ]
    trip_numeric_columns = [
        "HOUSEID",
        "PERSONID",
        "TRAVDAY",
        "TDAYDATE",
        "TDTRPNUM",
        "WHYFROM",
        "WHYTO",
        "STRTTIME",
        "ENDTIME",
    ]
    for column in numeric_columns:
        if column in perpub.columns:
            perpub[column] = pd.to_numeric(perpub[column], errors="coerce")
    for column in trip_numeric_columns:
        if column in trippub.columns:
            trippub[column] = pd.to_numeric(trippub[column], errors="coerce")

    perpub["HH_CBSA"] = perpub["HH_CBSA"].astype(str).str.strip()
    trippub["HH_CBSA"] = trippub["HH_CBSA"].astype(str).str.strip()
    perpub["derived_age"] = [
        age_from_row(row)
        for row in perpub.to_dict(orient="records")
    ]

    cbsa_filter = (
        None
        if config.cbsa_filter is None
        else str(config.cbsa_filter).strip()
    )
    if cbsa_filter and cbsa_filter.upper() != "ALL":
        perpub = perpub[perpub["HH_CBSA"] == cbsa_filter].copy()
        trippub = trippub[trippub["HH_CBSA"] == cbsa_filter].copy()

    if config.enable_min_age_filter:
        perpub = perpub[perpub["derived_age"] >= int(config.min_age)].copy()

    perpub = perpub.drop_duplicates(
        subset=["HOUSEID", "PERSONID", "TDAYDATE", "TRAVDAY"],
        keep="first",
    )
    trippub = trippub.dropna(
        subset=["HOUSEID", "PERSONID", "TDAYDATE", "TRAVDAY"]
    ).copy()
    trip_lookup = _build_trip_group_lookup(trippub)
    city_lookup = _load_cities_info(config.cities_info_path)

    row_indices = list(perpub.index)
    random.Random(config.random_seed).shuffle(row_indices)
    target_count = config.n_person_days

    samples: List[SampledPersonDay] = []
    for row_index in row_indices:
        row = perpub.loc[row_index].to_dict()
        key = (
            _safe_int(row.get("HOUSEID"), default=0),
            _safe_int(row.get("PERSONID"), default=0),
            _safe_int(row.get("TDAYDATE"), default=-1),
            _safe_int(row.get("TRAVDAY"), default=-1),
        )
        trip_indices = trip_lookup.get(key, [])
        if not trip_indices:
            continue
        trip_rows = trippub.iloc[trip_indices].copy()
        sample = _build_sample(
            sample_index=len(samples),
            perpub_row=row,
            trip_rows=trip_rows,
            config=config,
            city_lookup=city_lookup,
        )
        if sample is None:
            continue
        samples.append(sample)
        if target_count is not None and len(samples) >= target_count:
            break

    if target_count is not None and len(samples) < target_count:
        raise ValueError(
            "Requested "
            f"{target_count} person-days but only found "
            f"{len(samples)} valid samples."
        )
    return samples

from __future__ import annotations

from typing import Any, Dict


INCOME_BAND_LABELS = {
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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        out = int(float(value))
    except Exception:
        return default
    return out


def _safe_value(value: Any) -> str:
    text = str(value).strip()
    return text if text else "unknown"


def gender_label_from_row(row: Dict[str, Any]) -> str:
    for column in ("R_SEX_IMP", "R_SEX"):
        code = _safe_int(row.get(column), default=-1)
        if code == 1:
            return "male"
        if code == 2:
            return "female"
    return "unknown"


def age_from_row(row: Dict[str, Any]) -> int:
    for column in ("R_AGE_IMP", "R_AGE"):
        value = _safe_int(row.get(column), default=-1)
        if value >= 0:
            return value
    return 0


def derive_agent_role(worker_code: Any, schtyp_code: Any) -> str:
    worker = _safe_int(worker_code, default=-1)
    schtyp = _safe_int(schtyp_code, default=-1)
    if worker == 1:
        return "worker"
    if schtyp in {1, 2, 3}:
        return "student"
    return "homemaker"


def _work_schedule_type(code: int) -> str:
    if code == 1:
        return "full_time"
    if code == 2:
        return "part_time"
    if code > 2:
        return "irregular"
    return "not_applicable"


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
    return "non_worker"


def _subtype_hint(role: str, wkftpt_code: int, schtyp_code: int) -> str:
    if role == "worker":
        if wkftpt_code == 1:
            return "worker_ft"
        if wkftpt_code == 2:
            return "worker_pt"
        return "worker_generic"
    if role == "student":
        if schtyp_code == 1:
            return "student_k12"
        if schtyp_code == 2:
            return "student_college"
        return "student_generic"
    return "homemaker"


def _commute_mode(
    wrktrans_code: int,
    wrk_home_code: int,
    employment_status: str,
) -> str:
    if employment_status != "worker":
        return "unknown"
    if wrk_home_code == 1:
        return "work_from_home"
    if wrktrans_code == 1:
        return "walk"
    if wrktrans_code == 2:
        return "bike"
    if wrktrans_code in {3, 4, 5, 6, 18}:
        return "drive"
    if wrktrans_code == 8:
        return "motorcycle"
    if wrktrans_code in {10, 11, 12, 13, 14, 15, 16}:
        return "transit"
    if wrktrans_code in {17, 19, 20, 97}:
        return "other"
    return "unknown"


def _commute_time_band(minutes: int) -> str:
    if minutes < 0:
        return "unknown"
    if minutes < 15:
        return "0_14"
    if minutes < 30:
        return "15_29"
    if minutes < 45:
        return "30_44"
    if minutes < 60:
        return "45_59"
    return "60_plus"


def derive_prompt_attrs(
    row: Dict[str, Any],
    *,
    prompt_location_name: str,
    cbsa_code: str,
) -> Dict[str, Any]:
    worker = _safe_int(row.get("WORKER"), default=-1)
    schtyp = _safe_int(row.get("SCHTYP"), default=-1)
    wkftpt = _safe_int(row.get("WKFTPT"), default=-1)
    hhveh = _safe_int(row.get("HHVEHCNT"), default=-1)
    hhinc = _safe_int(row.get("HHFAMINC"), default=-1)
    lif_cyc = _safe_int(row.get("LIF_CYC"), default=-1)
    hhsize = _safe_int(row.get("HHSIZE"), default=-1)
    wrktrans = _safe_int(row.get("WRKTRANS"), default=-1)
    timetowk = _safe_int(row.get("TIMETOWK"), default=-1)
    wrk_home = _safe_int(row.get("WRK_HOME"), default=-1)
    occat = _safe_int(row.get("OCCAT"), default=-1)

    role = derive_agent_role(worker, schtyp)
    employment_status = _employment_status(worker, schtyp)
    work_schedule_type = _work_schedule_type(wkftpt)
    school_type = _school_type(schtyp)
    subtype_hint = _subtype_hint(role, wkftpt, schtyp)
    commute_mode = _commute_mode(wrktrans, wrk_home, employment_status)
    commute_time_band = _commute_time_band(timetowk)

    return {
        "age": age_from_row(row),
        "gender": gender_label_from_row(row),
        "role": role,
        "city_name": prompt_location_name,
        "employment_status": employment_status,
        "work_schedule_type": (
            "not_applicable" if role != "worker" else work_schedule_type
        ),
        "school_type": (
            school_type if role == "student" else "not_in_school"
        ),
        "household_vehicle_count": hhveh if hhveh >= 0 else None,
        "household_income_band": INCOME_BAND_LABELS.get(hhinc, "unknown"),
        "household_lifecycle_type": LIFECYCLE_LABELS.get(
            lif_cyc, "unknown"
        ),
        "subtype_hint": subtype_hint,
        "worker_type": (
            "full_time"
            if role == "worker" and wkftpt == 1
            else "part_time"
            if role == "worker" and wkftpt == 2
            else "not_worker"
            if role != "worker"
            else "unknown"
        ),
        "occupation_group_soc": (
            f"occupation_category_{occat}" if occat > 0 else "unknown"
        ),
        "industry_group_naics": "unknown",
        "commute_mode": commute_mode,
        "commute_time_band": commute_time_band,
        "household_size": hhsize if hhsize >= 0 else None,
        "attrs_source": "nhts_training_data_generator",
        "nhts_hh_cbsa_code": str(cbsa_code),
    }

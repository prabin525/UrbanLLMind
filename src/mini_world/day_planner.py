from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from mini_world.activity_taxonomy import (
    planner_activity_name_to_code,
    planner_activity_vocab_text,
)

_ACTIVITY_NAME_TO_CODE = planner_activity_name_to_code()

_ACTIVITY_VOCAB = planner_activity_vocab_text()
_SEGMENT_RE = re.compile(
    r"^\s*(\d{2}:\d{2})-(\d{2}:\d{2}):([a-z_]+)(?::([^;]+))?\s*$"
)


def activity_vocab_text() -> str:
    return _ACTIVITY_VOCAB


# def get_step_plan_reminder() -> str:
#     return (
#         "HH:MM-HH:MM:activity[:tag];... (activities: "
#         f"{activity_vocab_text()})"
#     )


def _parse_hhmm(value: str, *, allow_2400: bool) -> Optional[int]:
    if value == "24:00":
        return 24 * 60 if allow_2400 else None
    try:
        hour, minute = value.split(":")
        h = int(hour)
        m = int(minute)
    except Exception:
        return None
    if h < 0 or h > 23 or m < 0 or m > 59:
        return None
    return h * 60 + m


def parse_day_plan_line(
    day_plan_line: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    line = str(day_plan_line or "").strip()
    if not line:
        return None, "empty_plan_line"

    parts = [part.strip() for part in line.split(";") if part.strip()]
    if len(parts) < 3 or len(parts) > 6:
        return None, "segment_count_out_of_range"

    segments: List[Dict[str, Any]] = []
    prev_end = -1
    for idx, part in enumerate(parts):
        match = _SEGMENT_RE.match(part)
        if match is None:
            return None, f"segment_{idx}_invalid_format"
        start_str, end_str, activity_str, tag_str = match.groups()
        activity = str(activity_str).strip().lower()
        if activity not in _ACTIVITY_NAME_TO_CODE:
            return None, f"segment_{idx}_unknown_activity"

        start_min = _parse_hhmm(start_str, allow_2400=False)
        end_min = _parse_hhmm(end_str, allow_2400=True)
        if start_min is None or end_min is None:
            return None, f"segment_{idx}_invalid_time"
        if end_min <= start_min:
            return None, f"segment_{idx}_non_positive_duration"
        if start_min < prev_end:
            return None, f"segment_{idx}_overlaps_previous"
        prev_end = end_min

        segment: Dict[str, Any] = {
            "start": start_str,
            "end": end_str,
            "start_min": start_min,
            "end_min": end_min,
            "activity": activity,
            "activity_code": _ACTIVITY_NAME_TO_CODE[activity],
        }
        if tag_str:
            segment["tag"] = str(tag_str).strip()
        segments.append(segment)

    return {
        "raw_line": line,
        "segments": segments,
    }, None


def get_plan_line(plan: Optional[Dict[str, Any]]) -> str:
    if not plan:
        return ""
    return str(plan.get("raw_line", "")).strip()


def summarize_plan_for_diagnostics(
    *,
    plan: Optional[Dict[str, Any]],
    status: Dict[str, Any],
) -> Dict[str, Any]:
    state = str(status.get("state", "planner_disabled"))
    plan_line = get_plan_line(plan)
    segment_count = 0
    first_activity = None
    if isinstance(plan, dict):
        segments = plan.get("segments", [])
        if isinstance(segments, list):
            segment_count = len(segments)
            if segments:
                first = segments[0]
                if isinstance(first, dict):
                    first_activity = first.get("activity")

    return {
        "enabled": state != "planner_disabled",
        "state": state,
        "day_index": status.get("day_index"),
        "attempted": bool(status.get("attempted", False)),
        "plan_present": bool(status.get("plan_present", False)),
        "parse_ok": bool(status.get("parse_ok", False)),
        "plan_chars": int(status.get("plan_chars", len(plan_line))),
        "segment_count": int(segment_count),
        "first_activity": first_activity,
    }

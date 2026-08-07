from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import lru_cache
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mini_world.activity_taxonomy import (  # type: ignore[import-not-found]
    activity_internal_label,
    activity_prompt_label,
    format_activity_codes_block,
    format_activity_notes_block,
    prompt_activity_vocab_text,
)


PROMPT_PATH = SRC_ROOT / "mini_world" / "prompts" / "memory_stream.yaml"


@lru_cache(maxsize=1)
def load_prompt_templates() -> Dict[str, Any]:
    with PROMPT_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_tags(tags: Any) -> List[str]:
    if tags is None:
        return []
    if isinstance(tags, (list, tuple, set)):
        return [str(tag).strip().lower() for tag in tags if str(tag).strip()]
    return [str(tags).strip().lower()]


def _safe_format(template: str, values: Dict[str, Any]) -> str:
    class _SafeFormatDict(dict):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return str(template or "").format_map(_SafeFormatDict(values))


def extra_profile_attributes(attrs: Dict[str, Any]) -> str:
    fields = [
        ("employment_status", "employment status"),
        ("work_schedule_type", "work schedule"),
        ("school_type", "school type"),
        ("subtype_hint", "subtype"),
        ("household_vehicle_count", "household vehicles"),
        ("household_income_band", "household income"),
        ("household_lifecycle_type", "household lifecycle"),
        ("worker_type", "worker type"),
        ("occupation_group_soc", "occupation group"),
        ("industry_group_naics", "industry group"),
        ("commute_mode", "commute mode"),
        ("commute_time_band", "commute time band"),
        ("household_size", "household size"),
    ]
    details: List[str] = []
    for key, label in fields:
        value = attrs.get(key)
        if value in (None, "", "unknown"):
            continue
        details.append(f"{label}: {value}")
    return "; ".join(details) if details else "none provided"


def build_shared_system_prompt(profile: Dict[str, Any]) -> str:
    prompts = load_prompt_templates()
    values = {
        "age": profile["age"],
        "gender": profile["gender"],
        "role": profile["role"],
        "city_name": profile["city_name"],
        "extra_profile_attributes": extra_profile_attributes(profile["attrs"]),
        "activity_vocab_prompt_labels": prompt_activity_vocab_text(),
    }
    return _safe_format(prompts.get("shared_system_prompt", ""), values)


def build_reflection_system_prompt(profile: Dict[str, Any]) -> str:
    prompts = load_prompt_templates()
    values = {
        "age": profile["age"],
        "gender": profile["gender"],
        "role": profile["role"],
        "city_name": profile["city_name"],
        "extra_profile_attributes": extra_profile_attributes(profile["attrs"]),
    }
    return _safe_format(prompts.get("reflection_system_prompt", ""), values)


def minute_to_time_str(minute_of_day: int) -> str:
    minute = max(0, min(int(minute_of_day), 24 * 60))
    if minute == 24 * 60:
        minute = 24 * 60 - 1
    return f"{minute // 60:02d}:{minute % 60:02d}"


def minute_to_datetime_str(day_str: str, minute_of_day: int) -> str:
    return f"{day_str} {minute_to_time_str(minute_of_day)}"


def format_memories(memories: Sequence[Dict[str, Any]]) -> str:
    if not memories:
        return "- No notable memories yet."
    lines: List[str] = []
    for memory in memories:
        label = "reflection" if memory.get("type") == "reflection" else "memory"
        text = " ".join(str(memory.get("text", "")).strip().split())
        if not text:
            continue
        lines.append(f"- [{memory['time_label']}] ({label}) {text}")
    return "\n".join(lines) if lines else "- No notable memories yet."


def format_reflection_memory_dump(memories: Sequence[Dict[str, Any]]) -> str:
    lines = [f"- [{memory['time_label']}] {memory['text']}" for memory in memories]
    return "\n".join(lines)


def render_day_planner_prompt(
    *,
    starting_activity_type: int,
    day_of_week: str,
    memories: Sequence[Dict[str, Any]],
) -> str:
    prompts = load_prompt_templates()
    prompt_values = {
        "day_of_week": day_of_week,
        "bfrom": activity_prompt_label(starting_activity_type, "Home"),
        "memory_section": format_memories(memories).replace("\n", "\n    "),
        "activity_vocab_prompt_labels": prompt_activity_vocab_text(),
        "activity_notes_block": format_activity_notes_block(indent=2),
        "event_context": "",
    }
    template = str(prompts.get("day_planner_prompt_template") or "")
    return _safe_format(template, prompt_values)


def render_decision_prompt(
    *,
    current_activity_type: int,
    day_of_week: str,
    day_str: str,
    day_plan_text: str,
    today_activity_table: str,
    memories: Sequence[Dict[str, Any]],
) -> str:
    prompts = load_prompt_templates()
    prompt_values = {
        "day_of_week": day_of_week,
        "day": day_str,
        "bfrom": activity_prompt_label(current_activity_type, "Unknown"),
        "plan_section": (day_plan_text or "").replace("\n", "\n    "),
        "today_activity_table": today_activity_table.replace("\n", "\n    "),
        "memory_section": format_memories(memories).replace("\n", "\n    "),
        "activity_codes_block": format_activity_codes_block(indent=4),
        "activity_notes_block": format_activity_notes_block(indent=4),
        "event_context": "",
    }
    template = str(prompts.get("decision_prompt_template") or "")
    return _safe_format(template, prompt_values)


def render_reflection_prompt(
    *,
    day_of_week: str,
    day_str: str,
    time_str: str,
    memories: Sequence[Dict[str, Any]],
) -> str:
    prompts = load_prompt_templates()
    prompt_values = {
        "day_of_week": day_of_week,
        "day": day_str,
        "time": time_str,
        "memory_dump": format_reflection_memory_dump(memories).replace(
            "\n", "\n    "
        ),
        "event_context": "",
    }
    template = str(prompts.get("reflection_prompt_template") or "")
    return _safe_format(template, prompt_values)


def build_today_activity_table(
    *,
    completed_segments: Sequence[Tuple[int, int, int]],
    current_activity_type: int,
    current_activity_start_minute: int,
    now_minute: int,
) -> str:
    lines: List[str] = []
    for start_minute, end_minute, activity_type in completed_segments:
        start_time = minute_to_time_str(start_minute)
        end_time = minute_to_time_str(end_minute)
        label = activity_prompt_label(activity_type, "Unknown")
        lines.append(f"{start_time}-{end_time} | {label}")
    if now_minute > current_activity_start_minute:
        start_time = minute_to_time_str(current_activity_start_minute)
        label = activity_prompt_label(current_activity_type, "Unknown")
        lines.append(f"{start_time}-now | {label}")
    return "\n".join(lines) if lines else "(no activity yet)"


def make_memory_entry(
    *,
    minute_of_day: int,
    day_str: str,
    day_of_week: str,
    text: str,
    tags: Iterable[str],
    importance: float,
    memory_type: str,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    entry = {
        "tick": int(minute_of_day),
        "time_label": minute_to_datetime_str(day_str, minute_of_day),
        "text": text,
        "tags": {str(tag).strip().lower() for tag in tags if str(tag).strip()},
        "importance": float(importance),
        "type": memory_type,
    }
    if day_of_week:
        entry["tags"].add(day_of_week.strip().lower())
    if metadata:
        entry.update(metadata)
    return entry


def make_observation_memory(
    *,
    minute_of_day: int,
    day_str: str,
    day_of_week: str,
    activity_type: int,
    minutes_here: int,
) -> Dict[str, Any]:
    location_label = activity_prompt_label(activity_type, "Unknown")
    location_tag = activity_internal_label(activity_type, "unknown")
    text = (
        f"At {minute_to_time_str(minute_of_day)} on {day_str} you were at "
        f"{location_label} for {max(0, int(minutes_here))} minutes."
    )
    return make_memory_entry(
        minute_of_day=minute_of_day,
        day_str=day_str,
        day_of_week=day_of_week,
        text=text,
        tags={location_tag},
        importance=0.7,
        memory_type="observation",
    )


def make_decision_memory(
    *,
    minute_of_day: int,
    day_str: str,
    day_of_week: str,
    activity_type: int,
    stay_minutes: int,
    rationale: str,
) -> Dict[str, Any]:
    location_label = activity_prompt_label(activity_type, "Activity")
    location_tag = activity_internal_label(activity_type, "activity")
    text = (
        f"Decided to go to {location_label} and planned to stay "
        f"~{int(stay_minutes)} minutes."
    )
    rationale_text = str(rationale or "").strip()
    if rationale_text:
        text = f"{text} Rationale: {rationale_text}"
    return make_memory_entry(
        minute_of_day=minute_of_day,
        day_str=day_str,
        day_of_week=day_of_week,
        text=text,
        tags={location_tag},
        importance=1.2,
        memory_type="decision",
        metadata={
            "activity_type": int(activity_type),
            "stay_minutes": int(stay_minutes),
            "rationale": rationale_text or None,
        },
    )


def make_reflection_memory(
    *,
    minute_of_day: int,
    day_str: str,
    day_of_week: str,
    summary: str,
) -> Dict[str, Any]:
    return make_memory_entry(
        minute_of_day=minute_of_day,
        day_str=day_str,
        day_of_week=day_of_week,
        text=f"Reflection summary: {summary}",
        tags=extract_reflection_tags(summary),
        importance=1.5,
        memory_type="reflection",
    )


def current_focus_tags(
    *,
    current_activity_type: int,
    day_of_week: str,
    attrs: Dict[str, Any],
) -> Set[str]:
    tags = {activity_internal_label(current_activity_type, "unknown")}
    if day_of_week:
        tags.add(day_of_week.strip().lower())
    for key in (
        "subtype_hint",
        "employment_status",
        "work_schedule_type",
        "school_type",
        "worker_type",
        "commute_mode",
    ):
        value = str(attrs.get(key, "")).strip().lower()
        if value and value != "unknown":
            tags.add(value)
    return tags


def score_memory_components(
    *,
    memory: Dict[str, Any],
    current_minute: int,
    focus_tags: Set[str],
    recency_weight: float,
    importance_weight: float,
    relevance_weight: float,
) -> Dict[str, Any]:
    memory_tick = _safe_int(memory.get("tick"), default=current_minute)
    age = max(1, current_minute - memory_tick)
    recency_score = recency_weight / age
    importance = _safe_float(memory.get("importance"), default=0.0)
    importance_score = importance_weight * importance
    overlap = len(focus_tags & set(_normalize_tags(memory.get("tags"))))
    relevance_score = relevance_weight * overlap
    total_score = recency_score + importance_score + relevance_score
    return {
        "age_ticks": age,
        "recency_score": recency_score,
        "importance_score": importance_score,
        "relevance_score": relevance_score,
        "overlap_count": overlap,
        "total_score": total_score,
    }


def select_memories(
    *,
    memories: Sequence[Dict[str, Any]],
    current_minute: int,
    current_activity_type: int,
    day_of_week: str,
    attrs: Dict[str, Any],
    max_context_memories: int,
    recency_weight: float,
    importance_weight: float,
    relevance_weight: float,
) -> List[Dict[str, Any]]:
    if not memories:
        return []
    focus_tags = current_focus_tags(
        current_activity_type=current_activity_type,
        day_of_week=day_of_week,
        attrs=attrs,
    )
    scored: List[Tuple[float, int, Dict[str, Any], Dict[str, Any]]] = []
    for idx, memory in enumerate(memories):
        components = score_memory_components(
            memory=memory,
            current_minute=current_minute,
            focus_tags=focus_tags,
            recency_weight=recency_weight,
            importance_weight=importance_weight,
            relevance_weight=relevance_weight,
        )
        scored.append(
            (
                float(components["total_score"]),
                -idx,
                memory,
                components,
            )
        )
    scored.sort(reverse=True)

    selected_entries: List[Tuple[float, int, Dict[str, Any], Dict[str, Any]]] = []
    selected_memory_ids: Set[int] = set()

    def _select_from_pool(
        pool: Sequence[Tuple[float, int, Dict[str, Any], Dict[str, Any]]],
        max_items: int,
    ) -> None:
        remaining = max_items
        for entry in pool:
            memory = entry[2]
            memory_id = id(memory)
            if memory_id in selected_memory_ids:
                continue
            selected_entries.append(entry)
            selected_memory_ids.add(memory_id)
            remaining -= 1
            if len(selected_entries) >= max_context_memories or remaining <= 0:
                return

    reflection_pool = [entry for entry in scored if entry[2].get("type") == "reflection"]
    _select_from_pool(reflection_pool, 2)

    decision_rationale_pool = []
    for entry in scored:
        memory = entry[2]
        if memory.get("type") != "decision":
            continue
        rationale = str(memory.get("rationale") or "").strip()
        if rationale:
            decision_rationale_pool.append(entry)
    _select_from_pool(decision_rationale_pool, 3)

    observation_pool = [
        entry for entry in scored if entry[2].get("type") == "observation"
    ]
    _select_from_pool(
        observation_pool,
        max_context_memories - len(selected_entries),
    )
    _select_from_pool(scored, max_context_memories - len(selected_entries))

    selected = [memory for _, _, memory, _ in selected_entries]
    selected.sort(key=lambda memory: int(memory.get("tick", 0)), reverse=True)
    return selected


_REFLECTION_TAG_KEYWORDS = {
    "home": ["home", "residential", "rest", "sleep"],
    "work": ["work", "office", "job", "employment", "shift"],
    "eat_meal": [
        "restaurant",
        "food",
        "eat",
        "meal",
        "lunch",
        "dinner",
        "breakfast",
        "cafe",
    ],
    "education": [
        "school",
        "class",
        "study",
        "student",
        "lecture",
        "daycare",
        "child care",
        "childcare",
    ],
    "recreational": ["recreation", "leisure", "park", "gym", "exercise"],
    "shopping": ["shopping", "shop", "store", "market"],
    "care": ["health", "doctor", "clinic", "hospital", "adult care"],
    "community": ["religious", "church", "temple", "community", "volunteer"],
    "social_visit": ["visit", "friends", "friend", "relatives", "family"],
    "other": [
        "errand",
        "errands",
        "transfer",
        "transport",
        "pickup",
        "drop off",
        "other",
    ],
}


def extract_reflection_tags(summary: str) -> Set[str]:
    tags: Set[str] = {"reflection"}
    text_lower = str(summary).lower()
    for tag, keywords in _REFLECTION_TAG_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            tags.add(tag)
    return tags


def build_gold_day_outline(
    day_str: str,
    dwell_blocks: Sequence[Dict[str, Any]],
) -> str:
    lines: List[str] = []
    for block in dwell_blocks:
        start_minute = _safe_int(block["start_minute"])
        end_minute = _safe_int(block["end_minute"])
        label = activity_prompt_label(_safe_int(block["activity_type"]), "Unknown")
        lines.append(
            f"- {day_str} {minute_to_time_str(start_minute)} to "
            f"{minute_to_time_str(end_minute)}: {label}"
        )
    return "\n".join(lines)


def strip_code_fence(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

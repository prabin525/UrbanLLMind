from __future__ import annotations

import html
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

try:
    from constants import NEW_LOC_TYPES
except ImportError:  # pragma: no cover - fallback for module-style execution
    from analysis.constants import NEW_LOC_TYPES


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_ORDER = {"day_planner": 0, "decision": 1, "reflection": 2}

PLANNER_DAY_RE = re.compile(r"A new day has started \(([^)]+)\)")
DECISION_TIME_RE = re.compile(
    r"- Time:\s*([A-Za-z]+),\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s+([0-9]{2}:[0-9]{2})"
)
REFLECTION_DAY_RE = re.compile(
    r"- Day:\s*([A-Za-z]+),\s*([0-9]{4}-[0-9]{2}-[0-9]{2})"
)
REFLECTION_TIME_RE = re.compile(r"- Time:\s*([0-9]{2}:[0-9]{2})")


STYLE_BLOCK = """
<style>
.agent-story-root {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #0f172a;
}
.agent-story-title {
  margin: 0 0 14px 0;
  font-size: 26px;
  font-weight: 700;
}
.agent-story-subtitle {
  margin: 0 0 20px 0;
  color: #475569;
  font-size: 14px;
}
.agent-story-profile-card {
  border: 1px solid #dbe2ea;
  border-radius: 14px;
  background: #f8fafc;
  padding: 14px 16px;
  margin: 0 0 18px 0;
}
.agent-story-profile-headline {
  margin: 0 0 10px 0;
  font-size: 18px;
  font-weight: 700;
}
.agent-story-profile-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 8px 14px;
}
.agent-story-profile-item {
  font-size: 13px;
  line-height: 1.45;
}
.agent-story-profile-item strong {
  color: #334155;
}
.agent-story-model-card {
  border: 1px solid #dbe2ea;
  border-radius: 16px;
  background: #ffffff;
  padding: 18px 18px 8px 18px;
  margin: 0 0 18px 0;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
}
.agent-story-model-card > summary {
  cursor: pointer;
  margin: 0 0 10px 0;
}
.agent-story-model-title {
  margin: 0 0 12px 0;
  font-size: 21px;
  font-weight: 700;
}
.agent-story-day {
  border-top: 1px solid #e2e8f0;
  padding: 12px 0 10px 0;
}
.agent-story-day:first-of-type {
  border-top: none;
  padding-top: 0;
}
.agent-story-day > summary {
  cursor: pointer;
}
.agent-story-day-header {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}
.agent-story-day-title {
  font-size: 17px;
  font-weight: 700;
}
.agent-story-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.03em;
  text-transform: uppercase;
}
.agent-story-badge-date {
  background: #e2e8f0;
  color: #334155;
}
.agent-story-badge-storm {
  background: #dbeafe;
  color: #1d4ed8;
}
.agent-story-section-title {
  margin: 14px 0 6px 0;
  font-size: 13px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #475569;
}
.agent-story-text {
  margin: 0;
  line-height: 1.5;
  font-size: 14px;
  white-space: pre-wrap;
}
.agent-story-muted {
  color: #64748b;
}
.agent-story-reasoning {
  margin: 8px 0 0 0;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  background: #f8fafc;
}
.agent-story-reasoning summary {
  padding: 8px 10px;
  font-size: 13px;
  font-weight: 600;
  color: #334155;
}
.agent-story-reasoning-body {
  padding: 0 10px 10px 10px;
}
.agent-story-table-wrap {
  overflow-x: auto;
}
.agent-story-table {
  width: 100%;
  border-collapse: collapse;
  margin: 4px 0 0 0;
}
.agent-story-table th,
.agent-story-table td {
  border: 1px solid #e2e8f0;
  padding: 8px 10px;
  vertical-align: top;
  font-size: 13px;
  text-align: left;
}
.agent-story-table th {
  background: #f8fafc;
  color: #334155;
  font-weight: 700;
}
.agent-story-time {
  white-space: nowrap;
  color: #1e293b;
  font-weight: 600;
}
</style>
"""


def _resolve_run_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def _cohort_label(agent_type: Any) -> Optional[str]:
    mapping = {1: "worker", 2: "student", 3: "homemaker"}
    code = _safe_int(agent_type)
    if code is None:
        return None
    return mapping.get(code)


def _gender_label(value: Any) -> Optional[str]:
    code = _safe_int(value)
    mapping = {0: "male", 1: "female"}
    if code is None:
        return None
    return mapping.get(code, str(code))


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _format_text(value: Any) -> str:
    if value is None:
        text = ""
    else:
        try:
            text = "" if pd.isna(value) else str(value)
        except Exception:
            text = str(value)
    return html.escape(text).replace("\n", "<br>")


def _extract_prompt_metadata(
    prompt_name: Any, user_prompt: Any
) -> dict[str, Optional[str]]:
    meta = {"prompt_day_name": None, "prompt_date": None, "prompt_time": None}
    if not isinstance(user_prompt, str):
        return meta

    prompt_name = str(prompt_name or "").strip()
    if prompt_name == "day_planner":
        match = PLANNER_DAY_RE.search(user_prompt)
        if match:
            meta["prompt_day_name"] = match.group(1).strip()
        return meta

    if prompt_name == "decision":
        match = DECISION_TIME_RE.search(user_prompt)
        if match:
            meta["prompt_day_name"] = match.group(1).strip()
            meta["prompt_date"] = match.group(2).strip()
            meta["prompt_time"] = match.group(3).strip()
        return meta

    if prompt_name == "reflection":
        day_match = REFLECTION_DAY_RE.search(user_prompt)
        if day_match:
            meta["prompt_day_name"] = day_match.group(1).strip()
            meta["prompt_date"] = day_match.group(2).strip()
        time_match = REFLECTION_TIME_RE.search(user_prompt)
        if time_match:
            meta["prompt_time"] = time_match.group(1).strip()
        return meta

    return meta


@lru_cache(maxsize=128)
def load_agent_story_events(
    run_dir: str | Path,
    agent_id: int,
) -> pd.DataFrame:
    folder = _resolve_run_path(run_dir)
    files = sorted(folder.glob("memory_stream_interactions_rank*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No memory_stream_interactions_rank*.jsonl files found in {folder}"
        )

    agent_id = int(agent_id)
    rows: list[dict[str, Any]] = []
    for file in files:
        with file.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if payload.get("event") != "interaction":
                    continue
                if _safe_int(payload.get("agent_id")) != agent_id:
                    continue
                payload["_source_file"] = file.name
                rows.append(payload)

    if not rows:
        return pd.DataFrame(
            columns=[
                "tick",
                "agent_id",
                "prompt_name",
                "reasoning_text",
                "assistant_response",
                "parsed",
                "user_prompt",
                "prompt_day_name",
                "prompt_date",
                "prompt_time",
                "day_num",
            ]
        )

    df = pd.DataFrame(rows)
    df["tick"] = pd.to_numeric(df["tick"], errors="coerce").fillna(-1).astype(int)

    meta_rows = [
        _extract_prompt_metadata(row.get("prompt_name"), row.get("user_prompt"))
        for row in df.to_dict("records")
    ]
    meta_df = pd.DataFrame(meta_rows)
    df = pd.concat([df.reset_index(drop=True), meta_df], axis=1)

    prompt_dates = pd.to_datetime(df["prompt_date"], errors="coerce")
    tick_day_num = (df["tick"] // 288) + 1
    df["day_num"] = tick_day_num.astype(int)
    if prompt_dates.notna().any():
        base_date = prompt_dates.dropna().min()
        dated_mask = prompt_dates.notna()
        df.loc[dated_mask, "day_num"] = (
            (prompt_dates[dated_mask] - base_date).dt.days + 1
        ).astype(int)

    df["prompt_order"] = (
        df["prompt_name"].map(PROMPT_ORDER).fillna(99).astype(int)
    )
    df = df.sort_values(
        ["day_num", "tick", "prompt_order", "prompt_time"],
        kind="stable",
    ).reset_index(drop=True)
    return df


@lru_cache(maxsize=32)
def load_agent_profile(
    input_folder: str | Path,
    agent_id: int,
) -> Optional[dict[str, Any]]:
    folder = _resolve_run_path(input_folder)
    agents_path = folder / "input_agents.txt"
    attrs_path = folder / "input_agent_attrs.csv"
    if not agents_path.exists():
        return None

    agent_id = int(agent_id)
    agents_df = pd.read_csv(agents_path, sep="\t")
    row_df = agents_df[agents_df["agent_id"] == agent_id].copy()
    if row_df.empty:
        return None

    if attrs_path.exists():
        attrs_df = pd.read_csv(attrs_path)
        row_df = row_df.merge(attrs_df, on="agent_id", how="left")

    row = row_df.iloc[0].to_dict()
    row["cohort_label"] = _cohort_label(row.get("agent_type"))
    row["gender_label"] = _gender_label(row.get("gender"))
    return row


def _first_non_empty(values: Iterable[Any]) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip()
        if text:
            return text
    return None


def _planner_text(row: pd.Series) -> str:
    parsed = row.get("parsed")
    if isinstance(parsed, dict):
        plan = parsed.get("day_plan_line")
        if isinstance(plan, str) and plan.strip():
            return plan.strip()
    response = row.get("assistant_response")
    return "" if response is None else str(response).strip()


def _decision_payload(row: pd.Series) -> dict[str, Any]:
    parsed = row.get("parsed")
    if isinstance(parsed, dict):
        return parsed
    response = row.get("assistant_response")
    if not isinstance(response, str):
        return {}
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _activity_display(code: Any) -> str:
    activity_code = _safe_int(code)
    if activity_code is None:
        return "N/A"
    label = NEW_LOC_TYPES.get(activity_code, f"Code {activity_code}")
    return f"{html.escape(str(label))} ({activity_code})"


def _render_reasoning(title: str, text: Any) -> str:
    if text is None or not str(text).strip():
        return (
            f'<div class="agent-story-text agent-story-muted">{html.escape(title)} '
            "not available.</div>"
        )
    return (
        '<details class="agent-story-reasoning">'
        f"<summary>{html.escape(title)}</summary>"
        f'<div class="agent-story-reasoning-body"><div class="agent-story-text">{_format_text(text)}</div></div>'
        "</details>"
    )


def _render_profile_block(profile: Optional[Mapping[str, Any]]) -> str:
    if not profile:
        return ""

    age = _safe_int(profile.get("age"))
    gender = profile.get("gender_label")
    cohort = profile.get("cohort_label")
    headline_parts = []
    if age is not None:
        headline_parts.append(f"{age}-year-old")
    if gender:
        headline_parts.append(str(gender))
    if cohort:
        headline_parts.append(str(cohort))
    headline = " ".join(headline_parts) if headline_parts else "Agent profile"

    detail_pairs = [
        ("Occupation", profile.get("occupation")),
        ("Employment", profile.get("employment_status")),
        ("Work schedule", profile.get("work_schedule_type")),
        ("School", profile.get("school_type")),
        ("Subtype", profile.get("subtype_hint")),
        ("Household size", profile.get("household_size")),
        ("Household children", profile.get("household_children")),
        ("Household adults", profile.get("household_adults")),
        ("Household elder", profile.get("household_elder")),
        ("Vehicles", profile.get("household_vehicle_count")),
        ("Income", profile.get("household_income_band")),
        ("Work building", profile.get("work_building_tag")),
    ]
    details = []
    for label, value in detail_pairs:
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        text = str(value).strip()
        if not text:
            continue
        details.append(
            '<div class="agent-story-profile-item">'
            f"<strong>{html.escape(label)}:</strong> {html.escape(text)}"
            "</div>"
        )

    return (
        '<div class="agent-story-profile-card">'
        '<div class="agent-story-section-title">Agent Profile</div>'
        f'<div class="agent-story-profile-headline">{html.escape(headline)}</div>'
        f'<div class="agent-story-profile-grid">{"".join(details)}</div>'
        "</div>"
    )


def _render_planner_block(
    planner_row: Optional[pd.Series],
    *,
    show_day_plan_reasoning: bool,
    show_day_plan: bool,
) -> str:
    if planner_row is None or not (
        show_day_plan_reasoning or show_day_plan
    ):
        return ""

    body: list[str] = ['<div class="agent-story-section-title">Day Planner</div>']
    if show_day_plan:
        planner_text = _planner_text(planner_row)
        body.append(
            f'<div class="agent-story-text">{_format_text(planner_text or "No planner output logged.")}</div>'
        )
    if show_day_plan_reasoning:
        body.append(
            _render_reasoning(
                "Planner reasoning",
                planner_row.get("reasoning_text"),
            )
        )
    return "".join(body)


def _render_decision_table(
    decision_rows: pd.DataFrame,
    *,
    show_step_reasoning: bool,
    show_step_activity: bool,
    show_step_rationale: bool,
    show_step_stay_minutes: bool,
) -> str:
    if not (
        show_step_reasoning
        or show_step_activity
        or show_step_rationale
        or show_step_stay_minutes
    ):
        return ""

    pieces = [
        '<div class="agent-story-section-title">'
        f"Step Decisions ({len(decision_rows)})"
        "</div>"
    ]
    if decision_rows.empty:
        pieces.append(
            '<div class="agent-story-text agent-story-muted">No step decisions logged for this day.</div>'
        )
        return "".join(pieces)

    headers = ['<th class="agent-story-time">Time</th>']
    if show_step_activity:
        headers.append("<th>Activity</th>")
    if show_step_rationale:
        headers.append("<th>Rationale</th>")
    if show_step_stay_minutes:
        headers.append("<th>Stay (min)</th>")
    if show_step_reasoning:
        headers.append("<th>Decision reasoning</th>")

    row_html: list[str] = []
    for _, row in decision_rows.iterrows():
        payload = _decision_payload(row)
        prompt_time = row.get("prompt_time")
        try:
            has_prompt_time = not pd.isna(prompt_time) and str(prompt_time).strip()
        except Exception:
            has_prompt_time = bool(prompt_time)
        time_label = (
            str(prompt_time).strip()
            if has_prompt_time
            else f"tick {row.get('tick', '?')}"
        )
        cells = [
            '<td class="agent-story-time">'
            f"{html.escape(time_label)}"
            "</td>"
        ]
        if show_step_activity:
            cells.append(
                f"<td>{_activity_display(payload.get('next_activity_type'))}</td>"
            )
        if show_step_rationale:
            cells.append(
                f"<td>{_format_text(payload.get('rationale') or 'N/A')}</td>"
            )
        if show_step_stay_minutes:
            stay = payload.get("stay_minutes")
            cells.append(
                f"<td>{html.escape(str(stay)) if stay is not None else 'N/A'}</td>"
            )
        if show_step_reasoning:
            cells.append(
                "<td>"
                + _render_reasoning("Decision reasoning", row.get("reasoning_text"))
                + "</td>"
            )
        row_html.append("<tr>" + "".join(cells) + "</tr>")

    pieces.append(
        '<div class="agent-story-table-wrap"><table class="agent-story-table">'
        f"<thead><tr>{''.join(headers)}</tr></thead>"
        f"<tbody>{''.join(row_html)}</tbody>"
        "</table></div>"
    )
    return "".join(pieces)


def _render_reflection_block(
    reflection_row: Optional[pd.Series],
    *,
    show_reflection_reasoning: bool,
    show_reflection: bool,
) -> str:
    if reflection_row is None or not (show_reflection_reasoning or show_reflection):
        return ""

    body: list[str] = ['<div class="agent-story-section-title">Reflection</div>']
    if show_reflection:
        reflection_text = reflection_row.get("assistant_response")
        body.append(
            f'<div class="agent-story-text">{_format_text(reflection_text or "No reflection output logged.")}</div>'
        )
    if show_reflection_reasoning:
        body.append(
            _render_reasoning(
                "Reflection reasoning",
                reflection_row.get("reasoning_text"),
            )
        )
    return "".join(body)


def _day_header(
    day_rows: pd.DataFrame,
    day_num: int,
    storm_day_names: Optional[set[str]],
) -> str:
    day_name = _first_non_empty(day_rows.get("prompt_day_name", [])) or f"Day {day_num}"
    date_text = _first_non_empty(day_rows.get("prompt_date", []))
    badges = []
    if date_text:
        badges.append(
            f'<span class="agent-story-badge agent-story-badge-date">{html.escape(date_text)}</span>'
        )
    if storm_day_names and day_name in storm_day_names:
        badges.append(
            '<span class="agent-story-badge agent-story-badge-storm">storm day</span>'
        )
    return (
        '<div class="agent-story-day-header">'
        f'<div class="agent-story-day-title">Day {day_num}: {html.escape(day_name)}</div>'
        f"{''.join(badges)}"
        "</div>"
    )


def render_agent_story_compare_html(
    *,
    agent_id: int,
    sim_runs: Mapping[str, str],
    run_input_folders: Optional[Mapping[str, str]] = None,
    models: Optional[Sequence[str]] = None,
    days: Optional[Sequence[int]] = None,
    model_display: Optional[Mapping[str, str]] = None,
    storm_day_names: Optional[set[str]] = None,
    show_profile: bool = True,
    show_day_plan_reasoning: bool = True,
    show_day_plan: bool = True,
    show_step_reasoning: bool = True,
    show_step_activity: bool = True,
    show_step_rationale: bool = True,
    show_step_stay_minutes: bool = True,
    show_reflection_reasoning: bool = True,
    show_reflection: bool = True,
) -> str:
    model_display = dict(model_display or {})
    chosen_models = list(models or sim_runs.keys())
    chosen_days = [int(day) for day in (days or [1])]
    storm_day_names = set(storm_day_names or set())

    parts = [STYLE_BLOCK, '<div class="agent-story-root">']
    parts.append(
        f'<div class="agent-story-title">Agent {int(agent_id)} Story Comparison</div>'
    )
    parts.append(
        '<div class="agent-story-subtitle">'
        f"Models: {html.escape(', '.join(model_display.get(m, m) for m in chosen_models))}"
        f" | Days: {html.escape(', '.join(str(day) for day in chosen_days))}"
        "</div>"
    )

    if show_profile and run_input_folders:
        for model_name in chosen_models:
            input_folder = run_input_folders.get(model_name)
            if not input_folder:
                continue
            profile = load_agent_profile(input_folder, int(agent_id))
            if profile:
                parts.append(_render_profile_block(profile))
                break

    for model_name in chosen_models:
        model_label = model_display.get(model_name, model_name)
        parts.append('<details class="agent-story-model-card" open>')
        parts.append("<summary>")
        parts.append(
            f'<div class="agent-story-model-title">{html.escape(model_label)}</div>'
        )
        parts.append("</summary>")

        run_dir = sim_runs.get(model_name)
        if run_dir is None:
            parts.append(
                '<div class="agent-story-text agent-story-muted">'
                f"Missing run mapping for model '{html.escape(model_name)}'."
                "</div>"
            )
            parts.append("</details>")
            continue

        try:
            model_rows = load_agent_story_events(run_dir, int(agent_id))
        except FileNotFoundError as exc:
            parts.append(
                f'<div class="agent-story-text agent-story-muted">{html.escape(str(exc))}</div>'
            )
            parts.append("</details>")
            continue

        if model_rows.empty:
            parts.append(
                '<div class="agent-story-text agent-story-muted">'
                "No interaction logs were found for this agent in the selected run."
                "</div>"
            )
            parts.append("</details>")
            continue

        for day_num in chosen_days:
            day_rows = model_rows[model_rows["day_num"] == day_num]
            parts.append('<details class="agent-story-day" open>')
            parts.append("<summary>")
            if day_rows.empty:
                parts.append(
                    '<div class="agent-story-day-header">'
                    f'<div class="agent-story-day-title">Day {day_num}</div>'
                    '<span class="agent-story-badge agent-story-badge-date">missing</span>'
                    "</div>"
                )
                parts.append("</summary>")
                parts.append(
                    '<div class="agent-story-text agent-story-muted">'
                    "No planner / decision / reflection events were logged for this day."
                    "</div>"
                )
                parts.append("</details>")
                continue

            parts.append(_day_header(day_rows, day_num, storm_day_names))
            parts.append("</summary>")

            planner_row = (
                day_rows[day_rows["prompt_name"] == "day_planner"].head(1)
            )
            reflection_row = (
                day_rows[day_rows["prompt_name"] == "reflection"].head(1)
            )
            planner_row_series = planner_row.iloc[0] if not planner_row.empty else None
            reflection_row_series = (
                reflection_row.iloc[0] if not reflection_row.empty else None
            )
            decisions = day_rows[day_rows["prompt_name"] == "decision"].copy()

            parts.append(
                _render_planner_block(
                    planner_row_series,
                    show_day_plan_reasoning=show_day_plan_reasoning,
                    show_day_plan=show_day_plan,
                )
            )
            parts.append(
                _render_decision_table(
                    decisions,
                    show_step_reasoning=show_step_reasoning,
                    show_step_activity=show_step_activity,
                    show_step_rationale=show_step_rationale,
                    show_step_stay_minutes=show_step_stay_minutes,
                )
            )
            parts.append(
                _render_reflection_block(
                    reflection_row_series,
                    show_reflection_reasoning=show_reflection_reasoning,
                    show_reflection=show_reflection,
                )
            )
            parts.append("</details>")

        parts.append("</details>")

    parts.append("</div>")
    return "".join(parts)


def show_agent_story_compare(**kwargs: Any) -> str:
    from IPython.display import HTML, display

    story_html = render_agent_story_compare_html(**kwargs)
    display(HTML(story_html))
    return story_html

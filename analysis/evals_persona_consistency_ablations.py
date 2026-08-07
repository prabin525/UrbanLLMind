"""Persona-consistency metrics for UrbanLLMind ablation analysis.

The metrics in this module focus on longitudinal continuity within a simulated
week. They deliberately avoid scoring reflection text directly; reflection and
memory quality should be tested by downstream behavior under ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

try:
    from constants import NEW_LOC_TYPES, SIM_LOG_LABEL_TO_NEW_ACTION
except ImportError:  # pragma: no cover - package-style fallback
    from analysis.constants import NEW_LOC_TYPES, SIM_LOG_LABEL_TO_NEW_ACTION


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_RUNS: Dict[str, str] = {
    "OSS_20B": "simulation_outputs/SF_1k_oss_20b",
    "OSS_120B": "simulation_outputs/SF_1k_oss_120b",
    "OSS_20B_FT_SF": "simulation_outputs/SF_1k_oss_20b_lora-sf-5epoch",
    "OSS_20B_FT_US": "simulation_outputs/SF_1k_oss_20b_lora-5epoch",
}

DEFAULT_INPUT_FOLDER = "Inputs/SF_agents_1K"
DEFAULT_INPUT_FOLDERS: Dict[str, str] = {
    model: DEFAULT_INPUT_FOLDER for model in DEFAULT_MODEL_RUNS
}

ROLE_LABELS = {
    1: "worker",
    2: "student",
    3: "homemaker",
}

ROLE_ANCHOR_ACTIVITY = {
    1: 2,  # worker -> Work
    2: 4,  # student -> Education
}

ROLE_ANCHOR_LABEL = {
    1: "Work",
    2: "Education",
}

METRIC_DEFINITIONS = {
    "role_anchor_adherence": (
        "For workers and students, the fraction of weekdays where the agent "
        "has a substantial primary-role activity block: Work for workers and "
        "Education for students."
    ),
    "anchor_start_time_stability": (
        "For workers and students with at least three observed weekday anchor "
        "starts, the within-agent standard deviation of the first daily "
        "Work/Education start time."
    ),
    "weekday_routine_continuity": (
        "For each agent, the mean normalized edit-distance similarity across "
        "all pairs of Monday-Friday compressed activity chains."
    ),
    "weekday_routine_similarity_delta": (
        "Weekday routine continuity minus a matched control similarity from "
        "different agents with the same role and day of week in the same run."
    ),
}


@dataclass(frozen=True)
class PersonaEvalResult:
    """Container returned by ``evaluate_models``."""

    summary: pd.DataFrame
    agent_metrics: pd.DataFrame
    agent_days: pd.DataFrame


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _safe_int(value: Any) -> Optional[int]:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _time_to_minutes(value: Any) -> int:
    text = str(value)
    hour, minute = text.split(":")
    return int(hour) * 60 + int(minute)


def _activity_label(code: Any) -> str:
    try:
        return str(NEW_LOC_TYPES.get(int(code), str(code)))
    except Exception:
        return str(code)


def compress_adjacent(sequence: Iterable[Any]) -> list[int]:
    """Collapse adjacent duplicate activity codes."""

    out: list[int] = []
    for raw in sequence:
        code = _safe_int(raw)
        if code is None:
            continue
        if not out or out[-1] != code:
            out.append(code)
    return out


def format_activity_chain(sequence: Iterable[Any]) -> str:
    """Return a compact human-readable activity chain."""

    codes = compress_adjacent(sequence)
    return " -> ".join(_activity_label(code) for code in codes)


def edit_distance(left: Iterable[Any], right: Iterable[Any]) -> int:
    """Levenshtein edit distance for short activity-code sequences."""

    a = list(left)
    b = list(right)
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, av in enumerate(a, start=1):
        current = [i]
        for j, bv in enumerate(b, start=1):
            substitution = previous[j - 1] + (0 if av == bv else 1)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def normalized_edit_similarity(left: Iterable[Any], right: Iterable[Any]) -> float:
    """Similarity in [0, 1], where 1 means identical compressed chains."""

    a = compress_adjacent(left)
    b = compress_adjacent(right)
    denom = max(len(a), len(b))
    if denom == 0:
        return np.nan
    return float(1.0 - (edit_distance(a, b) / denom))


def load_agent_profiles(input_folder: str | Path) -> pd.DataFrame:
    input_path = resolve_path(input_folder)
    agents_path = input_path / "input_agents.txt"
    attrs_path = input_path / "input_agent_attrs.csv"
    if not agents_path.exists():
        raise FileNotFoundError(f"Missing input agents file: {agents_path}")

    profiles = pd.read_csv(agents_path, sep="\t")
    if attrs_path.exists():
        attrs = pd.read_csv(attrs_path)
        profiles = profiles.merge(attrs, on="agent_id", how="left")

    profiles["agent_id"] = pd.to_numeric(
        profiles["agent_id"], errors="coerce"
    ).astype("Int64")
    profiles["agent_type"] = pd.to_numeric(
        profiles["agent_type"], errors="coerce"
    ).astype("Int64")
    profiles["role_label"] = profiles["agent_type"].map(
        lambda value: ROLE_LABELS.get(_safe_int(value), "unknown")
    )
    return profiles


def load_activity_log(run_folder: str | Path) -> pd.DataFrame:
    run_path = resolve_path(run_folder)
    files = sorted(run_path.glob("activity_log_rank*.csv"))
    if not files:
        raise FileNotFoundError(f"No activity_log_rank*.csv files in {run_path}")

    frames = []
    for file in files:
        frame = pd.read_csv(file)
        if not frame.empty:
            frame["rank"] = int(file.stem.replace("activity_log_rank", ""))
        frames.append(frame)

    activity = pd.concat(frames, ignore_index=True, sort=False)
    if activity.empty:
        raise ValueError(f"Activity log files are empty in {run_path}")

    required = {"agent_id", "date", "location_type", "arrival_time", "departure_time"}
    missing = sorted(required.difference(activity.columns))
    if missing:
        raise ValueError(f"Activity log is missing columns: {missing}")

    activity = activity[activity["location_type"].notna()].copy()
    activity["agent_id"] = pd.to_numeric(
        activity["agent_id"], errors="coerce"
    ).astype("Int64")
    activity["date"] = pd.to_datetime(activity["date"], errors="coerce")
    activity["activity_code"] = activity["location_type"].map(
        lambda value: SIM_LOG_LABEL_TO_NEW_ACTION.get(
            str(value).strip().lower(), 9
        )
    )
    activity["arrival_minutes"] = activity["arrival_time"].map(_time_to_minutes)
    activity["departure_minutes"] = activity["departure_time"].map(_time_to_minutes)
    activity["duration_minutes"] = (
        activity["departure_minutes"] - activity["arrival_minutes"]
    )
    activity.loc[activity["duration_minutes"] <= 0, "duration_minutes"] += 24 * 60
    return activity.sort_values(["agent_id", "date", "arrival_minutes"])


def build_agent_day_table(
    run_folder: str | Path,
    input_folder: str | Path = DEFAULT_INPUT_FOLDER,
    *,
    anchor_min_minutes: int = 120,
) -> pd.DataFrame:
    """Build one row per agent-day with compressed chains and anchor fields."""

    activity = load_activity_log(run_folder)
    profiles = load_agent_profiles(input_folder)
    profile_cols = [
        col
        for col in [
            "agent_id",
            "agent_type",
            "role_label",
            "age",
            "gender",
            "employment_status",
            "work_schedule_type",
            "school_type",
            "occupation",
            "household_size",
            "household_children",
        ]
        if col in profiles.columns
    ]

    sequences = (
        activity.groupby(["agent_id", "date"], sort=False)["activity_code"]
        .apply(list)
        .reset_index(name="activity_chain_raw")
    )
    sequences["activity_chain"] = sequences["activity_chain_raw"].map(
        compress_adjacent
    )
    sequences["activity_chain_text"] = sequences["activity_chain"].map(
        format_activity_chain
    )
    sequences["activity_chain_length"] = sequences["activity_chain"].map(len)

    durations = (
        activity.pivot_table(
            index=["agent_id", "date"],
            columns="activity_code",
            values="duration_minutes",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    rename_duration = {
        code: f"{_activity_label(code).lower().replace(' ', '_')}_minutes"
        for code in NEW_LOC_TYPES
    }
    durations = durations.rename(columns=rename_duration)

    starts = (
        activity.groupby(["agent_id", "date", "activity_code"])["arrival_minutes"]
        .min()
        .unstack("activity_code")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    rename_start = {
        code: f"{_activity_label(code).lower().replace(' ', '_')}_first_start"
        for code in NEW_LOC_TYPES
    }
    starts = starts.rename(columns=rename_start)

    days = sequences.merge(durations, on=["agent_id", "date"], how="left")
    days = days.merge(starts, on=["agent_id", "date"], how="left")
    days = days.merge(profiles[profile_cols], on="agent_id", how="left")
    days["weekday"] = days["date"].dt.weekday
    days["day_name"] = days["date"].dt.day_name()
    days["is_weekend"] = days["weekday"] >= 5

    days["role_anchor_activity"] = days["agent_type"].map(
        lambda value: ROLE_ANCHOR_ACTIVITY.get(_safe_int(value))
    )
    days["role_anchor_label"] = days["agent_type"].map(
        lambda value: ROLE_ANCHOR_LABEL.get(_safe_int(value))
    )

    def anchor_minutes(row: pd.Series) -> float:
        label = row.get("role_anchor_label")
        if not label:
            return np.nan
        col = f"{label.lower().replace(' ', '_')}_minutes"
        return float(row.get(col, 0.0))

    def anchor_start(row: pd.Series) -> float:
        label = row.get("role_anchor_label")
        if not label:
            return np.nan
        col = f"{label.lower().replace(' ', '_')}_first_start"
        value = row.get(col, np.nan)
        return float(value) if pd.notna(value) else np.nan

    days["role_anchor_minutes"] = days.apply(anchor_minutes, axis=1)
    days["role_anchor_start"] = days.apply(anchor_start, axis=1)
    days["role_anchor_present"] = np.where(
        days["role_anchor_activity"].notna(),
        days["role_anchor_minutes"] >= float(anchor_min_minutes),
        np.nan,
    )
    return days.sort_values(["agent_id", "date"]).reset_index(drop=True)


def _weekday_pair_similarity(values: pd.DataFrame) -> tuple[float, int]:
    chains = list(values["activity_chain"])
    if len(chains) < 2:
        return np.nan, 0
    scores = [
        normalized_edit_similarity(left, right)
        for left, right in combinations(chains, 2)
    ]
    scores = [score for score in scores if pd.notna(score)]
    if not scores:
        return np.nan, 0
    return float(np.mean(scores)), len(scores)


def _compute_control_similarity(
    weekday_days: pd.DataFrame,
    *,
    seed: int,
    samples_per_agent_day: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sorted_days = weekday_days.sort_values(["agent_id", "weekday"]).reset_index(
        drop=True
    )
    groups: dict[tuple[int, int], np.ndarray] = {}
    for key, group in sorted_days.groupby(["agent_type", "weekday"], dropna=False):
        groups[key] = group.index.to_numpy()

    rows: list[dict[str, Any]] = []
    for agent_id, agent_days in sorted_days.groupby("agent_id", sort=True):
        control_scores: list[float] = []
        for idx, row in agent_days.iterrows():
            key = (row["agent_type"], row["weekday"])
            pool = groups.get(key, np.array([], dtype=int))
            if len(pool) == 0:
                continue
            pool = np.array(
                [
                    candidate
                    for candidate in pool
                    if sorted_days.loc[candidate, "agent_id"] != agent_id
                ],
                dtype=int,
            )
            if len(pool) == 0:
                continue
            sample_size = min(int(samples_per_agent_day), len(pool))
            chosen = rng.choice(pool, size=sample_size, replace=False)
            for candidate in chosen:
                score = normalized_edit_similarity(
                    row["activity_chain"],
                    sorted_days.loc[candidate, "activity_chain"],
                )
                if pd.notna(score):
                    control_scores.append(score)
        rows.append(
            {
                "agent_id": agent_id,
                "matched_control_similarity": (
                    float(np.mean(control_scores)) if control_scores else np.nan
                ),
                "matched_control_pair_count": len(control_scores),
            }
        )
    return pd.DataFrame(rows)


def compute_agent_metrics(
    agent_days: pd.DataFrame,
    *,
    seed: int = 42,
    control_samples_per_agent_day: int = 20,
    min_anchor_start_days: int = 3,
) -> pd.DataFrame:
    """Return one row per agent with the four persona-consistency metrics."""

    profile_cols = [
        col
        for col in [
            "agent_id",
            "agent_type",
            "role_label",
            "age",
            "gender",
            "employment_status",
            "work_schedule_type",
            "school_type",
            "occupation",
            "household_size",
            "household_children",
        ]
        if col in agent_days.columns
    ]
    agents = (
        agent_days[profile_cols]
        .sort_values("agent_id")
        .drop_duplicates("agent_id")
        .reset_index(drop=True)
    )

    weekdays = agent_days[~agent_days["is_weekend"]].copy()

    anchor_rows = weekdays[weekdays["role_anchor_activity"].notna()].copy()
    anchor_rows["role_anchor_present_num"] = pd.to_numeric(
        anchor_rows["role_anchor_present"], errors="coerce"
    )
    anchor = (
        anchor_rows.groupby("agent_id")["role_anchor_present_num"]
        .agg(role_anchor_adherence="mean", role_anchor_weekday_count="count")
        .reset_index()
    )

    start_values = anchor_rows[
        anchor_rows["role_anchor_start"].notna()
        & anchor_rows["role_anchor_present"].fillna(False)
    ].copy()
    start_stability = (
        start_values.groupby("agent_id")["role_anchor_start"]
        .agg(["count", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "anchor_start_observation_count",
                "std": "anchor_start_time_stability",
            }
        )
    )
    start_stability.loc[
        start_stability["anchor_start_observation_count"] < min_anchor_start_days,
        "anchor_start_time_stability",
    ] = np.nan

    continuity_rows = []
    for agent_id, group in weekdays.groupby("agent_id", sort=True):
        score, pair_count = _weekday_pair_similarity(group)
        continuity_rows.append(
            {
                "agent_id": agent_id,
                "weekday_routine_continuity": score,
                "weekday_pair_count": pair_count,
            }
        )
    continuity = pd.DataFrame(continuity_rows)

    controls = _compute_control_similarity(
        weekdays,
        seed=seed,
        samples_per_agent_day=control_samples_per_agent_day,
    )

    metrics = agents.merge(anchor, on="agent_id", how="left")
    metrics = metrics.merge(start_stability, on="agent_id", how="left")
    metrics = metrics.merge(continuity, on="agent_id", how="left")
    metrics = metrics.merge(controls, on="agent_id", how="left")
    metrics["weekday_routine_similarity_delta"] = (
        metrics["weekday_routine_continuity"]
        - metrics["matched_control_similarity"]
    )
    return metrics


def _role_metric(
    agent_metrics: pd.DataFrame,
    *,
    role_code: int,
    column: str,
    agg: str,
) -> float:
    values = pd.to_numeric(
        agent_metrics.loc[agent_metrics["agent_type"] == role_code, column],
        errors="coerce",
    ).dropna()
    if values.empty:
        return np.nan
    if agg == "mean":
        return float(values.mean())
    if agg == "median":
        return float(values.median())
    raise ValueError(f"Unsupported aggregation: {agg}")


def summarize_model_metrics(
    model_name: str,
    agent_days: pd.DataFrame,
    agent_metrics: pd.DataFrame,
) -> dict[str, Any]:
    """Build a compact model-level summary row."""

    summary: dict[str, Any] = {
        "model": model_name,
        "n_agents": int(agent_metrics["agent_id"].nunique()),
        "n_agent_days": int(agent_days[["agent_id", "date"]].drop_duplicates().shape[0]),
        "worker_role_anchor_adherence_mean": _role_metric(
            agent_metrics, role_code=1, column="role_anchor_adherence", agg="mean"
        ),
        "student_role_anchor_adherence_mean": _role_metric(
            agent_metrics, role_code=2, column="role_anchor_adherence", agg="mean"
        ),
        "worker_anchor_start_std_median_minutes": _role_metric(
            agent_metrics,
            role_code=1,
            column="anchor_start_time_stability",
            agg="median",
        ),
        "student_anchor_start_std_median_minutes": _role_metric(
            agent_metrics,
            role_code=2,
            column="anchor_start_time_stability",
            agg="median",
        ),
    }

    for column in [
        "weekday_routine_continuity",
        "matched_control_similarity",
        "weekday_routine_similarity_delta",
    ]:
        values = pd.to_numeric(agent_metrics[column], errors="coerce").dropna()
        summary[f"{column}_mean"] = float(values.mean()) if len(values) else np.nan
        summary[f"{column}_median"] = (
            float(values.median()) if len(values) else np.nan
        )
        summary[f"{column}_n"] = int(len(values))
    return summary


def evaluate_run(
    model_name: str,
    run_folder: str | Path,
    input_folder: str | Path = DEFAULT_INPUT_FOLDER,
    *,
    anchor_min_minutes: int = 120,
    seed: int = 42,
    control_samples_per_agent_day: int = 20,
    min_anchor_start_days: int = 3,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    agent_days = build_agent_day_table(
        run_folder,
        input_folder=input_folder,
        anchor_min_minutes=anchor_min_minutes,
    )
    agent_metrics = compute_agent_metrics(
        agent_days,
        seed=seed,
        control_samples_per_agent_day=control_samples_per_agent_day,
        min_anchor_start_days=min_anchor_start_days,
    )
    agent_days.insert(0, "model", model_name)
    agent_metrics.insert(0, "model", model_name)
    summary = summarize_model_metrics(model_name, agent_days, agent_metrics)
    return summary, agent_metrics, agent_days


def evaluate_models(
    model_runs: Mapping[str, str | Path] = DEFAULT_MODEL_RUNS,
    input_folders: Optional[Mapping[str, str | Path]] = None,
    *,
    anchor_min_minutes: int = 120,
    seed: int = 42,
    control_samples_per_agent_day: int = 20,
    min_anchor_start_days: int = 3,
) -> PersonaEvalResult:
    input_folders = dict(input_folders or {})
    summaries: list[dict[str, Any]] = []
    agent_metric_frames: list[pd.DataFrame] = []
    agent_day_frames: list[pd.DataFrame] = []

    for model_name, run_folder in model_runs.items():
        input_folder = input_folders.get(model_name, DEFAULT_INPUT_FOLDER)
        summary, agent_metrics, agent_days = evaluate_run(
            model_name,
            run_folder,
            input_folder=input_folder,
            anchor_min_minutes=anchor_min_minutes,
            seed=seed,
            control_samples_per_agent_day=control_samples_per_agent_day,
            min_anchor_start_days=min_anchor_start_days,
        )
        summaries.append(summary)
        agent_metric_frames.append(agent_metrics)
        agent_day_frames.append(agent_days)

    return PersonaEvalResult(
        summary=pd.DataFrame(summaries),
        agent_metrics=pd.concat(agent_metric_frames, ignore_index=True),
        agent_days=pd.concat(agent_day_frames, ignore_index=True),
    )


def select_representative_agents(
    agent_metrics: pd.DataFrame,
    *,
    score_col: str = "weekday_routine_continuity",
    n_per_model: int = 1,
) -> pd.DataFrame:
    """Select high- and low-scoring agents per model for story inspection."""

    rows: list[pd.DataFrame] = []
    base = agent_metrics.copy()
    base[score_col] = pd.to_numeric(base[score_col], errors="coerce")
    base = base[base[score_col].notna()].copy()
    for model_name, group in base.groupby("model", sort=False):
        sorted_group = group.sort_values(
            [score_col, "agent_id"], ascending=[False, True]
        )
        high = sorted_group.head(n_per_model).copy()
        high["example_type"] = "high"
        high["example_rank"] = np.arange(1, len(high) + 1)
        low = sorted_group.tail(n_per_model).sort_values(
            [score_col, "agent_id"], ascending=[True, True]
        ).copy()
        low["example_type"] = "low"
        low["example_rank"] = np.arange(1, len(low) + 1)
        rows.extend([high, low])
    if not rows:
        return pd.DataFrame()
    columns = [
        "model",
        "example_type",
        "example_rank",
        "agent_id",
        "role_label",
        "agent_type",
        score_col,
        "weekday_routine_similarity_delta",
        "matched_control_similarity",
        "role_anchor_adherence",
        "anchor_start_time_stability",
    ]
    available_columns = [col for col in columns if col in rows[0].columns]
    return pd.concat(rows, ignore_index=True)[available_columns]


def get_agent_weekday_chains(
    agent_days: pd.DataFrame,
    *,
    model: str,
    agent_id: int,
) -> pd.DataFrame:
    """Return compact weekday chains for notebook inspection."""

    subset = agent_days[
        (agent_days["model"] == model)
        & (agent_days["agent_id"].astype(int) == int(agent_id))
        & (~agent_days["is_weekend"])
    ].copy()
    keep_cols = [
        "model",
        "agent_id",
        "date",
        "day_name",
        "role_label",
        "activity_chain_text",
        "role_anchor_present",
        "role_anchor_start",
        "role_anchor_minutes",
    ]
    return subset[[col for col in keep_cols if col in subset.columns]].sort_values(
        "date"
    )


def _round_for_print(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(4)
    return out


if __name__ == "__main__":
    result = evaluate_models()
    print(_round_for_print(result.summary).to_string(index=False))

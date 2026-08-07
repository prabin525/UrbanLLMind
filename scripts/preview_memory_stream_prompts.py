#!/usr/bin/env python3
"""Render memory-stream prompts for a specific agent without running the sim.

This is a local debugging / inspection utility. It:
1) Loads a config YAML (for prompt-related params),
2) Loads an agent row from input_agents.txt (+ optional attrs sidecar),
3) Builds a lightweight fake model/agent with the same fields the prompt
   renderer expects,
4) Prints the rendered system, day-planner, and decision prompts.

No LLM calls are made.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mini_world.activity_taxonomy import activity_prompt_label  # noqa: E402
from mini_world.agent_attrs_loader import (  # noqa: E402
    default_agent_attrs,
    load_optional_agent_attrs,
)
from mini_world.agent_types import parse_agent_type  # noqa: E402
from mini_world.memory_stream_async import MemoryStreamPolicy  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview rendered memory-stream prompts for one agent."
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to a simulation YAML config (default: "
            "input_GABM_SERVER_calibrated_new.yaml if present, "
            "else input_GABM.yaml)"
        ),
    )
    parser.add_argument(
        "--input-data-folder",
        default=None,
        help=(
            "Input folder containing input_agents.txt (default: from config "
            "input_data_folder if present, else Inputs/InputData_100)"
        ),
    )
    parser.add_argument(
        "--agent-index",
        type=int,
        default=0,
        help="0-based row index into input_agents.txt (default: 0)",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=None,
        help="Agent ID to preview (overrides --agent-index if provided)",
    )
    parser.add_argument(
        "--tick",
        type=int,
        default=85,
        help=(
            "Simulation tick for prompt rendering (default: 85, roughly first "
            "decision after initial home dwell)"
        ),
    )
    parser.add_argument(
        "--activity-start-tick",
        type=int,
        default=0,
        help=(
            "Agent activity_start_tick used for prompt "
            "'minutes_here' (default: 0)"
        ),
    )
    parser.add_argument(
        "--current-activity",
        type=int,
        default=1,
        help="Current runtime activity code (default: 1 = home)",
    )
    parser.add_argument(
        "--day-plan-text",
        default=None,
        help=(
            "Optional planner output text to inject before rendering the "
            "decision prompt."
        ),
    )
    parser.add_argument(
        "--seed-decision-snapshot",
        action="store_true",
        default=True,
        help=(
            "Mimic runtime decision flow by recording a state snapshot before "
            "rendering the decision prompt (default: on)."
        ),
    )
    parser.add_argument(
        "--no-seed-decision-snapshot",
        dest="seed_decision_snapshot",
        action="store_false",
        help="Disable pre-decision state snapshot seeding.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional file path to write the rendered prompts.",
    )
    return parser.parse_args()


def _choose_default_config(repo_root: Path) -> Path:
    preferred = repo_root / "input_GABM_SERVER_calibrated_new.yaml"
    if preferred.exists():
        return preferred
    fallback = repo_root / "input_GABM.yaml"
    return fallback


def _load_params(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config is not a mapping: {config_path}")
    return dict(data)


def _resolve_input_data_folder(
    repo_root: Path,
    params: Dict[str, Any],
    arg_value: Optional[str],
) -> Path:
    def _resolve_candidate(raw: str) -> Path:
        p = Path(str(raw))
        if p.is_absolute():
            return p
        candidate = (repo_root / p).resolve()
        if candidate.exists():
            return candidate
        inputs_candidate = (repo_root / "Inputs" / p).resolve()
        if inputs_candidate.exists():
            return inputs_candidate
        return candidate

    if arg_value:
        return _resolve_candidate(arg_value)

    cfg_value = params.get("input_data_folder")
    if cfg_value:
        return _resolve_candidate(str(cfg_value))

    default_path = repo_root / "Inputs" / "InputData_100"
    if default_path.exists():
        return default_path
    legacy_default = (repo_root / "InputData").resolve()
    if legacy_default.exists():
        return legacy_default
    return default_path


def _parse_survey_start_date(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if value is None:
        return datetime.strptime("2025/09/08", "%Y/%m/%d")
    text = str(value).strip()
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported survey_start_date format: {value!r}")


@dataclass
class _FakeSchedule:
    tick: int


@dataclass
class _FakeRunner:
    schedule: _FakeSchedule


class _FakeModel:
    def __init__(self, params: Dict[str, Any], tick: int) -> None:
        self.params = dict(params)
        self.rank = 0
        self.out_folder_name = str(Path("tmp") / "prompt_preview")
        Path(self.out_folder_name).mkdir(parents=True, exist_ok=True)
        self.runner = _FakeRunner(schedule=_FakeSchedule(tick=int(tick)))
        self.survey_start_date = _parse_survey_start_date(
            self.params.get("survey_start_date", "2025/09/08")
        )


class _FakeAgent:
    def __init__(
        self,
        *,
        model: _FakeModel,
        agent_row: Dict[str, Any],
        agent_attrs: Dict[str, Any],
        current_activity: int,
        activity_start_tick: int,
    ) -> None:
        self.model = model
        self.id = int(agent_row["agent_id"])
        self.uid = (self.id, 0, 0)
        self.age = int(agent_row.get("age", 0))
        self.gender = int(agent_row.get("gender", 0))
        self.agent_type = parse_agent_type(
            agent_row.get("agent_type"),
            context=f"prompt preview agent_id={self.id} agent_type",
        )
        self.decision_policy = str(
            agent_row.get("decision_policy", "memory_stream_llm")
        )
        self.agent_attrs = dict(agent_attrs)
        self.llm_endpoint = str(
            model.params.get("llm_endpoint", "http://localhost:11434/v1")
        )

        # Runtime state fields used by prompt construction
        self.activity_type = int(current_activity)
        self.activity_start_tick = int(activity_start_tick)
        self.ttravel = -1
        self.tdwell = 0
        self.bfrom = int(agent_row.get("home", -1))
        self.bto = -1

        self.buildings = [0, 0, 0, 0]
        self.buildings[1] = int(agent_row.get("home", -1))
        self.buildings[2] = int(agent_row.get("work", -1))
        self.buildings[3] = int(agent_row.get("school", -1))

        # Needs are legacy but the prompt builder still expects these fields.
        self.work_need = [0.2, 0.01]
        self.food_need = [0.2, 0.01]
        self.social_need = [0.2, 0.01]
        self.errands_need = [0.2, 0.01]
        self.rest_need = [0.2, 0.01]

        # Daily history / planner state fields used in prompt construction
        self.today_segments: List[Tuple[int, int, int]] = []
        self.day_plan_state: Dict[str, Any] = {}
        self.day_plan_status: Dict[str, Any] = {}

    def _tick_to_datetime(self, tick: int) -> Tuple[str, str]:
        minutes_per_tick = 5
        ticks_per_day = 288
        day_offset = int(tick) // ticks_per_day
        minutes_today = (int(tick) % ticks_per_day) * minutes_per_tick
        current_date = self.model.survey_start_date + timedelta(
            days=day_offset
        )
        hours = minutes_today // 60
        minutes = minutes_today % 60
        return current_date.strftime("%Y-%m-%d"), f"{hours:02d}:{minutes:02d}"

    def get_today_activity_table(self) -> str:
        day_start = (self.model.runner.schedule.tick // 288) * 288
        lines: List[str] = []
        for seg_start, seg_end, activity_type in self.today_segments:
            start_time = self._tick_to_datetime(seg_start)[1]
            end_time = self._tick_to_datetime(seg_end)[1]
            label = activity_prompt_label(activity_type, "Unknown")
            lines.append(f"{start_time}-{end_time} | {label}")
        if self.ttravel == 0 and self.activity_start_tick is not None:
            seg_start = max(self.activity_start_tick, day_start)
            start_time = self._tick_to_datetime(seg_start)[1]
            label = activity_prompt_label(self.activity_type, "Unknown")
            lines.append(f"{start_time}-now | {label}")
        if not lines:
            return "(no activity yet)"
        return "\n".join(lines)


def _load_agents_df(input_data_folder: Path) -> pd.DataFrame:
    path = input_data_folder / "input_agents.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing input_agents file: {path}")
    agents = pd.read_csv(path, sep="\t")
    if "agent_type" not in agents.columns:
        raise ValueError(f"{path} is missing required column 'agent_type'")
    return agents


def _get_agent_row(
    agents_df: pd.DataFrame,
    *,
    agent_index: int,
    agent_id: Optional[int],
) -> Dict[str, Any]:
    if agent_id is not None:
        matches = agents_df[agents_df["agent_id"].astype(int) == int(agent_id)]
        if matches.empty:
            raise ValueError(
                f"agent_id {agent_id} not found in input_agents.txt"
            )
        row = matches.iloc[0]
    else:
        if agent_index < 0 or agent_index >= len(agents_df):
            raise IndexError(
                f"agent_index {agent_index} out of range "
                f"(0..{len(agents_df)-1})"
            )
        row = agents_df.iloc[int(agent_index)]
    data = row.to_dict()
    # Runtime expects these fields in many setups;
    # provide defaults for previews.
    data.setdefault("decision_policy", "memory_stream_llm")
    return data


def _load_agent_attrs_for_row(
    input_data_folder: Path,
    agent_row: Dict[str, Any],
) -> Dict[str, Any]:
    agent_id = int(agent_row["agent_id"])
    attrs_result = load_optional_agent_attrs(
        input_data_folder.as_posix(), [agent_id]
    )
    attrs = attrs_result.attrs_by_agent_id.get(agent_id)
    if attrs is None:
        attrs = default_agent_attrs()
    return dict(attrs)


def _render_prompt_bundle(
    policy: MemoryStreamPolicy,
    *,
    inject_day_plan_text: Optional[str],
    seed_decision_snapshot: bool,
) -> Dict[str, str]:
    current_tick = policy._current_tick()

    # Day planner prompt (runtime selects reflection memories only; often none)
    planner_memories = policy._select_planner_reflection_memories(current_tick)
    day_planner_prompt = policy._construct_day_planner_prompt(planner_memories)

    # Optionally emulate planner output to populate the decision prompt.
    if inject_day_plan_text is not None:
        policy.day_plan = {
            "day_index": int(current_tick // 288),
            "raw_line": str(inject_day_plan_text).strip(),
            "segments": [],
        }
        policy.day_plan_status = {
            "state": "planned",
            "day_index": int(current_tick // 288),
            "attempted": True,
            "plan_present": True,
            "parse_ok": True,
            "plan_chars": len(str(inject_day_plan_text).strip()),
            "error": None,
            "attempt_tick": int(current_tick),
        }

    # Match runtime decision call order: snapshot
    # happens before memory selection.
    if seed_decision_snapshot:
        policy._record_state_snapshot()
    decision_memories, _ = policy._select_memories_with_scores(current_tick)
    decision_prompt = policy._construct_prompt(decision_memories)

    return {
        "shared_system_prompt": policy._render_shared_system_prompt(),
        "day_planner_prompt": day_planner_prompt,
        "decision_prompt": decision_prompt,
    }


def _format_output(
    *,
    agent_row: Dict[str, Any],
    agent_attrs: Dict[str, Any],
    params: Dict[str, Any],
    input_data_folder: Path,
    tick: int,
    prompts: Dict[str, str],
    day_plan_injected: Optional[str],
    seed_decision_snapshot: bool,
) -> str:
    lines: List[str] = []
    lines.append("# Memory Stream Prompt Preview")
    lines.append("")
    lines.append("## Context")
    lines.append(f"- agent_id: {int(agent_row['agent_id'])}")
    lines.append(f"- age: {int(agent_row.get('age', 0))}")
    lines.append(f"- gender: {int(agent_row.get('gender', 0))}")
    lines.append(f"- agent_type: {int(agent_row.get('agent_type', 0))}")
    lines.append(
        f"- home/work/school: {agent_row.get('home')}/{agent_row.get('work')}/"
        f"{agent_row.get('school')}"
    )
    lines.append(f"- tick: {int(tick)}")
    lines.append(f"- survey_start_date: {params.get('survey_start_date')}")
    lines.append(f"- input_data_folder: {input_data_folder}")
    lines.append(
        f"- use_day_planner (config): "
        f"{bool(params.get('use_day_planner', False))}"
    )
    lines.append(f"- decision snapshot seeded: {bool(seed_decision_snapshot)}")
    lines.append(
        f"- day plan injected: {'yes' if day_plan_injected else 'no'}"
    )
    if agent_attrs:
        cbsa = agent_attrs.get("nhts_hh_cbsa_code")
        subtype = agent_attrs.get("subtype_hint")
        lines.append(
            f"- agent attrs sidecar: cbsa={cbsa}, subtype_hint={subtype}"
        )
    lines.append("")

    lines.append("## Shared System Prompt")
    lines.append("```text")
    lines.append(prompts["shared_system_prompt"])
    lines.append("```")
    lines.append("")

    lines.append("## Day Planner Prompt")
    lines.append("```text")
    lines.append(prompts["day_planner_prompt"])
    lines.append("```")
    lines.append("")

    if params.get("use_day_planner", False) and not day_plan_injected:
        lines.append(
            "> Note: The decision prompt below has no planner output injected."
            " At runtime, the planner LLM typically fills `Rough Plan` before "
            "the decision prompt is built."
        )
        lines.append("")

    lines.append("## Decision Prompt")
    lines.append("```text")
    lines.append(prompts["decision_prompt"])
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    repo_root = REPO_ROOT

    config_path = (
        Path(args.config)
        if args.config is not None
        else _choose_default_config(repo_root)
    )
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    params = _load_params(config_path)
    params.setdefault("llm_endpoint", "http://localhost:11434/v1")
    params.setdefault("llm_model", "prompt-preview")
    params.setdefault("survey_start_date", "2025/09/08")
    params.setdefault("use_day_planner", False)
    # Force-disable logging for prompt previews (no side-effect files needed).
    params["memory_stream_log_interactions"] = False
    params["memory_stream_log_memories"] = False
    params["memory_diag_enabled"] = False

    input_data_folder = _resolve_input_data_folder(
        repo_root, params, args.input_data_folder
    )
    agents_df = _load_agents_df(input_data_folder)
    agent_row = _get_agent_row(
        agents_df,
        agent_index=args.agent_index,
        agent_id=args.agent_id,
    )
    agent_attrs = _load_agent_attrs_for_row(input_data_folder, agent_row)

    model = _FakeModel(params=params, tick=args.tick)
    agent = _FakeAgent(
        model=model,
        agent_row=agent_row,
        agent_attrs=agent_attrs,
        current_activity=args.current_activity,
        activity_start_tick=args.activity_start_tick,
    )
    policy = MemoryStreamPolicy(
        agent,
        llm_endpoint=str(
            params.get("llm_endpoint", "http://localhost:11434/v1")
        ),
        llm_model=str(params.get("llm_model", "prompt-preview")),
    )

    prompts = _render_prompt_bundle(
        policy,
        inject_day_plan_text=args.day_plan_text,
        seed_decision_snapshot=bool(args.seed_decision_snapshot),
    )
    rendered = _format_output(
        agent_row=agent_row,
        agent_attrs=agent_attrs,
        params=params,
        input_data_folder=input_data_folder,
        tick=args.tick,
        prompts=prompts,
        day_plan_injected=args.day_plan_text,
        seed_decision_snapshot=bool(args.seed_decision_snapshot),
    )

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = (repo_root / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered)
        print(f"Wrote prompt preview to: {out_path}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()

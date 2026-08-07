from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, List, Sequence, Tuple

from .config import GeneratorConfig
from .nhts_loader import SampledPersonDay
from .runtime_contract import (
    build_gold_day_outline,
    build_reflection_system_prompt,
    build_shared_system_prompt,
    build_today_activity_table,
    make_decision_memory,
    make_observation_memory,
    make_reflection_memory,
    minute_to_datetime_str,
    minute_to_time_str,
    render_day_planner_prompt,
    render_decision_prompt,
    render_reflection_prompt,
    select_memories,
)


@dataclass
class ReplayState:
    sample: SampledPersonDay
    current_activity_type: int
    current_activity_start_minute: int
    completed_segments: List[Tuple[int, int, int]] = field(default_factory=list)
    memories: List[Dict[str, Any]] = field(default_factory=list)
    day_plan_text: str = ""


def _split_name(config: GeneratorConfig, person_key: str) -> str:
    digest = hashlib.sha256(person_key.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    if bucket < config.train_ratio:
        return "train"
    if bucket < config.train_ratio + config.val_ratio:
        return "val"
    return "test"


def _build_profile(sample: SampledPersonDay) -> Dict[str, Any]:
    return {
        "age": sample.age,
        "gender": sample.gender,
        "role": sample.agent_role,
        "city_name": sample.prompt_location_name,
        "attrs": sample.attrs,
    }


def _row_base(
    *,
    sample: SampledPersonDay,
    config: GeneratorConfig,
    task_type: str,
    step_index: int,
    day_plan_text: str,
) -> Dict[str, Any]:
    row = {
        "task_type": task_type,
        "split": _split_name(config, sample.person_key),
        "sample_index": sample.sample_index,
        "sample_day_id": sample.sample_day_id,
        "house_id": sample.house_id,
        "person_id": sample.person_id,
        "person_key": sample.person_key,
        "cbsa_code": sample.cbsa_code,
        "cbsa_title": sample.cbsa_title,
        "prompt_location_name": sample.prompt_location_name,
        "agent_role": sample.agent_role,
        "age": sample.age,
        "gender": sample.gender,
        "step_index": step_index,
        "day_plan_text": day_plan_text,
    }
    return row


def _build_messages(
    *,
    system_prompt: str,
    user_prompt: str,
    thinking: str,
    content: str,
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {
            "role": "assistant",
            "thinking": thinking,
            "content": content,
        },
    ]


async def replay_sample_day(
    *,
    sample: SampledPersonDay,
    config: GeneratorConfig,
    teacher_client,
) -> List[Dict[str, Any]]:
    profile = _build_profile(sample)
    shared_system_prompt = build_shared_system_prompt(profile)
    reflection_system_prompt = build_reflection_system_prompt(profile)

    state = ReplayState(
        sample=sample,
        current_activity_type=int(sample.dwell_blocks[0]["activity_type"]),
        current_activity_start_minute=int(sample.dwell_blocks[0]["start_minute"]),
    )
    rows: List[Dict[str, Any]] = []

    planner_user_prompt = render_day_planner_prompt(
        starting_activity_type=state.current_activity_type,
        day_of_week=sample.day_of_week,
        memories=[],
    )
    planner_output = await teacher_client.generate_day_planner(
        runtime_system_prompt=shared_system_prompt,
        runtime_user_prompt=planner_user_prompt,
        gold_day_outline=build_gold_day_outline(
            sample.date_str,
            sample.dwell_blocks,
        ),
    )
    state.day_plan_text = planner_output.content
    planner_row = _row_base(
        sample=sample,
        config=config,
        task_type="day_planner",
        step_index=0,
        day_plan_text=state.day_plan_text,
    )
    planner_row["messages"] = _build_messages(
        system_prompt=shared_system_prompt,
        user_prompt=planner_user_prompt,
        thinking=planner_output.thinking,
        content=planner_output.content,
    )
    rows.append(planner_row)

    for block_index, block in enumerate(sample.dwell_blocks):
        decision_time = int(block["start_minute"])
        gold_next_activity_type = int(block["activity_type"])
        gold_stay_minutes = int(block["end_minute"]) - int(block["start_minute"])

        observation_memory = make_observation_memory(
            minute_of_day=decision_time,
            day_str=sample.date_str,
            day_of_week=sample.day_of_week,
            activity_type=state.current_activity_type,
            minutes_here=decision_time - state.current_activity_start_minute,
        )
        state.memories.append(observation_memory)
        selected_memories = select_memories(
            memories=state.memories,
            current_minute=decision_time,
            current_activity_type=state.current_activity_type,
            day_of_week=sample.day_of_week,
            attrs=sample.attrs,
            max_context_memories=config.max_context_memories,
            recency_weight=config.recency_weight,
            importance_weight=config.importance_weight,
            relevance_weight=config.relevance_weight,
        )
        today_activity_table = build_today_activity_table(
            completed_segments=state.completed_segments,
            current_activity_type=state.current_activity_type,
            current_activity_start_minute=state.current_activity_start_minute,
            now_minute=decision_time,
        )
        decision_user_prompt = render_decision_prompt(
            current_activity_type=state.current_activity_type,
            day_of_week=sample.day_of_week,
            day_str=minute_to_datetime_str(sample.date_str, decision_time),
            day_plan_text=state.day_plan_text,
            today_activity_table=today_activity_table,
            memories=selected_memories,
        )
        decision_output = await teacher_client.generate_decision(
            runtime_system_prompt=shared_system_prompt,
            runtime_user_prompt=decision_user_prompt,
            gold_next_activity_type=gold_next_activity_type,
            gold_stay_minutes=gold_stay_minutes,
        )
        decision_row = _row_base(
            sample=sample,
            config=config,
            task_type="decision",
            step_index=block_index + 1,
            day_plan_text=state.day_plan_text,
        )
        decision_row["gold_next_activity_type"] = gold_next_activity_type
        decision_row["gold_stay_minutes"] = gold_stay_minutes
        decision_row["messages"] = _build_messages(
            system_prompt=shared_system_prompt,
            user_prompt=decision_user_prompt,
            thinking=decision_output.thinking,
            content=decision_output.content,
        )
        rows.append(decision_row)

        decision_payload = json.loads(decision_output.content)
        state.memories.append(
            make_decision_memory(
                minute_of_day=decision_time,
                day_str=sample.date_str,
                day_of_week=sample.day_of_week,
                activity_type=gold_next_activity_type,
                stay_minutes=gold_stay_minutes,
                rationale=str(decision_payload["rationale"]),
            )
        )
        if gold_next_activity_type != state.current_activity_type:
            if decision_time > state.current_activity_start_minute:
                state.completed_segments.append(
                    (
                        state.current_activity_start_minute,
                        decision_time,
                        state.current_activity_type,
                    )
                )
            state.current_activity_type = gold_next_activity_type
            state.current_activity_start_minute = decision_time

    reflection_input_memories = [
        memory for memory in state.memories if memory.get("type") != "reflection"
    ]
    reflection_user_prompt = render_reflection_prompt(
        day_of_week=sample.day_of_week,
        day_str=sample.date_str,
        time_str=minute_to_time_str(1439),
        memories=reflection_input_memories,
    )
    reflection_output = await teacher_client.generate_reflection(
        runtime_system_prompt=reflection_system_prompt,
        runtime_user_prompt=reflection_user_prompt,
    )
    state.memories.append(
        make_reflection_memory(
            minute_of_day=1439,
            day_str=sample.date_str,
            day_of_week=sample.day_of_week,
            summary=reflection_output.content,
        )
    )
    reflection_row = _row_base(
        sample=sample,
        config=config,
        task_type="reflection",
        step_index=len(sample.dwell_blocks) + 1,
        day_plan_text=state.day_plan_text,
    )
    reflection_row["messages"] = _build_messages(
        system_prompt=reflection_system_prompt,
        user_prompt=reflection_user_prompt,
        thinking=reflection_output.thinking,
        content=reflection_output.content,
    )
    rows.append(reflection_row)
    return rows

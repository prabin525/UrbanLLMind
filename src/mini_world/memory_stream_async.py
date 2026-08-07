import json
import os
from datetime import timedelta
from importlib import resources
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import asyncio

import openai
import yaml
# from termcolor import cprint

from mini_world.activity_taxonomy import (
    LLM_DEFAULT_STAY_MINUTES,
    VALID_ACTIVITY_TYPES,
    activity_internal_label,
    activity_prompt_label,
    format_activity_codes_block,
    format_activity_notes_block,
    prompt_activity_vocab_text,
)
from mini_world.agent_types import (
    HOMEMAKER,
    STUDENT,
    WORKER,
    role_label_for_agent_type,
    validated_agent_type,
)
from mini_world.day_planner import (
    activity_vocab_text,
    get_plan_line,
    # get_step_plan_reminder,
    summarize_plan_for_diagnostics,
)
from pathlib import Path
# import time

REASONING_EFFORT = 'medium'


def _load_yaml_from_path(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt YAML must be a mapping: {path}")
    return dict(payload)


def _load_yaml_from_package(resource_name: str) -> Dict[str, Any]:
    text = resources.files("mini_world.prompts").joinpath(
        resource_name
    ).read_text(encoding="utf-8")
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Prompt YAML must be a mapping: {resource_name}")
    return dict(payload)


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class MemoryStreamPolicy:
    """
    LLM policy that maintains a memory stream similar to Generative Agents.
    """

    # Class-level cache for prompt YAML (loaded once per process,
    # not per agent)
    _prompts_cache: Optional[Dict[str, Any]] = None

    def __init__(self, agent, llm_endpoint: str, llm_model: str) -> None:
        self.agent = agent
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.llm_client = None
        params = self.agent.model.params if self.agent.model else {}

        prompt_override = str(
            params.get("memory_stream_prompt_path", "")
        ).strip()
        if prompt_override:
            self.prompts = _load_yaml_from_path(prompt_override)
        else:
            if MemoryStreamPolicy._prompts_cache is None:
                MemoryStreamPolicy._prompts_cache = _load_yaml_from_package(
                    "memory_stream.yaml"
                )
            self.prompts = dict(MemoryStreamPolicy._prompts_cache)

        self.shared_system_prompt_template = str(
            self.prompts.get("shared_system_prompt")
            or self.prompts.get("persona_template")
            or ""
        ).strip()
        if not self.shared_system_prompt_template:
            self.shared_system_prompt_template = (
                "You choose realistic next actions for one simulated person. "
                "Return valid JSON when asked."
            )
        self.shared_system_prompt = self._render_shared_system_prompt()

        self.max_context_memories = params.get(
            "memory_stream_max_context_memories", 6
        )
        self.recency_weight = params.get("memory_stream_recency_weight", 1.5)
        self.importance_weight = params.get(
            "memory_stream_importance_weight", 1.0
        )
        self.relevance_weight = params.get(
            "memory_stream_relevance_weight", 1.2
        )
        self.store_limit = params.get("memory_stream_store_limit", 200)
        self.reflection_interval_ticks = params.get(
            "memory_stream_reflection_interval_ticks", 288
        )
        self.reflection_memory_count = params.get(
            "memory_stream_reflection_memory_count", 10
        )
        self.enable_interval_reflections = self._as_bool(
            params.get("memory_stream_enable_interval_reflections", False)
        )
        self.log_memories = self._as_bool(
            params.get(
                "memory_stream_log_memories",
                params.get("memory_stream_debug_log", False)
            )
        )
        self.log_interactions = self._as_bool(
            params.get("memory_stream_log_interactions", True)
        )
        self.include_behavior_priors = self._as_bool(
            params.get("memory_stream_include_behavior_priors", False)
        )
        self.day_planner_temperature = self._non_negative_float(
            params.get("memory_stream_day_planner_temperature", 0.4),
            default=0.4,
        )
        self.day_planner_top_p = self._top_p_or_none(
            params.get("memory_stream_day_planner_top_p")
        )
        self.decision_temperature = self._non_negative_float(
            params.get("memory_stream_decision_temperature", 0.7),
            default=0.7,
        )
        self.decision_top_p = self._top_p_or_none(
            params.get("memory_stream_decision_top_p")
        )
        self.reflection_temperature = self._non_negative_float(
            params.get("memory_stream_reflection_temperature", 0.5),
            default=0.5,
        )
        self.reflection_top_p = self._top_p_or_none(
            params.get("memory_stream_reflection_top_p")
        )
        self.memory_ablation_mode = self._normalize_ablation_mode(
            params.get("memory_ablation_mode", "baseline")
        )
        self.diag_enabled = self._as_bool(
            params.get("memory_diag_enabled", False)
        )
        self.diag_top_k = self._non_negative_int(
            params.get("memory_diag_top_k", 0),
            default=0
        )
        self.diag_sample_agents = self._parse_agent_id_filter(
            params.get("memory_diag_sample_agents", "all")
        )
        self.use_day_planner = self._as_bool(
            params.get("use_day_planner", False)
        )
        self.enable_prompt_event_injection = self._as_bool(
            params.get("memory_stream_enable_prompt_event_injection", False)
        )
        self.event_days_1based: List[int] = []
        self.day_planner_event_map: Dict[int, str] = {}
        self.decision_event_map: Dict[int, str] = {}
        self.reflection_event_map: Dict[int, str] = {}
        self._load_prompt_event_injection_config(params)

        self.memories: List[Dict[str, Any]] = []
        self.last_reflection_tick: int = (
            self.agent.model.runner.schedule.tick
            if self.agent and self.agent.model
            else 0
        )
        self.last_reflection_index: int = 0
        self.day_plan: Optional[Dict[str, Any]] = None
        initial_day_index = self._current_day_index()
        self.day_plan_status: Dict[str, Any] = self._empty_day_plan_status(
            initial_day_index
        )
        saved_plan = getattr(self.agent, "day_plan_state", None)
        if (
            self.use_day_planner
            and isinstance(saved_plan, dict)
            and saved_plan
        ):
            self.day_plan = dict(saved_plan)
        saved_plan_status = getattr(self.agent, "day_plan_status", None)
        if (
            self.use_day_planner
            and isinstance(saved_plan_status, dict)
            and saved_plan_status
        ):
            self.day_plan_status = dict(saved_plan_status)

        self.memory_log_path: Optional[str] = None
        self.interaction_log_path: Optional[str] = None
        self.diagnostics_log_path: Optional[str] = None
        self._logged_system_prompts: Set[str] = set()
        if self.agent and self.agent.model:
            folder = getattr(self.agent.model, "out_folder_name", ".")
            os.makedirs(folder, exist_ok=True)
            rank = int(self.agent.model.rank)
            if self.log_memories:
                self.memory_log_path = os.path.join(
                    folder, f"memory_stream_memories_rank{rank}.jsonl"
                )
            if self.log_interactions:
                self.interaction_log_path = os.path.join(
                    folder, f"memory_stream_interactions_rank{rank}.jsonl"
                )
            if self.diag_enabled:
                self.diagnostics_log_path = os.path.join(
                    folder, f"memory_stream_diagnostics_rank{rank}.jsonl"
                )
        self.pending_reflection_force = False
        self.pending_reflection_tick: Optional[int] = None

    def _agent_id(self) -> Any:
        agent_id = getattr(self.agent, "id", None)
        if agent_id is not None:
            try:
                return int(agent_id)
            except Exception:
                return agent_id
        uid = getattr(self.agent, "uid", None)
        if isinstance(uid, tuple) and uid:
            return uid[0]
        return uid

    def _agent_role_label(self) -> str:
        agent_type = validated_agent_type(
            getattr(self.agent, "agent_type", None),
            context=(
                f"MemoryStreamPolicy(agent_id={self._agent_id()}) agent_type"
            ),
        )
        return role_label_for_agent_type(
            agent_type,
            context=f"MemoryStreamPolicy(agent_id={self._agent_id()}) role",
        )

    def _agent_gender_label(self) -> str:
        gender_raw = getattr(self.agent, "gender", None)
        try:
            gender_value = int(gender_raw)
        except Exception:
            return str(gender_raw).strip().lower() or "unknown"
        if gender_value == 0:
            return "male"
        if gender_value == 1:
            return "female"
        return "unknown"

    def _resolve_city_name(self) -> str:
        if self.agent is not None and self.agent.model is not None:
            configured = str(
                self.agent.model.params.get("city_name", "")
            ).strip()
            if configured:
                return configured
        attrs = getattr(self.agent, "agent_attrs", {}) or {}
        attrs_city = str(attrs.get("city_name", "")).strip()
        if attrs_city and attrs_city.lower() != "unknown":
            return attrs_city
        cbsa_code = str(attrs.get("nhts_hh_cbsa_code", "")).strip()
        if cbsa_code == "41860":
            return "San Francisco"
        if cbsa_code == "31080":
            return "Los Angeles"
        return "San Francisco"  # Default to SF

    def _extra_profile_attributes(self) -> str:
        attrs = getattr(self.agent, "agent_attrs", {}) or {}
        fields = [
            ("employment_status", "employment status"),
            ("work_schedule_type", "work schedule"),
            ("school_type", "school type"),
            ("subtype_hint", "subtype"),
            ("household_vehicle_count", "household vehicles"),
            ("household_income_band", "household income"),
            ("worker_type", "worker type"),
            ("household_size", "household size"),
            ("work_building_tag", "work building tag"),
            ("occupation", "occupation"),
            ("household_adults", "household adults"),
            ("household_children", "household children"),
            ("household_elder", "household elder"),
        ]
        details: List[str] = []
        for key, label in fields:
            value = attrs.get(key)
            if self._is_unknown(value):
                continue
            details.append(f"{label}: {value}")
        return "; ".join(details) if details else "none provided"

    def _agent_prompt_values(self) -> Dict[str, Any]:
        try:
            age_value = int(getattr(self.agent, "age", 0))
        except Exception:
            age_value = 0
        return {
            "age": age_value,
            "gender": self._agent_gender_label(),
            "role": self._agent_role_label(),
            "city_name": self._resolve_city_name(),
            "extra_profile_attributes": self._extra_profile_attributes(),
            "activity_vocab_prompt_labels": prompt_activity_vocab_text(),
        }

    def _render_shared_system_prompt(self) -> str:
        template = str(
            getattr(self, "shared_system_prompt_template", "")
        ).strip()
        if not template:
            return (
                "You choose realistic next actions for one simulated person. "
                "Return valid JSON when asked."
            )
        values = self._agent_prompt_values()
        try:
            return template.format_map(_SafeFormatDict(values))
        except Exception:
            return template

    def _render_reflection_system_prompt(self) -> str:
        template = str(
            self.prompts.get("reflection_system_prompt")
            or ""
        ).strip()
        if not template:
            return self._render_shared_system_prompt()
        values = self._agent_prompt_values()
        try:
            return template.format_map(_SafeFormatDict(values))
        except Exception:
            return template

    def _normalize_string_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_values: Sequence[Any] = value.split(",")
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raw_values = [value]
        return [str(item).strip() for item in raw_values]

    def _normalize_positive_int_list(self, value: Any) -> List[int]:
        if value is None:
            return []
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned == "":
                return []
            raw_values: Sequence[Any] = cleaned.split(",")
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raw_values = [value]

        parsed: List[int] = []
        for item in raw_values:
            text = str(item).strip()
            if text == "":
                continue
            try:
                day_value = int(text)
            except Exception as exc:
                raise ValueError(
                    "Invalid memory_stream_event_days_1based value: "
                    f"{item!r}"
                ) from exc
            if day_value <= 0:
                raise ValueError(
                    "memory_stream_event_days_1based must contain only "
                    f"positive integers, got {day_value}"
                )
            parsed.append(day_value)
        return parsed

    def _load_prompt_event_injection_config(
        self,
        params: Dict[str, Any],
    ) -> None:
        days = self._normalize_positive_int_list(
            params.get("memory_stream_event_days_1based", [])
        )
        planner_texts = self._normalize_string_list(
            params.get("memory_stream_day_planner_event_texts", [])
        )
        decision_texts = self._normalize_string_list(
            params.get("memory_stream_decision_event_texts", [])
        )
        reflection_texts = self._normalize_string_list(
            params.get("memory_stream_reflection_event_texts", [])
        )

        any_values = bool(
            days
            or planner_texts
            or decision_texts
            or reflection_texts
            or self.enable_prompt_event_injection
        )
        if not any_values:
            return

        if len(set(days)) != len(days):
            raise ValueError(
                "memory_stream_event_days_1based contains duplicate day values"
            )

        expected = len(days)
        if expected == 0:
            if self.enable_prompt_event_injection:
                raise ValueError(
                    "memory_stream_enable_prompt_event_injection=True "
                    "requires "
                    "at least one day in memory_stream_event_days_1based"
                )
            return

        list_specs = (
            (
                "memory_stream_day_planner_event_texts",
                planner_texts,
            ),
            (
                "memory_stream_decision_event_texts",
                decision_texts,
            ),
            (
                "memory_stream_reflection_event_texts",
                reflection_texts,
            ),
        )
        for field_name, values in list_specs:
            if len(values) != expected:
                raise ValueError(
                    f"{field_name} length {len(values)} must match "
                    "memory_stream_event_days_1based length "
                    f"{expected}"
                )
            for day_value, text_value in zip(days, values):
                if text_value == "":
                    raise ValueError(
                        f"{field_name} has empty text for day {day_value}"
                    )

        self.event_days_1based = list(days)
        self.day_planner_event_map = {
            day_value: text
            for day_value, text in zip(days, planner_texts)
        }
        self.decision_event_map = {
            day_value: text
            for day_value, text in zip(days, decision_texts)
        }
        self.reflection_event_map = {
            day_value: text
            for day_value, text in zip(days, reflection_texts)
        }

    def _day_1based_from_tick(self, tick: int) -> int:
        return int(tick // 288) + 1

    def _event_context_for_prompt(
        self,
        prompt_name: str,
        tick: int,
    ) -> str:
        default_text = ""
        if not self.enable_prompt_event_injection:
            return default_text

        day_1based = self._day_1based_from_tick(int(tick))
        event_map_by_prompt = {
            "day_planner": self.day_planner_event_map,
            "decision": self.decision_event_map,
            "reflection": self.reflection_event_map,
        }
        selected_map = event_map_by_prompt.get(prompt_name, {})
        text = str(selected_map.get(day_1based, "")).strip()
        if not text:
            return default_text
        return (
            "Special Event Notice:\n"
            f"- {text}"
        )

    @staticmethod
    def _strip_surrounding_code_fence(text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned.startswith("```"):
            return cleaned
        lines = cleaned.splitlines()
        if not lines:
            return ""
        if lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _append_jsonl(
        self, path: Optional[str], payload: Dict[str, Any]
    ) -> None:
        if not path:
            return
        try:
            with open(path, "a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass

    def _current_tick(self) -> int:
        if self.agent is None or self.agent.model is None:
            return 0
        return int(self.agent.model.runner.schedule.tick)

    def _current_day_index(self) -> int:
        return int(self._current_tick() // 288)

    def _empty_day_plan_status(self, day_index: int) -> Dict[str, Any]:
        if not self.use_day_planner:
            return {
                "state": "planner_disabled",
                "day_index": int(day_index),
                "attempted": False,
                "plan_present": False,
                "parse_ok": False,
                "plan_chars": 0,
                "error": None,
                "attempt_tick": None,
            }
        return {
            "state": "not_attempted",
            "day_index": int(day_index),
            "attempted": False,
            "plan_present": False,
            "parse_ok": False,
            "plan_chars": 0,
            "error": None,
            "attempt_tick": None,
        }

    def _reset_day_plan_for_day(self, day_index: int) -> None:
        self.day_plan = None
        self.day_plan_status = self._empty_day_plan_status(day_index)
        self._sync_day_plan_to_agent()

    def _ensure_day_plan_day_alignment(self, day_index: int) -> None:
        status_day = self.day_plan_status.get("day_index")
        try:
            status_day_index = int(status_day)
        except Exception:
            status_day_index = day_index
        if status_day_index != int(day_index):
            self._reset_day_plan_for_day(day_index)

    def _sync_day_plan_to_agent(self) -> None:
        if self.agent is None:
            return
        if hasattr(self.agent, "day_plan_state"):
            self.agent.day_plan_state = dict(self.day_plan or {})
        if hasattr(self.agent, "day_plan_status"):
            self.agent.day_plan_status = dict(self.day_plan_status or {})

    async def ensure_day_plan_async(self, llm_client=None) -> None:
        if llm_client is not None:
            self.llm_client = llm_client
        if not self.use_day_planner:
            self.day_plan = None
            self.day_plan_status = self._empty_day_plan_status(
                self._current_day_index()
            )
            self._sync_day_plan_to_agent()
            return

        current_tick = self._current_tick()
        day_index = int(current_tick // 288)
        self._ensure_day_plan_day_alignment(day_index)
        if (
            isinstance(self.day_plan, dict)
            and int(self.day_plan.get("day_index", day_index)) == day_index
            and not bool(self.day_plan_status.get("attempted", False))
        ):
            self.day_plan_status.update(
                {
                    "state": "planned",
                    "attempted": True,
                    "plan_present": True,
                    "parse_ok": True,
                    "plan_chars": len(get_plan_line(self.day_plan)),
                    "error": None,
                }
            )
        if bool(self.day_plan_status.get("attempted", False)):
            return
        self.day_plan_status["attempted"] = True
        self.day_plan_status["attempt_tick"] = int(current_tick)
        self.day_plan_status["day_index"] = int(day_index)

        if self.llm_client is None:
            self.llm_client = openai.AsyncOpenAI(
                base_url=self.llm_endpoint,
                api_key="sk-no-key-required"
            )

        focus_tags = self._current_focus_tags()
        selected_memories: List[Dict[str, Any]] = []
        if self.memory_ablation_mode != "no_memory":
            selected_memories = self._select_planner_reflection_memories(
                current_tick,
                focus_tags=focus_tags,
            )
        planner_prompt = self._construct_day_planner_prompt(selected_memories)
        system_prompt = self._render_shared_system_prompt()
        planner_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": planner_prompt},
        ]
        self._log_system_prompt_once(
            tick=current_tick,
            prompt_name="day_planner",
            system_prompt=system_prompt,
        )

        llm_usage: Optional[Dict[str, Optional[int]]] = None
        parse_error = None
        planner_reasoning_text: Optional[str] = None
        status = "failed"
        assistant_content = ""
        parsed_payload: Optional[Dict[str, Any]] = None
        try:
            request_kwargs: Dict[str, Any] = {
                "model": self.llm_model,
                "messages": planner_messages,
                "temperature": self.day_planner_temperature,
                "timeout": None,
                "reasoning_effort": REASONING_EFFORT,
            }
            if self.day_planner_top_p is not None:
                request_kwargs["top_p"] = self.day_planner_top_p
            response = await self.llm_client.chat.completions.create(
                **request_kwargs
            )
            choice = response.choices[0].message
            assistant_content = (choice.content or "").strip()
            planner_reasoning_text = self._extract_reasoning_text(choice)
            llm_usage = self._extract_usage(response)
            plan_text = self._strip_surrounding_code_fence(
                assistant_content
            )
            # NEEDS_REMOVAL_MARKER: parser has been removed for free-form
            # planner text ingestion.
            # parsed_plan, parse_error = parse_day_plan_line(first_line)
            # if parsed_plan is not None:
            #     self.day_plan = {
            #         "day_index": int(day_index),
            #         "raw_line": parsed_plan["raw_line"],
            #         "segments": parsed_plan["segments"],
            #     }
            #     self.day_plan_status.update(
            #         {
            #             "state": "planned",
            #             "plan_present": True,
            #             "parse_ok": True,
            #             "plan_chars": len(parsed_plan["raw_line"]),
            #             "error": None,
            #         }
            #     )
            #     status = "ok"
            #     parsed_payload = {
            #         "day_plan_line": parsed_plan["raw_line"],
            #         "segment_count": len(parsed_plan["segments"]),
            #     }
            # else:
            #     self.day_plan = None
            #     self.day_plan_status.update(
            #         {
            #             "state": "plan_failed",
            #             "plan_present": False,
            #             "parse_ok": False,
            #             "plan_chars": len(first_line),
            #             "error": parse_error or "invalid_day_plan_line",
            #         }
            #     )
            if plan_text:
                self.day_plan = {
                    "day_index": int(day_index),
                    "raw_line": plan_text,
                    "segments": [],
                }
                self.day_plan_status.update(
                    {
                        "state": "planned",
                        "plan_present": True,
                        "parse_ok": True,
                        "plan_chars": len(plan_text),
                        "error": None,
                    }
                )
                status = "ok"
                parsed_payload = {
                    "day_plan_line": plan_text,
                    "segment_count": 0,
                }
            else:
                parse_error = "empty_plan_text"
                self.day_plan = None
                self.day_plan_status.update(
                    {
                        "state": "plan_failed",
                        "plan_present": False,
                        "parse_ok": False,
                        "plan_chars": 0,
                        "error": parse_error,
                    }
                )
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            self.day_plan = None
            self.day_plan_status.update(
                {
                    "state": "plan_failed",
                    "plan_present": False,
                    "parse_ok": False,
                    "plan_chars": 0,
                    "error": parse_error,
                }
            )

        self._log_interaction(
            tick=current_tick,
            prompt_name="day_planner",
            user_prompt=planner_prompt,
            assistant_response=assistant_content,
            status=status,
            parsed=parsed_payload,
            error=parse_error if status != "ok" else None,
            llm_usage=llm_usage,
            reasoning_text=planner_reasoning_text,
        )
        self._log_planner_diagnostics(
            tick=current_tick,
            status=status,
            llm_usage=llm_usage,
            error=parse_error if status != "ok" else None,
        )
        self._sync_day_plan_to_agent()

    def _normalize_ablation_mode(self, value: Any) -> str:
        mode = str(value).strip().lower()
        if mode in {"baseline", "no_memory", "no_reflection", "custom"}:
            return mode
        return "baseline"

    def _non_negative_int(self, value: Any, default: int = 0) -> int:
        try:
            parsed = int(value)
        except Exception:
            return default
        return parsed if parsed >= 0 else default

    def _non_negative_float(self, value: Any, default: float) -> float:
        try:
            parsed = float(value)
        except Exception:
            return float(default)
        return parsed if parsed >= 0.0 else float(default)

    def _top_p_or_none(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return None
        try:
            parsed = float(text)
        except Exception:
            return None
        if parsed <= 0.0 or parsed > 1.0:
            return None
        return parsed

    def _parse_agent_id_filter(self, value: Any) -> Optional[Set[int]]:
        if value is None:
            return None

        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned == "" or cleaned.lower() in {"all", "*"}:
                return None
            raw_values: Sequence[Any] = [
                part.strip() for part in cleaned.split(",")
            ]
        elif isinstance(value, (list, tuple, set)):
            raw_values = list(value)
        else:
            raw_values = [value]

        parsed: Set[int] = set()
        for item in raw_values:
            try:
                parsed.add(int(item))
            except Exception:
                continue
        return parsed if parsed else None

    def _should_log_diagnostics_for_agent(self) -> bool:
        if not self.diag_enabled or not self.diagnostics_log_path:
            return False
        if self.diag_sample_agents is None:
            return True
        agent_id = self._agent_id()
        try:
            return int(agent_id) in self.diag_sample_agents
        except Exception:
            return False

    def _needs_snapshot(self) -> Dict[str, float]:
        return {
            "work": float(self.agent.work_need[0]),
            "food": float(self.agent.food_need[0]),
            "social": float(self.agent.social_need[0]),
            "errands": float(self.agent.errands_need[0]),
            "rest": float(self.agent.rest_need[0]),
        }

    def _normalize_tags(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, set):
            return sorted(str(v).lower() for v in value)
        if isinstance(value, (list, tuple)):
            return sorted(str(v).lower() for v in value)
        return [str(value).lower()]

    def _serialize_memory_for_diagnostics(
        self,
        memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "tick": int(memory.get("tick", -1)),
            "time_label": str(memory.get("time_label", "")),
            "type": str(memory.get("type", "unknown")),
            "importance": float(memory.get("importance", 0.0)),
            "tags": self._normalize_tags(memory.get("tags")),
            "text": str(memory.get("text", "")),
        }
        if "activity_type" in memory:
            try:
                payload["activity_type"] = int(memory.get("activity_type"))
            except Exception:
                payload["activity_type"] = memory.get("activity_type")
        if "stay_minutes" in memory:
            stay_value = memory.get("stay_minutes")
            if stay_value is None:
                payload["stay_minutes"] = None
            else:
                try:
                    payload["stay_minutes"] = int(stay_value)
                except Exception:
                    payload["stay_minutes"] = stay_value
        if "rationale" in memory:
            rationale_value = memory.get("rationale")
            if rationale_value is None:
                payload["rationale"] = None
            else:
                payload["rationale"] = str(rationale_value)
        return payload

    def _serialize_memory_ref_for_diagnostics(
        self,
        memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "tick": int(memory.get("tick", -1)),
            "time_label": str(memory.get("time_label", "")),
            "type": str(memory.get("type", "unknown")),
        }

    def _log_decision_diagnostics(
        self,
        *,
        tick: int,
        status: str,
        focus_tags: Set[str],
        selected_memories: List[Dict[str, Any]],
        memory_candidates: List[Dict[str, Any]],
        parsed: Optional[Dict[str, Any]],
        llm_usage: Optional[Dict[str, Optional[int]]],
        reasoning_text_chars: int,
        assistant_response_chars: int,
        error: Optional[str],
        reflection_executed: bool
    ) -> None:
        if not self._should_log_diagnostics_for_agent():
            return

        _, day_of_week = self._get_human_readable_datetime(tick)
        activity_type = int(self.agent.activity_type)
        candidates_to_log = memory_candidates
        if self.diag_top_k > 0:
            candidates_to_log = memory_candidates[: self.diag_top_k]

        payload: Dict[str, Any] = {
            "event": "decision_diagnostics",
            "tick": int(tick),
            "rank": int(self.agent.model.rank),
            "agent_id": self._agent_id(),
            "ablation_mode": self.memory_ablation_mode,
            "day_of_week": day_of_week,
            "current_activity_type": activity_type,
            "current_activity": activity_internal_label(activity_type),
            "needs": self._needs_snapshot(),
            "focus_tags": sorted(str(tag).lower() for tag in focus_tags),
            "memory_retrieval_used": self.memory_ablation_mode != "no_memory",
            "selected_memory_count": len(selected_memories),
            "selected_memories": [
                self._serialize_memory_for_diagnostics(mem)
                for mem in selected_memories
            ],
            "memory_candidates_count": len(memory_candidates),
            "memory_candidates_logged": len(candidates_to_log),
            "memory_candidates": candidates_to_log,
            "status": status,
            "assistant_response_chars": int(assistant_response_chars),
            "reasoning_text_chars": int(reasoning_text_chars),
            "reflection_mode_enabled": (
                self.memory_ablation_mode != "no_reflection"
            ),
            "reflection_executed": bool(reflection_executed),
            "reflection_pending_force": bool(self.pending_reflection_force),
            "planner": summarize_plan_for_diagnostics(
                plan=self.day_plan,
                status=self.day_plan_status,
            ),
            "interaction_ref": {
                "event": "interaction",
                "prompt_name": "decision",
                "tick": int(tick),
            },
        }
        if parsed is not None:
            payload["parsed"] = parsed
        if llm_usage is not None:
            payload["llm_usage"] = llm_usage
        if error:
            payload["error"] = error

        self._append_jsonl(self.diagnostics_log_path, payload)

    def _log_reflection_diagnostics(
        self,
        *,
        tick: int,
        status: str,
        sampled_memories: List[Dict[str, Any]],
        llm_usage: Optional[Dict[str, Optional[int]]],
        reasoning_text_chars: int,
        summary_chars: int,
        force: bool,
        trigger: str,
        error: Optional[str] = None
    ) -> None:
        if not self._should_log_diagnostics_for_agent():
            return

        _, day_of_week = self._get_human_readable_datetime(tick)
        payload: Dict[str, Any] = {
            "event": "reflection_diagnostics",
            "tick": int(tick),
            "rank": int(self.agent.model.rank),
            "agent_id": self._agent_id(),
            "ablation_mode": self.memory_ablation_mode,
            "day_of_week": day_of_week,
            "status": status,
            "force": bool(force),
            "trigger": trigger,
            "sampled_memory_count": len(sampled_memories),
            "sampled_memories": [
                self._serialize_memory_ref_for_diagnostics(mem)
                for mem in sampled_memories
            ],
            "summary_chars": int(summary_chars),
            "reasoning_text_chars": int(reasoning_text_chars),
            "interaction_ref": {
                "event": "interaction",
                "prompt_name": "reflection",
                "tick": int(tick),
            },
        }
        if llm_usage is not None:
            payload["llm_usage"] = llm_usage
        if error:
            payload["error"] = error
        self._append_jsonl(self.diagnostics_log_path, payload)

    def _log_planner_diagnostics(
        self,
        *,
        tick: int,
        status: str,
        llm_usage: Optional[Dict[str, Optional[int]]],
        error: Optional[str],
    ) -> None:
        if not self._should_log_diagnostics_for_agent():
            return

        _, day_of_week = self._get_human_readable_datetime(tick)
        plan_line = get_plan_line(self.day_plan)
        payload: Dict[str, Any] = {
            "event": "planner_diagnostics",
            "tick": int(tick),
            "rank": int(self.agent.model.rank),
            "agent_id": self._agent_id(),
            "ablation_mode": self.memory_ablation_mode,
            "day_of_week": day_of_week,
            "day_index": int(
                self.day_plan_status.get("day_index", tick // 288)
            ),
            "status": status,
            "state": str(self.day_plan_status.get("state", "unknown")),
            "attempted": bool(self.day_plan_status.get("attempted", False)),
            "plan_present": bool(
                self.day_plan_status.get("plan_present", False)
            ),
            "parse_ok": bool(self.day_plan_status.get("parse_ok", False)),
            "plan_chars": int(self.day_plan_status.get("plan_chars", 0)),
            "plan_line": plan_line,
            "interaction_ref": {
                "event": "interaction",
                "prompt_name": "day_planner",
                "tick": int(tick),
            },
        }
        if llm_usage is not None:
            payload["llm_usage"] = llm_usage
        if error:
            payload["error"] = error
        self._append_jsonl(self.diagnostics_log_path, payload)

    def _log_system_prompt_once(
        self,
        *,
        tick: int,
        prompt_name: str,
        system_prompt: str
    ) -> None:
        if not self.interaction_log_path:
            return
        if prompt_name in self._logged_system_prompts:
            return
        self._logged_system_prompts.add(prompt_name)
        payload = {
            "event": "system_prompt",
            "tick": int(tick),
            "rank": int(self.agent.model.rank),
            "agent_id": self._agent_id(),
            "prompt_name": prompt_name,
            "system_prompt": system_prompt,
        }
        self._append_jsonl(self.interaction_log_path, payload)

    def _log_interaction(
        self,
        *,
        tick: int,
        prompt_name: str,
        user_prompt: str,
        assistant_response: str,
        status: str,
        parsed: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        llm_usage: Optional[Dict[str, Optional[int]]] = None,
        reasoning_text: Optional[str] = None
    ) -> None:
        if not self.interaction_log_path:
            return
        payload: Dict[str, Any] = {
            "event": "interaction",
            "tick": int(tick),
            "rank": int(self.agent.model.rank),
            "agent_id": self._agent_id(),
            "prompt_name": prompt_name,
            "status": status,
            "user_prompt": user_prompt,
            "assistant_response": assistant_response,
        }
        if parsed:
            payload["parsed"] = parsed
        if error:
            payload["error"] = error
        if llm_usage is not None:
            payload["llm_usage"] = llm_usage
        if reasoning_text:
            payload["reasoning_text"] = reasoning_text
        self._append_jsonl(self.interaction_log_path, payload)

    def _to_int_or_none(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _reasoning_tokens_from_details(self, details: Any) -> Optional[int]:
        if details is None:
            return None
        if isinstance(details, dict):
            return self._to_int_or_none(details.get("reasoning_tokens"))
        value = getattr(details, "reasoning_tokens", None)
        return self._to_int_or_none(value)

    def _extract_usage(
        self, response: Any
    ) -> Optional[Dict[str, Optional[int]]]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None

        if isinstance(usage, dict):
            prompt_tokens = self._to_int_or_none(usage.get("prompt_tokens"))
            completion_tokens = self._to_int_or_none(
                usage.get("completion_tokens")
            )
            total_tokens = self._to_int_or_none(usage.get("total_tokens"))
            reasoning_tokens = self._reasoning_tokens_from_details(
                usage.get("completion_tokens_details")
            )
            if reasoning_tokens is None:
                reasoning_tokens = self._reasoning_tokens_from_details(
                    usage.get("output_tokens_details")
                )
        else:
            prompt_tokens = self._to_int_or_none(
                getattr(usage, "prompt_tokens", None)
            )
            completion_tokens = self._to_int_or_none(
                getattr(usage, "completion_tokens", None)
            )
            total_tokens = self._to_int_or_none(
                getattr(usage, "total_tokens", None)
            )
            reasoning_tokens = self._reasoning_tokens_from_details(
                getattr(usage, "completion_tokens_details", None)
            )
            if reasoning_tokens is None:
                reasoning_tokens = self._reasoning_tokens_from_details(
                    getattr(usage, "output_tokens_details", None)
                )

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "reasoning_tokens": reasoning_tokens,
        }

    def _extract_reasoning_text(self, choice: Any) -> Optional[str]:
        candidates: List[Any] = []
        for attr in ("reasoning_content", "reasoning", "reasoning_text"):
            candidates.append(getattr(choice, attr, None))
        if isinstance(choice, dict):
            for key in ("reasoning_content", "reasoning", "reasoning_text"):
                candidates.append(choice.get(key))
        for value in candidates:
            text = self._normalize_reasoning_text(value)
            if text:
                return text
        return None

    def _normalize_reasoning_text(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        if isinstance(value, dict):
            for key in ("text", "content", "reasoning"):
                if key in value:
                    text = self._normalize_reasoning_text(value.get(key))
                    if text:
                        return text
            return json.dumps(value)
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                text = self._normalize_reasoning_text(item)
                if text:
                    parts.append(text)
            if parts:
                return "\n".join(parts)
            return None
        text = str(value).strip()
        return text if text else None

    def _log_llm_failure(
        self, stage: str, content: str, error_msg: str
    ) -> None:
        """Append a brief log line for bad / empty LLM responses."""
        try:
            folder = getattr(self.agent.model, "out_folder_name", ".")
            Path(folder).mkdir(parents=True, exist_ok=True)
            path = (
                Path(folder)
                / f"failed_llm_responses_rank{self.agent.model.rank}.log"
            )
            tick = (
                self.agent.model.runner.schedule.tick
                if self.agent and self.agent.model
                else -1
            )
            line = (
                f"tick={tick} agent={self.agent.uid} policy=memory_stream "
                f"stage={stage} endpoint={self.llm_endpoint} "
                f"error=\"{error_msg}\" content=\"{str(content)[:800]}\"\n"
            )
            with open(path, "a") as f:
                f.write(line)
        except Exception:
            pass

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    def decide_next_action(self) -> tuple[int, Optional[int]]:
        """Sync wrapper for async decision (for compatibility)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.decide_next_action_async())
        raise RuntimeError(
            "decide_next_action() called from an active event loop; "
            "use decide_next_action_async() instead."
        )

    async def decide_next_action_async(
        self,
        llm_client=None
    ) -> tuple[int, Optional[int]]:
        """Ask the LLM for the next activity using retrieved memories."""
        if llm_client is not None:
            self.llm_client = llm_client
        if self.llm_client is None:
            self.llm_client = openai.AsyncOpenAI(
                base_url=self.llm_endpoint,
                api_key="sk-no-key-required"
            )
        self._record_state_snapshot()
        # Day plan is already ensured by model._gather_day_plans() before
        # _gather_llm_decisions() calls this method.  Only call here as a
        # safety-net if no plan was produced yet (e.g. standalone usage).
        if self.use_day_planner and not bool(
            self.day_plan_status.get("attempted", False)
        ):
            await self.ensure_day_plan_async(self.llm_client)

        current_tick = self.agent.model.runner.schedule.tick
        focus_tags = self._current_focus_tags()
        selected_memories: List[Dict[str, Any]] = []
        memory_candidates: List[Dict[str, Any]] = []
        if self.memory_ablation_mode != "no_memory":
            (
                selected_memories,
                memory_candidates
            ) = self._select_memories_with_scores(
                current_tick,
                focus_tags=focus_tags
            )
        prompt = self._construct_prompt(selected_memories)
        system_prompt = self._render_shared_system_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        self._log_system_prompt_once(
            tick=current_tick,
            prompt_name="decision",
            system_prompt=system_prompt,
        )

        async def call_and_parse(stage: str):
            # request_timeout = 6*60
            request_timeout = None
            for attempt in range(2):
                try:
                    request_kwargs: Dict[str, Any] = {
                        "model": self.llm_model,
                        "messages": messages,
                        "temperature": self.decision_temperature,
                        "response_format": {"type": "json_object"},
                        "timeout": request_timeout,
                        "reasoning_effort": REASONING_EFFORT,
                    }
                    if self.decision_top_p is not None:
                        request_kwargs["top_p"] = self.decision_top_p
                    response = await self.llm_client.chat.completions.create(
                        **request_kwargs
                    )
                    choice = response.choices[0].message
                    content = choice.content or ""
                    llm_usage = self._extract_usage(response)
                    reasoning_text = self._extract_reasoning_text(choice)
                    if not content:
                        self._log_llm_failure(
                            stage, str(choice), "empty content"
                        )
                        return None
                    try:
                        parsed = self._parse_response(content)
                        return parsed, content, llm_usage, reasoning_text
                    except Exception as e:
                        self._log_llm_failure(
                            stage, content, f"{type(e).__name__}: {e}"
                        )
                        return None
                except (
                    openai.APITimeoutError, openai.APIConnectionError
                ) as e:
                    self._log_llm_failure(
                        stage, "", f"{type(e).__name__}: {e}"
                    )
                    await asyncio.sleep(1.0)
                    continue
                except Exception as e:
                    self._log_llm_failure(
                        stage, "", f"{type(e).__name__}: {e}"
                    )
                    return None
            return None

        result = await call_and_parse("try1")
        if result is None:
            result = await call_and_parse("try2")

        if result is None:
            fallback_activity = int(getattr(self.agent, "activity_type", 1))
            if fallback_activity not in VALID_ACTIVITY_TYPES:
                fallback_activity = 1
            fallback_stay_minutes = LLM_DEFAULT_STAY_MINUTES
            fallback_rationale = "fallback after invalid LLM response"
            self._record_decision_memory(
                fallback_rationale,
                fallback_activity,
                fallback_stay_minutes,
                current_tick
            )
            fallback_parsed = {
                "next_activity_type": fallback_activity,
                "stay_minutes": fallback_stay_minutes,
                "rationale": fallback_rationale,
            }
            self._log_interaction(
                tick=current_tick,
                prompt_name="decision",
                user_prompt=prompt,
                assistant_response="",
                status="fallback",
                parsed=fallback_parsed,
                error="invalid_or_empty_or_oov_activity_response",
            )
            reflection_executed = False
            if self.memory_ablation_mode != "no_reflection":
                reflection_executed = await self._maybe_reflect_async(
                    current_tick
                )
            self._log_decision_diagnostics(
                tick=current_tick,
                status="fallback",
                focus_tags=focus_tags,
                selected_memories=selected_memories,
                memory_candidates=memory_candidates,
                parsed=fallback_parsed,
                llm_usage=None,
                reasoning_text_chars=0,
                assistant_response_chars=0,
                error="invalid_or_empty_or_oov_activity_response",
                reflection_executed=reflection_executed,
            )
            return fallback_activity, fallback_stay_minutes

        parsed_decision, assistant_content, llm_usage, reasoning_text = result
        next_activity_type = int(parsed_decision["next_activity_type"])
        stay_minutes = parsed_decision.get("stay_minutes")
        rationale = parsed_decision.get("rationale")
        if stay_minutes is None:
            stay_minutes = LLM_DEFAULT_STAY_MINUTES
        parsed_payload = {
            "next_activity_type": next_activity_type,
            "stay_minutes": stay_minutes,
            "rationale": rationale,
        }
        self._log_interaction(
            tick=current_tick,
            prompt_name="decision",
            user_prompt=prompt,
            assistant_response=assistant_content,
            status="ok",
            parsed=parsed_payload,
            llm_usage=llm_usage,
            reasoning_text=reasoning_text,
        )
        self._record_decision_memory(
            rationale,
            next_activity_type,
            stay_minutes,
            current_tick
        )
        reflection_executed = False
        if not self.pending_reflection_force:
            if self.memory_ablation_mode != "no_reflection":
                reflection_executed = await self._maybe_reflect_async(
                    current_tick
                )
        self._log_decision_diagnostics(
            tick=current_tick,
            status="ok",
            focus_tags=focus_tags,
            selected_memories=selected_memories,
            memory_candidates=memory_candidates,
            parsed=parsed_payload,
            llm_usage=llm_usage,
            reasoning_text_chars=len(reasoning_text or ""),
            assistant_response_chars=len(assistant_content or ""),
            error=None,
            reflection_executed=reflection_executed,
        )

        return next_activity_type, stay_minutes

    def on_new_day(self, day: int) -> None:
        """Optional daily hook to trigger reflections."""
        if self.agent is None or self.agent.model is None:
            return
        if self.llm_client is None:
            endpoint = (
                self.agent.llm_endpoint
                or self.agent.model.params.get("llm_endpoint")
            )
            if endpoint and hasattr(self.agent.model, "_async_llm_clients"):
                self.llm_client = self.agent.model._async_llm_clients.get(
                    endpoint
                )
        current_tick = self.agent.model.runner.schedule.tick
        if self.use_day_planner:
            self._reset_day_plan_for_day(int(current_tick // 288))
        self.pending_reflection_force = True
        self.pending_reflection_tick = current_tick

    async def run_pending_reflection_async(self) -> None:
        if not self.pending_reflection_force:
            return
        if self.memory_ablation_mode == "no_reflection":
            self.pending_reflection_force = False
            self.pending_reflection_tick = None
            return
        current_tick = (
            self.pending_reflection_tick
            if self.pending_reflection_tick is not None
            else self.agent.model.runner.schedule.tick
        )
        await self._maybe_reflect_async(
            current_tick,
            force=True,
            trigger="pending_reflection"
        )
        self.pending_reflection_force = False
        self.pending_reflection_tick = None

    # --------------------------------------------------------------------- #
    # Memory bookkeeping                                                    #
    # --------------------------------------------------------------------- #

    def _record_state_snapshot(self) -> None:
        tick = self.agent.model.runner.schedule.tick
        date_str, time_str = self.agent._tick_to_datetime(tick)
        location_label = activity_prompt_label(
            self.agent.activity_type,
            default="Unknown",
        )
        location_tag = activity_internal_label(
            self.agent.activity_type,
            default="unknown",
        )
        minutes_here = 0
        start_tick = getattr(self.agent, "activity_start_tick", None)
        if start_tick is not None:
            minutes_here = max(0, (tick - start_tick) * 5)

        # NEEDS_REMOVAL_MARKER: removed needs-specific snapshot text.
        # text = (
        #     f"At {time_str} on {date_str} you were at {location} for "
        #     f"{minutes_here} minutes; feelings: "
        #     f"work drive {self._need_to_label(...)}, "
        #     f"hunger {self._need_to_label(...)}, "
        #     f"social desire {self._need_to_label(...)}, "
        #     f"errands {self._need_to_label(...)}, "
        #     f"tiredness {self._need_to_label(...)}."
        # )
        text = (
            f"At {time_str} on {date_str} you were at {location_label} for "
            f"{minutes_here} minutes."
        )
        # NEEDS_REMOVAL_MARKER: removed need-derived snapshot tag.
        # needs = {"work": ..., "food": ..., "social": ..., "errands": ...}
        # highest_need = max(needs, key=needs.get)
        tags = {location_tag}
        self._add_memory(
            tick=tick,
            text=text,
            tags=tags,
            importance=0.7,
            mtype="observation"
        )

    def _record_decision_memory(
        self,
        rationale: Optional[str],
        activity_type: int,
        stay_minutes: Optional[int],
        tick: int
    ) -> None:
        location_label = activity_prompt_label(activity_type, "Activity")
        location_tag = activity_internal_label(activity_type, "activity")
        stay_text = (
            f"planned to stay ~{stay_minutes} minutes"
            if stay_minutes is not None
            else "did not specify stay duration"
        )
        text = (
            f"Decided to go to {location_label} and {stay_text}."
        )
        rationale_text = str(rationale or "").strip()
        if rationale_text:
            text = f"{text} Rationale: {rationale_text}"
        tags = {location_tag}
        self._add_memory(
            tick=tick,
            text=text,
            tags=tags,
            importance=1.2,
            mtype="decision",
            metadata={
                "activity_type": int(activity_type),
                "stay_minutes": stay_minutes,
                "rationale": rationale_text or None,
            },
        )

    def _add_memory(
        self,
        *,
        tick: int,
        text: str,
        tags: Optional[Sequence[str]] = None,
        importance: float = 1.0,
        mtype: str = "observation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.agent is None or self.agent.model is None:
            return
        tags_set: Set[str] = set()
        if tags:
            tags_set = {str(t).lower() for t in tags}
        _, day_of_week = self._get_human_readable_datetime(tick)
        if day_of_week:
            tags_set.add(str(day_of_week).strip().lower())
        date_str, time_str = self.agent._tick_to_datetime(tick)
        entry = {
            "tick": tick,
            "time_label": f"{date_str} {time_str}",
            "text": text,
            "tags": tags_set,
            "importance": float(importance),
            "type": mtype
        }
        if metadata:
            entry.update(metadata)
        self.memories.append(entry)
        if len(self.memories) > self.store_limit:
            self.memories.pop(0)
            self.last_reflection_index = max(0, self.last_reflection_index - 1)

        if self.memory_log_path:
            debug_entry = dict(entry)
            debug_entry["tags"] = sorted(entry["tags"])
            debug_entry["event"] = "memory"
            debug_entry["rank"] = int(self.agent.model.rank)
            debug_entry["agent_id"] = self._agent_id()
            self._append_jsonl(self.memory_log_path, debug_entry)

    # --------------------------------------------------------------------- #
    # Prompt construction                                                   #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _need_to_label(value: float) -> str:
        """Convert a 0-4 need float to a qualitative band."""
        if value < 0.5:
            return "low"
        elif value < 1.0:
            return "moderate"
        elif value < 2.0:
            return "high"
        else:
            return "very high"

    def _construct_prompt(
        self,
        selected_memories: List[Dict[str, Any]]
    ) -> str:
        current_tick = self.agent.model.runner.schedule.tick
        current_datetime_str, day_of_week_str = \
            self._get_human_readable_datetime(current_tick)
        current_activity_name = activity_prompt_label(
            self.agent.activity_type, "unknown activity"
        )

        minutes_here = 0
        start_tick = getattr(self.agent, "activity_start_tick", None)
        if start_tick is not None:
            minutes_here = max(0, (current_tick - start_tick) * 5)

        memory_section = self._format_memories(selected_memories)
        if memory_section:
            memory_section = memory_section.replace("\n", "\n    ")
        today_activity_table = (
            self.agent.get_today_activity_table()
            if self.agent is not None else "(no data)"
        )
        if today_activity_table:
            today_activity_table = today_activity_table.replace(
                "\n", "\n    "
            )
        agent_profile_context = self._build_agent_profile_context()
        if agent_profile_context:
            agent_profile_context = agent_profile_context.replace(
                "\n", "\n    "
            )
        plan_section = self._build_day_plan_section()
        if plan_section:
            plan_section = plan_section.replace("\n", "\n    ")
        behavior_priors_block = "(disabled)"
        if self.include_behavior_priors:
            behavior_priors = self._build_behavior_priors(day_of_week_str)
            if behavior_priors:
                behavior_priors = behavior_priors.replace("\n", "\n    ")
                behavior_priors_block = behavior_priors

        prompt_values = {
            "day": current_datetime_str,
            "day_of_week": day_of_week_str,
            "bfrom": current_activity_name,
            "minutes_here": int(minutes_here),
            "agent_profile_context": agent_profile_context,
            "plan_section": plan_section,
            "behavior_priors_block": behavior_priors_block,
            "work_need": self._need_to_label(self.agent.work_need[0]),
            "food_need": self._need_to_label(self.agent.food_need[0]),
            "social_need": self._need_to_label(self.agent.social_need[0]),
            "errands_need": self._need_to_label(self.agent.errands_need[0]),
            "rest_need": self._need_to_label(self.agent.rest_need[0]),
            "memory_section": memory_section,
            "today_activity_table": today_activity_table,
            "activity_codes_block": format_activity_codes_block(indent=4),
            "activity_notes_block": format_activity_notes_block(indent=4),
            "event_context": self._event_context_for_prompt(
                "decision",
                current_tick,
            ),
        }
        decision_prompt_template = str(
            self.prompts.get("decision_prompt_template") or ""
        )
        try:
            prompt = decision_prompt_template.format_map(
                _SafeFormatDict(prompt_values)
            )
        except Exception:
            prompt = decision_prompt_template
        return prompt

    def _construct_day_planner_prompt(
        self,
        selected_memories: List[Dict[str, Any]],
    ) -> str:
        current_tick = self._current_tick()
        current_datetime_str, day_of_week_str = \
            self._get_human_readable_datetime(current_tick)
        current_activity_name = activity_prompt_label(
            self.agent.activity_type, "unknown activity"
        )
        minutes_here = 0
        start_tick = getattr(self.agent, "activity_start_tick", None)
        if start_tick is not None:
            minutes_here = max(0, (current_tick - start_tick) * 5)

        memory_section = self._format_memories(selected_memories)
        if memory_section:
            memory_section = memory_section.replace("\n", "\n    ")
        today_activity_table = (
            self.agent.get_today_activity_table()
            if self.agent is not None else "(no data)"
        )
        if today_activity_table:
            today_activity_table = today_activity_table.replace(
                "\n", "\n    "
            )
        agent_profile_context = self._build_agent_profile_context()
        if agent_profile_context:
            agent_profile_context = agent_profile_context.replace(
                "\n", "\n    "
            )
        planner_prompt_values = {
            "day": current_datetime_str,
            "day_of_week": day_of_week_str,
            "bfrom": current_activity_name,
            "minutes_here": int(minutes_here),
            "agent_profile_context": agent_profile_context,
            "work_need": self._need_to_label(self.agent.work_need[0]),
            "food_need": self._need_to_label(self.agent.food_need[0]),
            "social_need": self._need_to_label(self.agent.social_need[0]),
            "errands_need": self._need_to_label(self.agent.errands_need[0]),
            "rest_need": self._need_to_label(self.agent.rest_need[0]),
            "memory_section": memory_section,
            "today_activity_table": today_activity_table,
            "activity_vocab": activity_vocab_text(),
            "activity_vocab_prompt_labels": prompt_activity_vocab_text(),
            "activity_codes_block": format_activity_codes_block(indent=4),
            "activity_notes_block": format_activity_notes_block(indent=4),
            # "plan_grammar_hint": get_step_plan_reminder(),
            "event_context": self._event_context_for_prompt(
                "day_planner",
                current_tick,
            ),
        }
        planner_prompt_template = str(
            self.prompts.get("day_planner_prompt_template") or ""
        )
        try:
            return planner_prompt_template.format_map(
                _SafeFormatDict(planner_prompt_values)
            )
        except Exception:
            return planner_prompt_template

    def _build_day_plan_section(self) -> str:
        if not self.use_day_planner:
            return ""
        plan_line = get_plan_line(self.day_plan)
        if not plan_line:
            return ""
        # reminder = get_step_plan_reminder()
        lines = [
            # "Today plan (soft guidance, created at first decision):",
            f"{plan_line}",
            # f"Plan reminder: {reminder}",
        ]
        return "\n".join(lines)

    def _build_agent_profile_context(self) -> str:
        agent_type = validated_agent_type(
            getattr(self.agent, "agent_type", None),
            context=(
                f"MemoryStreamPolicy(agent_id={self._agent_id()}) "
                "profile context agent_type"
            ),
        )
        role = role_label_for_agent_type(
            agent_type,
            context=(
                f"MemoryStreamPolicy(agent_id={self._agent_id()}) "
                "profile context role"
            ),
        )
        gender = "male" if self.agent.gender == 0 else "female"
        lines = [
            f"- Role: {role}",
            f"- Age: {int(self.agent.age)}",
            f"- Gender: {gender}",
        ]

        attrs = getattr(self.agent, "agent_attrs", {}) or {}
        semantic_fields = [
            ("employment_status", "Employment status"),
            ("work_schedule_type", "Work schedule"),
            ("school_type", "School type"),
            ("subtype_hint", "Subtype"),
            ("worker_type", "Worker type"),
            ("household_vehicle_count", "Household vehicle count"),
            ("household_income_band", "Household income"),
            ("household_size", "Household size"),
            ("work_building_tag", "Work building tag"),
            ("occupation", "Occupation"),
            ("household_adults", "Household adults"),
            ("household_children", "Household children"),
            ("household_elder", "Household elder"),
        ]
        for key, label in semantic_fields:
            value = attrs.get(key)
            if self._is_unknown(value):
                continue
            lines.append(f"- {label}: {value}")

        return "\n".join(lines)

    def _build_behavior_priors(self, day_of_week: str) -> str:
        agent_type = validated_agent_type(
            getattr(self.agent, "agent_type", None),
            context=(
                f"MemoryStreamPolicy(agent_id={self._agent_id()}) "
                "behavior priors agent_type"
            ),
        )
        day_type = (
            "weekend" if day_of_week in {"Saturday", "Sunday"} else "weekday"
        )
        attrs = getattr(self.agent, "agent_attrs", {}) or {}
        subtype = str(attrs.get("subtype_hint", "unknown")).strip().lower()
        work_schedule = str(
            attrs.get("work_schedule_type", "unknown")
        ).strip().lower()
        school_type = str(attrs.get("school_type", "unknown")).strip().lower()
        vehicle_count = attrs.get("household_vehicle_count")

        lines = [
            "- Use these priors as soft guidance, not hard constraints.",
            "- Keep decisions coherent with the day timeline and avoid rapid "
            "oscillation between activity types.",
            "- Pick stay_minutes that are long enough to meaningfully satisfy "
            "the chosen need.",
        ]

        if agent_type == WORKER:
            lines.append(
                "- Worker prior: on weekdays, maintain a coherent work block"
                " unless the day context strongly suggests otherwise."
            )
            lines.append(
                "- Worker prior: late evening and night usually transition "
                "toward home unless an urgent unmet need dominates."
            )
            if subtype == "worker_full_time" or work_schedule == "full_time":
                lines.append(
                    "- Full-time hint: prefer fewer, longer work segments over"
                    " many short fragmented ones."
                )
            elif work_schedule == "part_time":
                lines.append(
                    "- Part-time hint: shorter or flexible work windows are "
                    "acceptable if the day remains coherent."
                )
        elif agent_type == STUDENT:
            lines.append(
                "- Student prior: on weekdays, include a plausible school "
                "block unless this is clearly a non-school day."
            )
            lines.append(
                "- Student prior: evening should typically trend toward home "
                "and recovery."
            )
            if school_type == "college" or subtype == "student_college":
                lines.append(
                    "- College hint: timing can be more flexible, but keep "
                    "transitions plausible and not overly fragmented."
                )
            elif school_type in {
                "elementary_school",
                "middle_school",
                "high_school",
                "kindergarten",
            } or subtype in {
                "student_elementary_school",
                "student_middle_school",
                "student_high_school",
                "student_kindergarten",
            }:
                lines.append(
                    "- K-12 hint: school participation should look more "
                    "structured on weekdays."
                )
        elif agent_type == HOMEMAKER:
            lines.append(
                "- Homemaker prior: keep a plausible home-centered daily "
                "rhythm with practical household and family-oriented "
                "adjustments."
            )

        if day_type == "weekend":
            lines.append(
                "- Weekend prior: mandatory work/school participation can be "
                "lower, but avoid extreme behavior spikes."
            )

        if vehicle_count is not None:
            try:
                v = int(vehicle_count)
                if v == 0:
                    lines.append(
                        "- Mobility access hint: household_vehicle_count=0 "
                        "(no household vehicle access). Prefer plausible local"
                        " or transit-compatible choices when relevant."
                    )
                elif v > 0:
                    lines.append(
                        f"- Mobility access hint: household_vehicle_count={v} "
                        "(household vehicle access available). This can widen "
                        "feasible options, but should not force long trips."
                    )
            except Exception:
                pass

        return "\n".join(lines)

    def _is_unknown(self, value: Any) -> bool:
        if value is None:
            return True
        value_str = str(value).strip()
        if value_str == "":
            return True
        return value_str.lower() == "unknown"

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _get_human_readable_datetime(self, tick: int) -> tuple[str, str]:
        """
        Converts simulation tick to a human-readable datetime string.
        """
        if self.agent.model is None:
            return ("", "")
        minutes_per_tick = 5
        ticks_per_day = 288

        day_offset = tick // ticks_per_day
        minutes_today = (tick % ticks_per_day) * minutes_per_tick

        current_date = self.agent.model.survey_start_date + timedelta(
            days=day_offset
        )
        hours = minutes_today // 60
        minutes = minutes_today % 60

        date_str = current_date.strftime("%Y-%m-%d")
        time_str = f"{hours:02d}:{minutes:02d}"
        return f"{date_str} {time_str}", current_date.strftime("%A")

    def _format_memories(
        self,
        memories: List[Dict[str, Any]]
    ) -> str:
        if not memories:
            return "- No notable memories yet."

        lines = []
        for mem in memories:
            if mem.get("type") == "reflection":
                label = "reflection"
            else:
                label = "memory"
            text = str(mem.get("text", "")).strip().replace("\n", " ")
            if text:
                text = " ".join(text.split())
            lines.append(
                f"- [{mem['time_label']}] ({label}) {text}"
            )
        return "\n".join(lines)

    # --------------------------------------------------------------------- #
    # Memory retrieval                                                      #
    # --------------------------------------------------------------------- #

    def _select_memories(
        self,
        current_tick: int
    ) -> List[Dict[str, Any]]:
        selected, _ = self._select_memories_with_scores(current_tick)
        return selected

    def _select_planner_reflection_memories(
        self,
        current_tick: int,
        focus_tags: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.memories:
            return []
        if focus_tags is None:
            focus_tags = self._current_focus_tags()

        reflection_scored: List[Tuple[float, int, Dict[str, Any]]] = []
        for idx, mem in enumerate(self.memories):
            if mem.get("type") != "reflection":
                continue
            components = self._score_memory_components(
                mem,
                current_tick,
                focus_tags,
            )
            reflection_scored.append(
                (float(components["total_score"]), -idx, mem)
            )

        if not reflection_scored:
            return []

        reflection_scored.sort(reverse=True)
        selected = [
            mem
            for _, _, mem in reflection_scored[:self.max_context_memories]
        ]
        selected.sort(key=lambda m: m["tick"], reverse=True)
        return selected

    def _select_memories_with_scores(
        self,
        current_tick: int,
        focus_tags: Optional[Set[str]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self.memories:
            return [], []

        if focus_tags is None:
            focus_tags = self._current_focus_tags()
        scored: List[Tuple[float, int, Dict[str, Any], Dict[str, Any]]] = []
        for idx, mem in enumerate(self.memories):
            components = self._score_memory_components(
                mem,
                current_tick,
                focus_tags
            )
            score = float(components["total_score"])
            # Use negative index so that more recent entries win ties.
            scored.append((score, -idx, mem, components))

        scored.sort(reverse=True)
        selected_entries: List[
            Tuple[float, int, Dict[str, Any], Dict[str, Any]]
        ] = []
        selected_memory_ids: Set[int] = set()

        def _select_from_pool(
            pool: List[Tuple[float, int, Dict[str, Any], Dict[str, Any]]],
            max_items: int
        ) -> None:
            if max_items <= 0:
                return
            for entry in pool:
                memory = entry[2]
                memory_id = id(memory)
                if memory_id in selected_memory_ids:
                    continue
                selected_entries.append(entry)
                selected_memory_ids.add(memory_id)
                if len(selected_entries) >= self.max_context_memories:
                    return
                if max_items is not None:
                    max_items -= 1
                    if max_items <= 0:
                        return

        # 1) Up to 2 reflections
        reflection_pool = [
            entry for entry in scored
            if entry[2].get("type") == "reflection"
        ]
        _select_from_pool(reflection_pool, 2)

        # 2) Up to 3 decisions with non-empty rationale
        decision_rationale_pool = []
        for entry in scored:
            memory = entry[2]
            if memory.get("type") != "decision":
                continue
            rationale = str(memory.get("rationale") or "").strip()
            if rationale:
                decision_rationale_pool.append(entry)
        _select_from_pool(decision_rationale_pool, 3)

        # 3) Fill remaining slots with observations
        observation_pool = [
            entry for entry in scored
            if entry[2].get("type") == "observation"
        ]
        remaining = self.max_context_memories - len(selected_entries)
        _select_from_pool(observation_pool, remaining)

        # 4) Final safety fill from any remaining memories by score
        remaining = self.max_context_memories - len(selected_entries)
        _select_from_pool(scored, remaining)

        selected = [mem for _, _, mem, _ in selected_entries]
        selected.sort(key=lambda m: m["tick"], reverse=True)

        candidates: List[Dict[str, Any]] = []
        for _, _, mem, components in scored:
            candidate = self._serialize_memory_for_diagnostics(mem)
            candidate["age_ticks"] = components["age_ticks"]
            candidate["score_components"] = {
                "recency": components["recency_score"],
                "importance": components["importance_score"],
                "relevance": components["relevance_score"],
                "tag_overlap_count": components["overlap_count"],
            }
            candidate["score"] = components["total_score"]
            candidates.append(candidate)
        return selected, candidates

    def _score_memory(
        self,
        memory: Dict[str, Any],
        current_tick: int,
        focus_tags: Set[str]
    ) -> float:
        components = self._score_memory_components(
            memory,
            current_tick,
            focus_tags
        )
        return float(components["total_score"])

    def _score_memory_components(
        self,
        memory: Dict[str, Any],
        current_tick: int,
        focus_tags: Set[str]
    ) -> Dict[str, Any]:
        memory_tick = int(memory.get("tick", current_tick))
        age = max(1, current_tick - memory_tick)
        recency_score = self.recency_weight / age
        importance = float(memory.get("importance", 0.0))
        importance_score = self.importance_weight * importance
        memory_tags = set(self._normalize_tags(memory.get("tags")))
        overlap = len(focus_tags & memory_tags)
        relevance_score = self.relevance_weight * overlap
        total_score = recency_score + importance_score + relevance_score
        return {
            "age_ticks": age,
            "recency_score": recency_score,
            "importance_score": importance_score,
            "relevance_score": relevance_score,
            "overlap_count": overlap,
            "total_score": total_score,
        }

    def _current_focus_tags(self) -> Set[str]:
        tags = set()
        location = activity_internal_label(self.agent.activity_type, "unknown")
        tags.add(location)
        current_tick = self._current_tick()
        _, day_of_week = self._get_human_readable_datetime(current_tick)
        if day_of_week:
            tags.add(str(day_of_week).strip().lower())
        # NEEDS_REMOVAL_MARKER: removed need-derived focus tag.
        # needs = {"work": ..., "food": ..., "social": ..., "errands": ...}
        # highest_need = max(needs, key=needs.get)
        # tags.add(highest_need)
        attrs = getattr(self.agent, "agent_attrs", {}) or {}
        for key in (
            "subtype_hint",
            "employment_status",
            "work_schedule_type",
            "school_type",
            "worker_type",
            "work_building_tag",
            "household_children",
            "household_elder",
        ):
            value = str(attrs.get(key, "")).strip().lower()
            if value and value != "unknown":
                tags.add(value)
        return {t.lower() for t in tags}

    # Activity keywords used by focus tags — match against reflection text
    _REFLECTION_TAG_KEYWORDS = {
        "home": ["home", "residential", "rest", "sleep"],
        "work": ["work", "office", "job", "employment", "shift"],
        "eat_meal": [
            "restaurant", "food", "eat", "meal", "lunch", "dinner",
            "breakfast", "hungry", "cafe"
        ],
        "education": [
            "school", "class", "study", "student", "lecture", "daycare",
            "child care", "childcare"
        ],
        "recreational": [
            "recreation", "leisure", "park",
            "gym", "exercise"
        ],
        "shopping": [
            "shopping", "shop", "store", "buy goods", "buy services",
            "mall", "market"
        ],
        "care": [
            "health", "healthcare", "doctor", "clinic", "hospital",
            "adult care"
        ],
        "community": [
            "religious", "church", "temple", "community", "volunteer"
        ],
        "social_visit": [
            "visit", "friends", "friend", "relatives", "family"
        ],
        "other": [
            "errand", "errands", "transfer", "transport", "pickup",
            "pick-up", "drop off", "drop-off", "other"
        ],
    }

    def _extract_reflection_tags(self, summary: str) -> Set[str]:
        """Extract activity-related tags from reflection text."""
        tags: Set[str] = {"reflection"}
        text_lower = summary.lower()
        for tag, keywords in self._REFLECTION_TAG_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                tags.add(tag)
        return tags

    # --------------------------------------------------------------------- #
    # Reflection                                                            #
    # --------------------------------------------------------------------- #

    async def _maybe_reflect_async(
        self,
        current_tick: int,
        force: bool = False,
        trigger: str = "decision_loop"
    ) -> bool:
        if self.memory_ablation_mode == "no_reflection":
            return False
        if not self.prompts.get("reflection_prompt_template"):
            return False

        elapsed_ticks = max(0, current_tick - self.last_reflection_tick)
        interval_ticks = self._non_negative_int(
            self.reflection_interval_ticks,
            default=288
        )
        interval_ticks = max(48, interval_ticks)
        interval_ready = (
            self.enable_interval_reflections
            and interval_ticks > 0
            and (
                elapsed_ticks >= interval_ticks
            )
        )
        count_ready = (
            self.enable_interval_reflections
            and elapsed_ticks >= interval_ticks
            and self.reflection_memory_count > 0
            and (
                len(self.memories) - self.last_reflection_index
            ) >= self.reflection_memory_count
        )
        # REFLECTION_POLICY_MARKER: by default we run only end-of-day forced
        # reflections. Interval/count triggers are enabled only when
        # memory_stream_enable_interval_reflections=True.
        if not force:
            if not self.enable_interval_reflections:
                return False
            if not interval_ready and not count_ready:
                return False

        target_day_index = max(0, int(current_tick) - 1) // 288

        def _memory_tick(mem: Dict[str, Any]) -> int:
            try:
                return int(mem.get("tick", current_tick))
            except Exception:
                return int(current_tick)

        same_day_non_reflections = [
            mem for mem in self.memories
            if mem.get("type") != "reflection"
            and (_memory_tick(mem) // 288) == target_day_index
        ]
        prior_reflections = [
            mem for mem in self.memories
            if mem.get("type") == "reflection"
            and _memory_tick(mem) < int(current_tick)
        ]
        prior_reflections.sort(key=_memory_tick, reverse=True)
        prior_reflections = prior_reflections[:6]
        prior_reflections.sort(key=_memory_tick)

        # Reflection context policy:
        # 1) include all non-reflection memories from the target day.
        # 2) include up to 6 prior reflections for continuity.
        reflection_inputs = same_day_non_reflections + prior_reflections
        if not reflection_inputs:
            return False

        memory_dump_lines = [
            f"- [{mem['time_label']}] {mem['text']}"
            for mem in reflection_inputs
        ]
        memory_dump = "\n".join(memory_dump_lines)
        if memory_dump:
            memory_dump = memory_dump.replace("\n", "\n    ")
        context_tick = max(0, int(current_tick) - 1)
        _, context_day_of_week = self._get_human_readable_datetime(
            context_tick
        )
        context_day, context_time = self.agent._tick_to_datetime(
            context_tick
        )
        reflection_prompt_values = {
            "memory_dump": memory_dump,
            "day_of_week": context_day_of_week,
            "day": context_day,
            "time": context_time,
            "event_context": self._event_context_for_prompt(
                "reflection",
                context_tick,
            ),
        }
        reflection_prompt_template = str(
            self.prompts.get("reflection_prompt_template") or ""
        )
        try:
            prompt = reflection_prompt_template.format_map(
                _SafeFormatDict(reflection_prompt_values)
            )
        except Exception:
            prompt = reflection_prompt_template
        reflection_system_prompt = self._render_reflection_system_prompt()
        messages = [
            {
                "role": "system",
                "content": reflection_system_prompt
            },
            {"role": "user", "content": prompt}
        ]
        self._log_system_prompt_once(
            tick=current_tick,
            prompt_name="reflection",
            system_prompt=reflection_system_prompt,
        )

        request_timeout = None
        try:
            request_kwargs: Dict[str, Any] = {
                "model": self.llm_model,
                "messages": messages,
                "temperature": self.reflection_temperature,
                "timeout": request_timeout,
                "reasoning_effort": REASONING_EFFORT,
            }
            if self.reflection_top_p is not None:
                request_kwargs["top_p"] = self.reflection_top_p
            response = await self.llm_client.chat.completions.create(
                **request_kwargs
            )
        except Exception as exc:
            self._log_llm_failure(
                "reflection", "", f"{type(exc).__name__}: {exc}"
            )
            self._log_reflection_diagnostics(
                tick=current_tick,
                status="error",
                sampled_memories=reflection_inputs,
                llm_usage=None,
                reasoning_text_chars=0,
                summary_chars=0,
                force=force,
                trigger=trigger,
                error=f"{type(exc).__name__}: {exc}",
            )
            return False
        choice = response.choices[0].message
        summary = (choice.content or "").strip()
        llm_usage = self._extract_usage(response)
        reasoning_text = self._extract_reasoning_text(choice)
        self._log_interaction(
            tick=current_tick,
            prompt_name="reflection",
            user_prompt=prompt,
            assistant_response=summary,
            status="ok",
            llm_usage=llm_usage,
            reasoning_text=reasoning_text,
        )

        self._add_memory(
            tick=current_tick,
            text=f"Reflection summary: {summary}",
            tags=self._extract_reflection_tags(summary),
            importance=1.5,
            mtype="reflection"
        )
        self.last_reflection_tick = current_tick
        self.last_reflection_index = len(self.memories)
        self._log_reflection_diagnostics(
            tick=current_tick,
            status="ok",
            sampled_memories=reflection_inputs,
            llm_usage=llm_usage,
            reasoning_text_chars=len(reasoning_text or ""),
            summary_chars=len(summary),
            force=force,
            trigger=trigger,
        )
        return True

    # --------------------------------------------------------------------- #
    # Response parsing                                                      #
    # --------------------------------------------------------------------- #

    def _parse_response(
        self,
        response: str
    ) -> Dict[str, Any]:
        try:
            response_json = json.loads(response)
        except json.JSONDecodeError:
            return {
                "next_activity_type": 1,
                "stay_minutes": None,
                "rationale": None,
            }

        next_activity_type = int(
            response_json.get("next_activity_type", 1)
        )
        if next_activity_type not in VALID_ACTIVITY_TYPES:
            raise ValueError(
                "next_activity_type must be one of "
                f"{sorted(VALID_ACTIVITY_TYPES)}"
            )
        stay_minutes = response_json.get("stay_minutes")
        if stay_minutes is not None:
            try:
                stay_minutes = int(stay_minutes)
            except (TypeError, ValueError):
                stay_minutes = None
        rationale: Optional[str] = None
        rationale_raw = response_json.get("rationale")
        if rationale_raw is not None:
            rationale_text = str(rationale_raw).strip()
            if rationale_text:
                rationale = rationale_text

        return {
            "next_activity_type": next_activity_type,
            "stay_minutes": stay_minutes,
            "rationale": rationale,
        }

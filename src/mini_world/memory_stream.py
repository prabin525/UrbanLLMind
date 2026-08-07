import json
import os
from datetime import timedelta
from importlib import resources
from typing import Any, Dict, List, Optional, Sequence, Set

import openai
import yaml
# from termcolor import cprint

from mini_world.config import dic_map_goal
from mini_world.agent_types import (
    role_label_for_agent_type, validated_agent_type
)
from pathlib import Path
import time


class MemoryStreamPolicy:
    """
    LLM policy that maintains a memory stream similar to Generative Agents.
    """

    def __init__(self, agent, llm_endpoint: str, llm_model: str) -> None:
        self.agent = agent
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.llm_client = openai.OpenAI(
            base_url=self.llm_endpoint,
            api_key="sk-no-key-required"
        )

        params = self.agent.model.params if self.agent.model else {}
        prompt_override = str(
            params.get("memory_stream_prompt_path", "")
        ).strip()
        if prompt_override:
            with open(prompt_override, "r", encoding="utf-8") as f:
                self.prompts = yaml.safe_load(f) or {}
        else:
            text = resources.files("mini_world.prompts").joinpath(
                "memory_stream.yaml"
            ).read_text(encoding="utf-8")
            self.prompts = yaml.safe_load(text) or {}

        gender_str = "male" if self.agent.gender == 0 else "female"
        agent_type = validated_agent_type(
            getattr(self.agent, "agent_type", None),
            context="memory_stream agent_type",
        )
        agent_type_str = role_label_for_agent_type(
            agent_type,
            context="memory_stream role",
        )
        self.persona = self.prompts["persona_template"].format(
            age=self.agent.age,
            gender=gender_str,
            agent_type_str=agent_type_str
        )
        # cprint(f"Agent {self.agent.uid} Persona:\n{self.persona}\n", "green")

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
        self.log_memories = self._as_bool(
            params.get(
                "memory_stream_log_memories",
                params.get("memory_stream_debug_log", False)
            )
        )
        self.log_interactions = self._as_bool(
            params.get("memory_stream_log_interactions", True)
        )

        self.memories: List[Dict[str, Any]] = []
        self.last_reflection_tick: int = (
            self.agent.model.runner.schedule.tick
            if self.agent and self.agent.model
            else 0
        )
        self.last_reflection_index: int = 0

        self.memory_log_path: Optional[str] = None
        self.interaction_log_path: Optional[str] = None
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
        error: Optional[str] = None
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
        self._append_jsonl(self.interaction_log_path, payload)

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
        """Ask the LLM for the next activity using retrieved memories."""
        self._record_state_snapshot()

        current_tick = self.agent.model.runner.schedule.tick
        selected_memories = self._select_memories(current_tick)
        prompt = self._construct_prompt(selected_memories)
        # cprint(f"Agent {self.agent.uid} Prompt:\n{prompt}\n", "cyan")

        messages = [
            {"role": "system", "content": self.persona},
            {"role": "user", "content": prompt}
        ]
        self._log_system_prompt_once(
            tick=current_tick,
            prompt_name="decision",
            system_prompt=self.persona,
        )

        def call_and_parse(stage: str):
            request_timeout = None
            for attempt in range(2):
                try:
                    response = self.llm_client.chat.completions.create(
                        model=self.llm_model,
                        messages=messages,
                        temperature=0.7,
                        response_format={"type": "json_object"},
                        timeout=request_timeout
                    )
                    choice = response.choices[0].message
                    # cot = getattr(choice, "reasoning_content", None)
                    content = choice.content or ""
                    # cprint(
                    #     f"Agent {self.agent.uid}\n"
                    #     f"Response: {content}\n"
                    #     f"Chain of Thought: {cot}\n",
                    #     "magenta"
                    # )
                    if not content:
                        self._log_llm_failure(
                            stage, str(choice), "empty content"
                        )
                        return None
                    try:
                        parsed = self._parse_response(content)
                        return parsed, content
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
                    time.sleep(1.0)
                    continue
                except Exception as e:
                    self._log_llm_failure(
                        stage, "", f"{type(e).__name__}: {e}"
                    )
                    return None
            return None

        result = call_and_parse("try1")
        if result is None:
            result = call_and_parse("try2")

        if result is None:
            # fallback_content = (
            #     '{"reasoning": "fallback after invalid LLM response", '
            #     '"next_activity_type": 1, "stay_minutes": 15}'
            # )
            self._record_decision_memory(
                "fallback after invalid LLM response",
                1,
                15,
                current_tick
            )
            self._log_interaction(
                tick=current_tick,
                prompt_name="decision",
                user_prompt=prompt,
                assistant_response="",
                status="fallback",
                parsed={
                    "next_activity_type": 1,
                    "stay_minutes": 15,
                    "reasoning": "fallback after invalid LLM response",
                },
                error="invalid_or_empty_response",
            )
            self._maybe_reflect(current_tick)
            return 1, 15

        parsed_decision, assistant_content = result
        next_activity_type, stay_minutes, reasoning = parsed_decision
        self._log_interaction(
            tick=current_tick,
            prompt_name="decision",
            user_prompt=prompt,
            assistant_response=assistant_content,
            status="ok",
            parsed={
                "next_activity_type": next_activity_type,
                "stay_minutes": stay_minutes,
                "reasoning": reasoning,
            },
        )
        self._record_decision_memory(
            reasoning,
            next_activity_type,
            stay_minutes,
            current_tick
        )
        self._maybe_reflect(current_tick)

        return next_activity_type, stay_minutes

    def on_new_day(self, day: int) -> None:
        """Optional daily hook to trigger reflections."""
        if self.agent is None or self.agent.model is None:
            return
        current_tick = self.agent.model.runner.schedule.tick
        self._maybe_reflect(current_tick, force=True)

    # --------------------------------------------------------------------- #
    # Memory bookkeeping                                                    #
    # --------------------------------------------------------------------- #

    def _record_state_snapshot(self) -> None:
        tick = self.agent.model.runner.schedule.tick
        date_str, time_str = self.agent._tick_to_datetime(tick)
        location = dic_map_goal.get(self.agent.activity_type, "unknown")
        minutes_here = 0
        start_tick = getattr(self.agent, "activity_start_tick", None)
        if start_tick is not None:
            minutes_here = max(0, (tick - start_tick) * 5)

        needs = {
            "work": self.agent.work_need[0],
            "food": self.agent.food_need[0],
            "social": self.agent.social_need[0],
            "errands": self.agent.errands_need[0]
        }
        highest_need = max(needs, key=needs.get)
        text = (
            f"At {time_str} on {date_str} you were at {location} for "
            f"{minutes_here} minutes; needs are "
            f"Work {needs['work']:.2f}, Food {needs['food']:.2f}, "
            f"Social {needs['social']:.2f}, Errands {needs['errands']:.2f}."
        )
        tags = {location, highest_need}
        self._add_memory(
            tick=tick,
            text=text,
            tags=tags,
            importance=0.7,
            mtype="observation"
        )

    def _record_decision_memory(
        self,
        reasoning: str,
        activity_type: int,
        stay_minutes: Optional[int],
        tick: int
    ) -> None:
        location = dic_map_goal.get(activity_type, "activity")
        stay_text = (
            f"planned to stay ~{stay_minutes} minutes"
            if stay_minutes is not None
            else "did not specify stay duration"
        )
        text = (
            f"Decided to go to {location} and {stay_text}. "
            f"Reasoning: {reasoning}"
        )
        tags = {location}
        self._add_memory(
            tick=tick,
            text=text,
            tags=tags,
            importance=1.2,
            mtype="decision"
        )

    def _add_memory(
        self,
        *,
        tick: int,
        text: str,
        tags: Optional[Sequence[str]] = None,
        importance: float = 1.0,
        mtype: str = "observation"
    ) -> None:
        if self.agent is None or self.agent.model is None:
            return
        tags_set: Set[str] = set()
        if tags:
            tags_set = {str(t).lower() for t in tags}
        date_str, time_str = self.agent._tick_to_datetime(tick)
        entry = {
            "tick": tick,
            "time_label": f"{date_str} {time_str}",
            "text": text,
            "tags": tags_set,
            "importance": float(importance),
            "type": mtype
        }
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

    def _construct_prompt(
        self,
        selected_memories: List[Dict[str, Any]]
    ) -> str:
        current_tick = self.agent.model.runner.schedule.tick
        current_datetime_str, day_of_week_str = \
            self._get_human_readable_datetime(current_tick)
        current_activity_name = dic_map_goal.get(
            self.agent.activity_type, "unknown activity"
        )

        minutes_here = 0
        start_tick = getattr(self.agent, "activity_start_tick", None)
        if start_tick is not None:
            minutes_here = max(0, (current_tick - start_tick) * 5)

        memory_section = self._format_memories(selected_memories)
        if memory_section:
            memory_section = memory_section.replace("\n", "\n    ")

        prompt = self.prompts["decision_prompt_template"].format(
            day=current_datetime_str,
            day_of_week=day_of_week_str,
            bfrom=current_activity_name,
            minutes_here=int(minutes_here),
            work_need=self.agent.work_need[0],
            food_need=self.agent.food_need[0],
            social_need=self.agent.social_need[0],
            errands_need=self.agent.errands_need[0],
            memory_section=memory_section
        )
        return prompt

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
            return "• Nothing notable yet—your log is just getting started."

        lines = []
        for mem in memories:
            if mem.get("type") == "reflection":
                label = "reflection"
            else:
                label = "memory"
            lines.append(
                f"• [{mem['time_label']}] ({label}) {mem['text']}"
            )
        return "\n".join(lines)

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    # --------------------------------------------------------------------- #
    # Memory retrieval                                                      #
    # --------------------------------------------------------------------- #

    def _select_memories(
        self,
        current_tick: int
    ) -> List[Dict[str, Any]]:
        if not self.memories:
            return []

        focus_tags = self._current_focus_tags()
        scored: List[tuple[float, int, Dict[str, Any]]] = []
        for idx, mem in enumerate(self.memories):
            score = self._score_memory(mem, current_tick, focus_tags)
            # Use negative index so that more recent entries win ties.
            scored.append((score, -idx, mem))

        scored.sort(reverse=True)
        selected = [mem for _, _, mem in scored[: self.max_context_memories]]
        selected.sort(key=lambda m: m["tick"], reverse=True)
        return selected

    def _score_memory(
        self,
        memory: Dict[str, Any],
        current_tick: int,
        focus_tags: Set[str]
    ) -> float:
        age = max(1, current_tick - memory["tick"])
        recency_score = self.recency_weight / age
        importance_score = self.importance_weight * memory["importance"]
        overlap = len(focus_tags & memory["tags"])
        relevance_score = self.relevance_weight * overlap
        return recency_score + importance_score + relevance_score

    def _current_focus_tags(self) -> Set[str]:
        tags = set()
        location = dic_map_goal.get(self.agent.activity_type, "unknown")
        tags.add(location)

        needs = {
            "work": self.agent.work_need[0],
            "food": self.agent.food_need[0],
            "social": self.agent.social_need[0],
            "errands": self.agent.errands_need[0]
        }
        highest_need = max(needs, key=needs.get)
        tags.add(highest_need)
        return {t.lower() for t in tags}

    # --------------------------------------------------------------------- #
    # Reflection                                                            #
    # --------------------------------------------------------------------- #

    def _maybe_reflect(self, current_tick: int, force: bool = False) -> None:
        if not self.prompts.get("reflection_prompt_template"):
            return

        interval_ready = (
            self.reflection_interval_ticks > 0
            and (
                current_tick - self.last_reflection_tick
            ) >= self.reflection_interval_ticks
        )
        count_ready = (
            self.reflection_memory_count > 0
            and (
                len(self.memories) - self.last_reflection_index
            ) >= self.reflection_memory_count
        )

        if not force and not interval_ready and not count_ready:
            return

        recent_non_reflections = [
            m for m in self.memories if m.get("type") != "reflection"
        ]
        if not recent_non_reflections:
            return

        window_size = self.reflection_memory_count
        if window_size <= 0 or window_size > len(recent_non_reflections):
            window_size = min(10, len(recent_non_reflections))
        sample = recent_non_reflections[-window_size:]

        memory_dump_lines = [
            f"- [{mem['time_label']}] {mem['text']}"
            for mem in reversed(sample)
        ]
        memory_dump = "\n".join(memory_dump_lines)
        if memory_dump:
            memory_dump = memory_dump.replace("\n", "\n    ")
        prompt = self.prompts["reflection_prompt_template"].format(
            memory_dump=memory_dump
        )
        messages = [
            {
                "role": "system",
                "content": self.prompts["reflection_system_prompt"]
            },
            {"role": "user", "content": prompt}
        ]
        self._log_system_prompt_once(
            tick=current_tick,
            prompt_name="reflection",
            system_prompt=self.prompts["reflection_system_prompt"],
        )

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.5
        )
        summary = response.choices[0].message.content.strip()
        self._log_interaction(
            tick=current_tick,
            prompt_name="reflection",
            user_prompt=prompt,
            assistant_response=summary,
            status="ok",
        )

        self._add_memory(
            tick=current_tick,
            text=f"Reflection summary: {summary}",
            tags={"reflection"},
            importance=1.5,
            mtype="reflection"
        )
        self.last_reflection_tick = current_tick
        self.last_reflection_index = len(self.memories)

    # --------------------------------------------------------------------- #
    # Response parsing                                                      #
    # --------------------------------------------------------------------- #

    def _parse_response(
        self,
        response: str
    ) -> tuple[int, Optional[int], str]:
        try:
            response_json = json.loads(response)
        except json.JSONDecodeError:
            return 1, None, "Failed to parse response JSON."

        reasoning = response_json.get(
            "reasoning", "No reasoning provided."
        )
        next_activity_type = int(response_json.get("next_activity_type", 1))
        stay_minutes = response_json.get("stay_minutes")
        if stay_minutes is not None:
            try:
                stay_minutes = int(stay_minutes)
            except (TypeError, ValueError):
                stay_minutes = None

        return next_activity_type, stay_minutes, reasoning.strip()

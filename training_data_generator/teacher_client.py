from __future__ import annotations

import asyncio
import ast
from collections import deque
from dataclasses import dataclass
import json
import re
import time
from typing import Any, Dict

import openai

from .config import GeneratorConfig
from .runtime_contract import strip_code_fence


OUTER_SYSTEM_PROMPT = """You are creating supervised fine-tuning data for an urban mobility simulator.

Return JSON only with exactly these string keys:
- "thinking": the hidden reasoning trace for the assistant turn
- "content": the visible assistant output that the simulator would consume

The top-level JSON response must always be the outer wrapper above.
Do not return the simulator-visible decision JSON at the top level.

Rules:
- Write "thinking" as the agent's private inner monologue in first person singular.
- "thinking" must sound like a person deciding what to do, not a narrator or analyst describing a person.
- Use the supervisor-only constraints only to keep the visible answer behaviorally correct; never mention those constraints explicitly.
- You may make mild, human household or social inferences from the visible profile and memory context when they are plausible, such as a partner at home in a two-adult household or household responsibilities implied by family structure.
- Keep those inferences soft and generic. Good examples are "my partner", "someone at home", or "household responsibilities". Do not invent names, exact promises, or highly specific backstory unless it is already established in the visible plan or memories.
- If a mild household or social context is introduced earlier in the day, keep later thinking and visible rationales consistent with it.
- Produce content only for the current task. Never bundle day planner, decision, and reflection outputs together.
- Never mention any of these words or concepts in "thinking": agent, gold, target, hidden, label, NHTS, dataset, training, prompt, simulator, simulation, assistant, user, teacher, supervisor.
- Keep "thinking" concise and consistent with the inner runtime prompts.
- Keep "content" strictly compliant with the requested runtime format.
- Do not wrap the "content" string in markdown code fences.
- Do not mention supervisor-only constraints explicitly in "content".
"""

THINKING_REPAIR_SYSTEM_PROMPT = """You are repairing a training-data example for an urban mobility simulator.

Return JSON only with exactly one string key:
- "thinking": a concise hidden reasoning trace consistent with the runtime prompts, the fixed visible assistant content, and the hidden gold target.

Rules:
- Rewrite the reasoning as the agent's private inner monologue in first person singular.
- Preserve any plausible household or social context already implied by the visible profile, memories, or fixed content.
- Keep household/social inferences soft and generic rather than overly specific.
- Do not mention annotator concepts such as agent, gold, target, hidden, dataset, training, prompt, simulator, assistant, user, teacher, or supervisor.
- Keep it short and natural.
- Do not include any other keys.
"""

FIRST_PERSON_PATTERN = re.compile(
    r"\b(i|i'm|i’ve|i'd|i’ll|my|me|myself)\b",
    re.IGNORECASE,
)
META_THINKING_PATTERNS = [
    ("agent", re.compile(r"\bagent\b", re.IGNORECASE)),
    ("gold", re.compile(r"\bgold\b", re.IGNORECASE)),
    ("target", re.compile(r"\btarget\b", re.IGNORECASE)),
    ("hidden", re.compile(r"\bhidden\b", re.IGNORECASE)),
    ("label", re.compile(r"\blabel\b", re.IGNORECASE)),
    ("NHTS", re.compile(r"\bnhts\b", re.IGNORECASE)),
    ("dataset", re.compile(r"\bdataset\b", re.IGNORECASE)),
    ("training", re.compile(r"\btraining\b", re.IGNORECASE)),
    ("prompt", re.compile(r"\bprompt\b", re.IGNORECASE)),
    ("simulator", re.compile(r"\bsimulator\b", re.IGNORECASE)),
    ("simulation", re.compile(r"\bsimulation\b", re.IGNORECASE)),
    ("assistant", re.compile(r"\bassistant\b", re.IGNORECASE)),
    ("user", re.compile(r"\buser\b", re.IGNORECASE)),
    ("teacher", re.compile(r"\bteacher\b", re.IGNORECASE)),
    ("supervisor", re.compile(r"\bsupervisor\b", re.IGNORECASE)),
]


@dataclass(frozen=True)
class TeacherOutput:
    thinking: str
    content: str


class TeacherGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        task_type: str,
        outer_user_prompt: str,
        raw_outer_payload_text: str = "",
        raw_thinking: str = "",
        raw_content: str = "",
    ) -> None:
        super().__init__(message)
        self.task_type = task_type
        self.outer_user_prompt = outer_user_prompt
        self.raw_outer_payload_text = raw_outer_payload_text
        self.raw_thinking = raw_thinking
        self.raw_content = raw_content


class AsyncRequestRateLimiter:
    def __init__(self, *, max_requests: int, window_seconds: float) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            sleep_seconds = 0.0
            async with self._lock:
                now = time.monotonic()
                while (
                    self._timestamps
                    and now - self._timestamps[0] >= self._window_seconds
                ):
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max_requests:
                    self._timestamps.append(now)
                    return
                sleep_seconds = max(
                    0.0,
                    self._window_seconds - (now - self._timestamps[0]),
                )
            if sleep_seconds > 0:
                await asyncio.sleep(sleep_seconds)


class TeacherClient:
    def __init__(self, config: GeneratorConfig) -> None:
        client_kwargs: Dict[str, Any] = {"api_key": config.openai_api_key}
        if config.openai_api_base_url:
            client_kwargs["base_url"] = config.openai_api_base_url
        self._client = openai.AsyncOpenAI(**client_kwargs)
        self._config = config
        self._rate_limiter = (
            AsyncRequestRateLimiter(
                max_requests=config.max_requests_per_minute,
                window_seconds=config.rate_limit_window_seconds,
            )
            if config.enable_request_rate_limiter
            else None
        )

    async def _create_completion(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
    ):
        if self._rate_limiter is not None:
            await self._rate_limiter.acquire()
        if self._config.openai_model == "openai/gpt-oss-120b":
            # Special handling for the GPT-OSS-120B model
            return await self._client.chat.completions.create(
                model=self._config.openai_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=temperature,
                timeout=self._config.openai_timeout_seconds,
                reasoning_effort="high"
            )
        return await self._client.chat.completions.create(
            model=self._config.openai_model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            timeout=self._config.openai_timeout_seconds,
        )

    async def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await result

    async def generate_day_planner(
        self,
        *,
        runtime_system_prompt: str,
        runtime_user_prompt: str,
        gold_day_outline: str,
    ) -> TeacherOutput:
        return await self._generate(
            task_type="day_planner",
            runtime_system_prompt=runtime_system_prompt,
            runtime_user_prompt=runtime_user_prompt,
            gold_payload={"gold_day_outline": gold_day_outline},
            temperature=self._config.day_planner_temperature,
            validator=self._validate_day_planner,
        )

    async def generate_decision(
        self,
        *,
        runtime_system_prompt: str,
        runtime_user_prompt: str,
        gold_next_activity_type: int,
        gold_stay_minutes: int,
    ) -> TeacherOutput:
        return await self._generate(
            task_type="decision",
            runtime_system_prompt=runtime_system_prompt,
            runtime_user_prompt=runtime_user_prompt,
            gold_payload={
                "gold_next_activity_type": int(gold_next_activity_type),
                "gold_stay_minutes": int(gold_stay_minutes),
            },
            temperature=self._config.decision_temperature,
            validator=lambda thinking, content: self._validate_decision(
                thinking,
                content,
                gold_next_activity_type=int(gold_next_activity_type),
                gold_stay_minutes=int(gold_stay_minutes),
            ),
        )

    async def generate_reflection(
        self,
        *,
        runtime_system_prompt: str,
        runtime_user_prompt: str,
    ) -> TeacherOutput:
        return await self._generate(
            task_type="reflection",
            runtime_system_prompt=runtime_system_prompt,
            runtime_user_prompt=runtime_user_prompt,
            gold_payload={},
            temperature=self._config.reflection_temperature,
            validator=self._validate_reflection,
        )

    async def _generate(
        self,
        *,
        task_type: str,
        runtime_system_prompt: str,
        runtime_user_prompt: str,
        gold_payload: Dict[str, Any],
        temperature: float,
        validator,
    ) -> TeacherOutput:
        outer_user_prompt = self._build_outer_user_prompt(
            task_type=task_type,
            runtime_system_prompt=runtime_system_prompt,
            runtime_user_prompt=runtime_user_prompt,
            gold_payload=gold_payload,
        )
        for attempt in range(self._config.max_completion_retries):
            try:
                response = await self._create_completion(
                    messages=[
                        {"role": "system", "content": OUTER_SYSTEM_PROMPT},
                        {"role": "user", "content": outer_user_prompt},
                    ],
                    temperature=temperature,
                )
            except Exception as exc:
                if attempt + 1 >= self._config.max_completion_retries:
                    raise TeacherGenerationError(
                        f"{task_type} teacher request failed after retries: {exc}",
                        task_type=task_type,
                        outer_user_prompt=outer_user_prompt,
                    ) from exc
                continue

            message = response.choices[0].message
            payload_text = strip_code_fence(message.content or "")
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                if attempt + 1 >= self._config.max_completion_retries:
                    raise TeacherGenerationError(
                        f"{task_type} teacher did not return valid JSON: {payload_text}",
                        task_type=task_type,
                        outer_user_prompt=outer_user_prompt,
                        raw_outer_payload_text=payload_text,
                    ) from exc
                continue

            try:
                thinking, content = self._extract_outer_payload(
                    task_type=task_type,
                    payload=payload,
                    payload_text=payload_text,
                )
            except Exception as exc:
                if attempt + 1 >= self._config.max_completion_retries:
                    raise TeacherGenerationError(
                        f"{task_type} outer payload extraction failed: {exc}",
                        task_type=task_type,
                        outer_user_prompt=outer_user_prompt,
                        raw_outer_payload_text=payload_text,
                    ) from exc
                continue
            thinking_issue = self._thinking_style_issue(thinking)
            if thinking_issue is not None:
                thinking = await self._repair_thinking(
                    task_type=task_type,
                    runtime_system_prompt=runtime_system_prompt,
                    runtime_user_prompt=runtime_user_prompt,
                    gold_payload=gold_payload,
                    visible_content=content,
                    invalid_thinking=thinking,
                    issue=thinking_issue,
                )
            content = self._coerce_task_content(task_type, content)
            try:
                return validator(thinking, content)
            except Exception as exc:
                if attempt + 1 >= self._config.max_completion_retries:
                    raise TeacherGenerationError(
                        f"{task_type} output validation failed: {exc}",
                        task_type=task_type,
                        outer_user_prompt=outer_user_prompt,
                        raw_outer_payload_text=payload_text,
                        raw_thinking=thinking,
                        raw_content=content,
                    ) from exc
        raise RuntimeError(f"{task_type} teacher request exhausted retries")

    def _extract_outer_payload(
        self,
        *,
        task_type: str,
        payload: Dict[str, Any],
        payload_text: str,
    ) -> tuple[str, str]:
        if "thinking" in payload or "content" in payload:
            thinking = str(payload.get("thinking", "")).strip()
            content = str(payload.get("content", "")).strip()
            return thinking, content
        if task_type == "decision" and {
            "rationale",
            "next_activity_type",
            "stay_minutes",
        }.issubset(payload.keys()):
            normalized_content = json.dumps(
                {
                    "rationale": str(payload["rationale"]).strip(),
                    "next_activity_type": int(payload["next_activity_type"]),
                    "stay_minutes": int(payload["stay_minutes"]),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return "", normalized_content
        raise ValueError(
            f"{task_type} teacher returned unexpected outer JSON shape: {payload_text}"
        )

    async def _repair_thinking(
        self,
        *,
        task_type: str,
        runtime_system_prompt: str,
        runtime_user_prompt: str,
        gold_payload: Dict[str, Any],
        visible_content: str,
        invalid_thinking: str,
        issue: str,
    ) -> str:
        repair_user_prompt = self._build_thinking_repair_prompt(
            task_type=task_type,
            runtime_system_prompt=runtime_system_prompt,
            runtime_user_prompt=runtime_user_prompt,
            gold_payload=gold_payload,
            visible_content=visible_content,
            invalid_thinking=invalid_thinking,
            issue=issue,
        )
        for attempt in range(self._config.max_completion_retries):
            try:
                response = await self._create_completion(
                    messages=[
                        {"role": "system", "content": THINKING_REPAIR_SYSTEM_PROMPT},
                        {"role": "user", "content": repair_user_prompt},
                    ],
                    temperature=temperature_or_default(
                        self._config.decision_temperature,
                        fallback=self._config.day_planner_temperature,
                    ),
                )
            except Exception as exc:
                if attempt + 1 >= self._config.max_completion_retries:
                    raise TeacherGenerationError(
                        f"{task_type} thinking repair failed after retries: {exc}",
                        task_type=task_type,
                        outer_user_prompt=repair_user_prompt,
                        raw_content=visible_content,
                    ) from exc
                continue
            payload_text = strip_code_fence(response.choices[0].message.content or "")
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                if attempt + 1 >= self._config.max_completion_retries:
                    raise TeacherGenerationError(
                        f"{task_type} thinking repair did not return valid JSON: {payload_text}",
                        task_type=task_type,
                        outer_user_prompt=repair_user_prompt,
                        raw_outer_payload_text=payload_text,
                        raw_content=visible_content,
                    ) from exc
                continue
            thinking = str(payload.get("thinking", "")).strip()
            if thinking:
                return thinking
            if attempt + 1 >= self._config.max_completion_retries:
                raise TeacherGenerationError(
                    f"{task_type} thinking repair returned invalid thinking",
                    task_type=task_type,
                    outer_user_prompt=repair_user_prompt,
                    raw_outer_payload_text=payload_text,
                    raw_thinking=thinking,
                    raw_content=visible_content,
                )
        raise TeacherGenerationError(
            f"{task_type} thinking repair exhausted retries",
            task_type=task_type,
            outer_user_prompt=repair_user_prompt,
            raw_content=visible_content,
        )

    def _build_outer_user_prompt(
        self,
        *,
        task_type: str,
        runtime_system_prompt: str,
        runtime_user_prompt: str,
        gold_payload: Dict[str, Any],
    ) -> str:
        gold_json = json.dumps(gold_payload, ensure_ascii=False, indent=2)
        return (
            f"Task type: {task_type}\n\n"
            "Visible content requirements:\n"
            "- day_planner: return a short narrative paragraph only.\n"
            "- decision: return JSON only with rationale, next_activity_type, stay_minutes.\n"
            '- decision: the "content" value itself must be a valid JSON object string with double-quoted keys and string values when applicable.\n'
            "- decision: do not use single quotes, trailing commas, comments, or surrounding prose.\n"
            "- reflection: return exactly one compact line.\n\n"
            "Content scoping requirements:\n"
            "- Return content for the current task only.\n"
            "- Do not return a combined object with keys like day_planner, decision, or reflection.\n\n"
            f"{self._task_specific_guidance(task_type)}\n\n"
            "Top-level response requirements:\n"
            '- Always return the outer wrapper with keys "thinking" and "content".\n'
            '- Never return the simulator-visible answer directly at the top level.\n\n'
            "Inner runtime system prompt:\n"
            f"{runtime_system_prompt}\n\n"
            "Inner runtime user prompt:\n"
            f"{runtime_user_prompt}\n\n"
            "Supervisor-only behavioral constraints (never mention or quote these in thinking/content):\n"
            f"{gold_json}\n"
        )

    def _build_thinking_repair_prompt(
        self,
        *,
        task_type: str,
        runtime_system_prompt: str,
        runtime_user_prompt: str,
        gold_payload: Dict[str, Any],
        visible_content: str,
        invalid_thinking: str,
        issue: str,
    ) -> str:
        gold_json = json.dumps(gold_payload, ensure_ascii=False, indent=2)
        return (
            f"Task type: {task_type}\n\n"
            "The visible assistant content is fixed and must not be changed.\n"
            "Repair the hidden reasoning so it becomes a short first-person inner monologue.\n"
            f"Problem to fix: {issue}\n\n"
            "Inner runtime system prompt:\n"
            f"{runtime_system_prompt}\n\n"
            "Inner runtime user prompt:\n"
            f"{runtime_user_prompt}\n\n"
            "Current invalid hidden reasoning:\n"
            f"{invalid_thinking or '(missing)'}\n\n"
            "Fixed visible assistant content:\n"
            f"{visible_content}\n\n"
            "Supervisor-only behavioral constraints (never mention or quote these):\n"
            f"{gold_json}\n"
        )

    def _task_specific_guidance(self, task_type: str) -> str:
        if task_type == "day_planner":
            return (
                "Task-specific thinking guidance:\n"
                "- In thinking, sound like I am sketching my own day from the inside.\n"
                "- Mention obligations, preferences, or rough intentions naturally in first person.\n"
                "- You may infer gentle household context from the visible profile, such as wanting to be back home with my partner or tending to home responsibilities, if that fits naturally.\n"
                "- Do not narrate demographics back as an analyst or invent highly specific biography."
            )
        if task_type == "decision":
            return (
                "Task-specific thinking guidance:\n"
                "- In thinking, sound like I am choosing my next move right now.\n"
                "- Use natural first-person reasoning tied to time, fatigue, obligations, recent memories, and today's rough plan.\n"
                "- When the visible profile suggests it, you may naturally reference soft household or social motivations like getting back to my partner, getting home before it gets too late, or handling household responsibilities.\n"
                "- If the day plan or earlier memories imply a relationship or home-life context, preserve that framing instead of switching to a different story later in the day.\n"
                "- Keep those inferences generic and plausible; do not invent names, exact appointments, or detailed unsupported events.\n"
                "- Avoid quoting exact supervisor-only durations unless they are already obvious from the runtime context; prefer phrases like 'for a while', 'for most of the day', or 'briefly'."
            )
        return (
            "Task-specific thinking guidance:\n"
            "- In thinking, sound like I am privately reflecting on what stood out in my own routine.\n"
            "- Keep the reflection grounded in evidence from the day's memories, but phrase it as my own internal summary.\n"
            "- If the day consistently implied a soft household or relationship context, it is fine to preserve that framing in a generic way."
        )

    def _normalize_thinking(self, thinking: str) -> str:
        return " ".join(strip_code_fence(thinking).split())

    def _thinking_style_issue(self, thinking: str) -> str | None:
        normalized = self._normalize_thinking(thinking)
        if not normalized:
            return "missing thinking"
        if not FIRST_PERSON_PATTERN.search(normalized):
            return "thinking must be in first person singular"
        for label, pattern in META_THINKING_PATTERNS:
            if pattern.search(normalized):
                return f"thinking must not mention '{label}'"
        return None

    def _coerce_task_content(self, task_type: str, content: str) -> str:
        normalized = strip_code_fence(content).strip()
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(normalized)
            except (ValueError, SyntaxError):
                return content
        if not isinstance(parsed, dict):
            return content
        if task_type == "day_planner":
            planner_value = parsed.get("day_planner")
            if isinstance(planner_value, str) and planner_value.strip():
                return planner_value.strip()
            return content
        if task_type == "reflection":
            reflection_value = parsed.get("reflection")
            if isinstance(reflection_value, str) and reflection_value.strip():
                return reflection_value.strip()
            return content
        decision_value = parsed.get("decision")
        if isinstance(decision_value, dict):
            return json.dumps(decision_value, ensure_ascii=False, separators=(",", ":"))
        if isinstance(decision_value, str) and decision_value.strip():
            return decision_value.strip()
        return content

    def _validate_day_planner(
        self,
        thinking: str,
        content: str,
    ) -> TeacherOutput:
        thinking = self._normalize_thinking(thinking)
        thinking_issue = self._thinking_style_issue(thinking)
        if thinking_issue is not None:
            raise ValueError(thinking_issue)
        content = strip_code_fence(content).strip()
        if not content:
            raise ValueError("empty planner content")
        if content.startswith("{"):
            raise ValueError("planner content must not be JSON")
        return TeacherOutput(thinking=thinking, content=content)

    def _validate_decision(
        self,
        thinking: str,
        content: str,
        *,
        gold_next_activity_type: int,
        gold_stay_minutes: int,
    ) -> TeacherOutput:
        thinking = self._normalize_thinking(thinking)
        thinking_issue = self._thinking_style_issue(thinking)
        if thinking_issue is not None:
            raise ValueError(thinking_issue)
        parsed = self._parse_jsonish_object(content)
        rationale = str(parsed.get("rationale", "")).strip()
        next_activity_type = int(parsed.get("next_activity_type"))
        stay_minutes = int(parsed.get("stay_minutes"))
        if not rationale:
            raise ValueError("missing decision rationale")
        if next_activity_type != gold_next_activity_type:
            raise ValueError(
                "decision activity mismatch: "
                f"expected {gold_next_activity_type}, got {next_activity_type}"
            )
        if stay_minutes != gold_stay_minutes:
            raise ValueError(
                "decision duration mismatch: "
                f"expected {gold_stay_minutes}, got {stay_minutes}"
            )
        normalized_content = json.dumps(
            {
                "rationale": rationale,
                "next_activity_type": next_activity_type,
                "stay_minutes": stay_minutes,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return TeacherOutput(thinking=thinking, content=normalized_content)

    def _validate_reflection(
        self,
        thinking: str,
        content: str,
    ) -> TeacherOutput:
        thinking = self._normalize_thinking(thinking)
        thinking_issue = self._thinking_style_issue(thinking)
        if thinking_issue is not None:
            raise ValueError(thinking_issue)
        content = " ".join(strip_code_fence(content).split())
        if not content:
            raise ValueError("empty reflection content")
        return TeacherOutput(thinking=thinking, content=content)

    def _parse_jsonish_object(self, content: str) -> Dict[str, Any]:
        normalized = strip_code_fence(content).strip()
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError as json_exc:
            try:
                parsed = ast.literal_eval(normalized)
            except (ValueError, SyntaxError) as literal_exc:
                raise json_exc from literal_exc
        if not isinstance(parsed, dict):
            raise ValueError("decision content must decode to an object")
        return parsed


def temperature_or_default(primary: float, *, fallback: float) -> float:
    if primary > 0:
        return primary
    return fallback

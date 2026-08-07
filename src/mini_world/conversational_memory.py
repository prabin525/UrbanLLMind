import openai
import yaml
from importlib import resources
from datetime import datetime, timedelta
from typing import Optional
# from termcolor import cprint
import json
from pathlib import Path
import time

from mini_world.config import dic_map_goal
from mini_world.agent_types import (
    role_label_for_agent_type,
    validated_agent_type
)


class ConversationalMemory:
    """
    Manages the conversation with an LLM for a single agent,
    acting as the agent's "brain."
    """

    def __init__(self, agent, llm_endpoint: str, llm_model: str):
        """
        Initializes the ConversationalMemory.

        Args:
            agent: The agent instance that this memory belongs to.
            llm_endpoint: The URL of the local LLM service.
        """
        self.agent = agent
        self.llm_endpoint = llm_endpoint
        self.llm_client = openai.OpenAI(
            base_url=self.llm_endpoint,
            api_key="sk-no-key-required"
        )
        self.llm_model = llm_model
        self.full_history = []
        params = self.agent.model.params if self.agent.model else {}

        # Load prompts from YAML file
        prompts_file_path = str(
            params.get("conversational_prompt_path", "")
        ).strip()
        if prompts_file_path:
            with open(prompts_file_path, "r", encoding="utf-8") as f:
                self.prompts = yaml.safe_load(f) or {}
        else:
            text = resources.files("mini_world.prompts").joinpath(
                "conversational_memory.yaml"
            ).read_text(encoding="utf-8")
            self.prompts = yaml.safe_load(text) or {}

        summarization_prompts_file_path = str(
            params.get("summarization_prompt_path", "")
        ).strip()
        if summarization_prompts_file_path:
            with open(
                summarization_prompts_file_path,
                "r",
                encoding="utf-8",
            ) as f:
                self.summarization_prompts = yaml.safe_load(f) or {}
        else:
            summary_text = resources.files("mini_world.prompts").joinpath(
                "summarization.yaml"
            ).read_text(encoding="utf-8")
            self.summarization_prompts = yaml.safe_load(summary_text) or {}

        gender_str = "male" if self.agent.gender == 0 else "female"
        agent_type = validated_agent_type(
            getattr(self.agent, "agent_type", None),
            context="conversational_memory agent_type",
        )
        agent_type_str = role_label_for_agent_type(
            agent_type,
            context="conversational_memory role",
        )
        self.persona = self.prompts['persona_template'].format(
            age=self.agent.age,
            gender=gender_str,
            agent_type_str=agent_type_str
        )
        # cprint(f"Agent {self.agent.uid} Persona:\n{self.persona}\n", 'green')

        self.conversation_history = [
            {"role": "system", "content": self.persona}
        ]
        self.survey_start_date = datetime.strptime(
            self.agent.model.params['survey_start_date'], '%Y/%m/%d'
        )

    def _log_llm_failure(
        self, stage: str, content: str, error_msg: str
    ) -> None:
        """Append a brief log line for bad / empty LLM responses."""
        try:
            folder = getattr(self.agent.model, "out_folder_name", ".")
            Path(folder).mkdir(parents=True, exist_ok=True)
            rank = self.agent.model.rank
            path = Path(folder) / f"failed_llm_responses_rank{rank}.log"
            tick = (
                self.agent.model.runner.schedule.tick
                if self.agent and self.agent.model
                else -1
            )
            line = (
                f"tick={tick} agent={self.agent.uid} policy=conversational "
                f"stage={stage} endpoint={self.llm_endpoint} "
                f"error=\"{error_msg}\" content=\"{str(content)[:800]}\"\n"
            )
            with open(path, "a") as f:
                f.write(line)
        except Exception:
            # Logging must never crash the sim
            pass

    def decide_next_action(self) -> tuple[int, Optional[int]]:
        """
        Decides the agent's next action by querying the LLM.

        Returns:
            The activity type for the agent's next action.
        """
        # 1. Construct the prompt from the agent's current state
        prompt = self._construct_prompt()
        # cprint(f"Agent {self.agent.uid} Prompt:\n{prompt}\n", 'cyan')

        # 2. Append the user prompt to the conversation history
        self.conversation_history.append({"role": "user", "content": prompt})

        def call_and_parse(stage: str):
            # Small retry loop to survive transient timeouts
            request_timeout = 300  # seconds
            for attempt in range(2):
                try:
                    response = self.llm_client.chat.completions.create(
                        model=self.llm_model,
                        messages=self.conversation_history,
                        temperature=0.7,
                        response_format={"type": "json_object"},
                        reasoning_effort='medium',  # low, medium, high
                        timeout=request_timeout,
                    )
                    choice = response.choices[0].message
                    # cot = getattr(choice, "reasoning_content", None)
                    content = choice.content
                    # cprint(
                    #     f"Agent {self.agent.uid} \n"
                    #     f"Response: {content}\n"
                    #     f"Chain of Thought: {cot}\n",
                    #     'magenta'
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
                        stage,
                        "",
                        f"{type(e).__name__}: {e}"
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
            # Fallback: stay home briefly
            fallback_content = (
                '{"reasoning": "fallback after invalid LLM response", '
                '"next_activity_type": 1, "stay_minutes": 15}'
            )
            self.conversation_history.append(
                {"role": "assistant", "content": fallback_content}
            )
            return 1, 15

        (next_activity_type, stay_minutes), assistant_content = result
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_content}
        )
        return next_activity_type, stay_minutes

    def on_new_day(self, day: int) -> None:
        period = self.agent.model.params.get('summarization_period_days', 0)
        if (
            period and period > 0 and
            day > 0 and
            day % period == 0
        ):
            self.summarize_conversation()

    def summarize_conversation(self):
        """
        Summarizes the conversation history to keep it within a manageable
        size.
        """
        self.full_history.append(self.conversation_history)

        summarization_prompt = self.summarization_prompts[
            'summarization_prompt_template'
        ].format(history=str(self.conversation_history))
        # print(summarization_prompt)

        summarization_messages = [
            {
                "role": "system",
                "content": self.summarization_prompts['summarization_persona']
            },
            {"role": "user", "content": summarization_prompt}
        ]

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=summarization_messages,
            temperature=0.7,
        )

        summary_text = response.choices[0].message.content
        # cprint(
        #     f"Agent {self.agent.uid} Summary:\n{summary_text}\n",
        #     'blue'
        # )

        self.conversation_history = [
            {"role": "system", "content": self.persona},
            {
                "role": "user",
                "content": (
                    f"Here is a summary of your recent past: {summary_text}"
                )
            },
            {
                "role": "assistant",
                "content": (
                    "Thank you for that summary. I will take it into account "
                    "for my future decisions."
                )
            }
        ]

    def _get_human_readable_datetime(self, tick: int) -> tuple[str, str]:
        """
        Converts simulation tick to a human-readable date and time.
        """
        day_offset = int(tick / 288)
        minutes_today = int((tick % 288) * 5)

        current_date = self.survey_start_date + timedelta(days=day_offset)
        hours = minutes_today // 60
        minutes = minutes_today % 60

        date_str = current_date.strftime("%Y-%m-%d")
        time_str = f" {hours:02d}:{minutes:02d}"
        day_of_week_str = current_date.strftime('%A')

        return date_str + time_str, day_of_week_str

    def _construct_prompt(self) -> str:
        """
        Constructs the prompt for the LLM based on the agent's current state.
        """
        (
            current_datetime_str,
            day_of_week_str
        ) = self._get_human_readable_datetime(
            self.agent.model.runner.schedule.tick
        )
        current_activity_name = dic_map_goal.get(
            self.agent.activity_type, "unknown activity"
        )

        minutes_here = 0
        start_tick = getattr(self.agent, "activity_start_tick", None)
        if start_tick is not None and self.agent.model is not None:
            minutes_here = max(
                0,
                (self.agent.model.runner.schedule.tick - start_tick) * 5
            )

        prompt = self.prompts['decision_prompt_template'].format(
            day=current_datetime_str,
            day_of_week=day_of_week_str,
            bfrom=current_activity_name,
            minutes_here=int(minutes_here),
            work_need=self.agent.work_need[0],
            food_need=self.agent.food_need[0],
            social_need=self.agent.social_need[0],
            errands_need=self.agent.errands_need[0]
        )
        return prompt

    def _parse_response(self, response: str) -> tuple[int, Optional[int]]:
        """
        Parses the LLM's response to get the next activity type.
        """
        try:
            response_json = json.loads(response)
            reasoning = response_json.get(  # noqa: F841
                "reasoning",
                "No reasoning provided."
            )
            next_activity_type = response_json.get("next_activity_type", 1)
            stay_minutes = response_json.get("stay_minutes")
            if stay_minutes is not None:
                try:
                    stay_minutes = int(stay_minutes)
                except (TypeError, ValueError):
                    stay_minutes = None

            # You can log or store the reasoning here if needed
            # For now, we'll just print it
            # cprint(f"LLM Reasoning: {reasoning}", 'yellow')

            return next_activity_type, stay_minutes
        except json.JSONDecodeError:
            print(f"Error decoding JSON from LLM response: {response}")
            return 1, None  # Default to residential if parsing fails

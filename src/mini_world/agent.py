from repast4py import core
from repast4py.space import DiscretePoint as dpt
import numpy as np
from typing import Any, Dict, Optional, Tuple
from datetime import timedelta
# from conversational_memory import ConversationalMemory
# from memory_stream import MemoryStreamPolicy
from mini_world.memory_stream_async import MemoryStreamPolicy
from mini_world.utils import travel_time, getting_dwell_time
from mini_world.config import dic_map_goal
from mini_world.activity_taxonomy import (
    LLM_DEFAULT_STAY_TICKS,
    VALID_ACTIVITY_TYPES,
    activity_prompt_label,
    activity_route_infra_type,
    activity_route_policy,
)
from mini_world.agent_types import (
    cohort_label_for_agent_type,
    parse_agent_type,
)


# --- Defining worker class
class Worker(core.Agent):

    # --- Defining the type of agents
    TYPE = 1

    # --- This function initializes the agent
    def __init__(
        self, model, local_id: int, rank: int, pt: dpt, schedule_size: int,
        schedule: np.array, buildings: np.array, bfrom: int, bto: int,
        ttravel: int, tdwell: int, work_need: np.array, food_need: np.array,
        social_need: np.array, errands_need: np.array, rest_need: np.array,
        activity_type: int,
        decision_policy: str, age: int, gender: int,
        llm_endpoint: Optional[str],
        agent_type: int,
        agent_attrs: Optional[Dict[str, Any]] = None
    ):

        super().__init__(id=local_id, type=Worker.TYPE, rank=rank)
        self.model = model
        self.pt = pt
        self.schedule_size = schedule_size
        self.schedule = schedule
        self.buildings = buildings
        self.bfrom = bfrom
        self.bto = bto
        self.ttravel = ttravel
        self.tdwell = tdwell
        self.work_need = work_need
        self.food_need = food_need
        self.social_need = social_need
        self.errands_need = errands_need
        self.rest_need = rest_need
        self.activity_type = activity_type
        self.decision_policy = decision_policy
        self.age = age
        self.gender = gender
        self.agent_type = parse_agent_type(
            agent_type,
            context=f"Worker(agent_id={local_id}) agent_type",
        )  # 1 worker, 2 student, 3 homemaker
        self.agent_attrs = dict(agent_attrs) if agent_attrs else {}
        self.memory_model = None
        self.llm_endpoint = (
            llm_endpoint
            or (
                self.model.params.get("llm_endpoint") if self.model else None
            )
        )
        coords = pt.coordinates
        self.current_position = (
            int(coords[0]),
            int(coords[1])
        )
        self.last_logged_position: Optional[Tuple[int, int]] = None
        self.travel_origin: Tuple[int, int] = self.current_position
        self.travel_destination: Tuple[int, int] = self.current_position
        self.total_travel_ticks = 0
        self.pending_stay_ticks: Optional[int] = None
        self.current_stay_ticks_total = tdwell
        if self.model:
            self.activity_start_tick = self.model.runner.schedule.tick
        else:
            self.activity_start_tick = 0
        self.decay_activity_type: Optional[int] = None
        self.work_decay_per_tick = 0.0
        self.food_decay_per_tick = 0.0
        self.social_decay_per_tick = 0.0
        self.errands_decay_per_tick = 0.0
        self.rest_decay_per_tick = 0.0
        if self.model:
            self.today_day_index = self.model.runner.schedule.tick // 288
        else:
            self.today_day_index = 0
        self.today_segments = []
        self.day_plan_state: Dict[str, Any] = {}
        self.day_plan_status: Dict[str, Any] = {}

    def initialize_logging(self) -> None:
        if self.model is None:
            return
        self._record_position(
            self.current_position[0],
            self.current_position[1],
            self.model.runner.schedule.tick
        )

    def _record_position(self, x: int, y: int, tick: int) -> None:
        if (
            not hasattr(self.model, 'file_positions') or
            self.model.file_positions is None
        ):
            return
        if self.model.rank == 0 and self.last_logged_position != (x, y):
            tick_value = int(tick)
            self.model.file_positions.write(
                f"{tick_value},{self.id},{x},{y}\n"
            )
            self.last_logged_position = (x, y)

    def _move_and_log(self, x: float, y: float) -> None:
        if self.model is None:
            return
        x_int = int(round(x))
        y_int = int(round(y))
        self.current_position = (x_int, y_int)
        self.pt = dpt(x_int, y_int, 0)
        self.model.grid.move(self, self.pt)
        self._record_position(x_int, y_int, self.model.runner.schedule.tick)

    def _safe_rng_choice(self, candidates: Any) -> Optional[int]:
        if self.model is None or candidates is None:
            return None
        try:
            if len(candidates) == 0:
                return None
        except TypeError:
            return None
        try:
            return int(self.model.rng.choice(candidates))
        except ValueError:
            return None

    def _tick_to_datetime(self, tick: int) -> Tuple[str, str]:
        if self.model is None:
            return ("", "")
        day_offset = tick // 288
        minutes = (tick % 288) * 5
        dt = self.model.survey_start_date + \
            timedelta(days=day_offset, minutes=minutes)
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M")

    def _log_activity_period(self, end_tick: int) -> None:
        if self.model is None or self.activity_start_tick is None:
            return
        self._record_today_segment(
            self.activity_start_tick, end_tick, self.activity_type
        )
        if (
            not getattr(self.model, "create_activity_table", False) or
            self.model.activity_log_file is None
        ):
            self.activity_start_tick = None
            return
        first_start_tick = int(self.activity_start_tick)
        start_day = first_start_tick // 288
        # end_tick is exclusive, so (end_tick - 1) gives the day of the last
        # minute
        end_day_idx = int(end_tick - 1) // 288

        for d in range(start_day, end_day_idx + 1):
            # Calculate boundaries for this day
            seg_start = max(first_start_tick, d * 288)
            seg_end = min(end_tick, (d + 1) * 288)

            # Skip zero duration segments if any (shouldn't happen with proper
            # end_day_idx)
            if seg_end <= seg_start:
                continue

            d_str, arr_time = self._tick_to_datetime(seg_start)
            _, dep_time = self._tick_to_datetime(seg_end)

            location_label = dic_map_goal.get(self.activity_type, "unknown")
            line = (
                f"{self.id},{d_str},{location_label},{arr_time},"
                f"{dep_time}\n"
            )
            self.model.activity_log_file.write(line)
            self.model.activity_log_file.flush()

        self.activity_start_tick = None

    def _initiate_travel(self) -> None:
        if self.model is None:
            return
        if self.bfrom not in self.model.dB or self.bto not in self.model.dB:
            self.ttravel = 0
            self.total_travel_ticks = 0
            self.travel_origin = self.current_position
            self.travel_destination = self.current_position
            return
        self.travel_origin = (
            self.model.dB[self.bfrom]["x"],
            self.model.dB[self.bfrom]["y"]
        )
        self.travel_destination = (
            self.model.dB[self.bto]["x"],
            self.model.dB[self.bto]["y"]
        )
        if self.travel_origin == self.travel_destination:
            self.ttravel = 0
            self.total_travel_ticks = 0
            return
        self.ttravel = travel_time(
            self.bfrom,
            self.bto,
            self.model.dB,
            self.model.params['speed']
        )
        self.total_travel_ticks = self.ttravel

    def _finish_travel(self) -> None:
        if self.model is None:
            return
        stay_ticks = self.pending_stay_ticks
        if stay_ticks is None or stay_ticks <= 0:
            stay_ticks = getting_dwell_time(
                self.activity_type, self.model.params
            )
        self.tdwell = stay_ticks
        self.current_stay_ticks_total = stay_ticks
        self.pending_stay_ticks = None
        if self.model is not None:
            self.activity_start_tick = self.model.runner.schedule.tick
        # --- Get destination building
        dest_building_id = self.bto

        # --- Get coordinates
        x = self.model.dB[dest_building_id]["x"]
        y = self.model.dB[dest_building_id]["y"]

        # --- Resetting the variables
        self.bfrom = self.bto
        self.bto = -1
        self.total_travel_ticks = 0
        self.travel_origin = self.travel_destination

        # --- Moving the agent to this location
        self.model.grid.move(self, dpt(x, y, 0))

        # --- Adding activity
        self.adding_activity(self.activity_type)

        # --- Arrival log (monolithic style)
        if getattr(self.model, "file_activity", None):
            self.model.file_activity.write(
                f"{self.model.runner.schedule.tick},{self.id},"
                f"{dest_building_id},{self.activity_type}\n"
            )

    def _update_needs_and_daily(self) -> None:
        if self.model is not None:
            current_day = self.model.runner.schedule.tick // 288
            if self.today_day_index != current_day:
                self._reset_today_segments(current_day)
        if self.decision_policy == "memory_stream_llm":
            self._update_needs_memory_stream()
        else:
            # --- Updating needs of the agents
            self.work_need[0] += self.work_need[1]
            self.food_need[0] += self.food_need[1]
            self.social_need[0] += self.social_need[1]
            self.errands_need[0] += self.errands_need[1]
            self.rest_need[0] += self.rest_need[1]

        # --- Trigger for summarization at the end of the day ---
        if self.model.local_time() == 0 and self.memory_model is not None:
            day = int(self.model.runner.schedule.tick / 288)
            if hasattr(self.memory_model, "on_new_day"):
                self.memory_model.on_new_day(day)

        # --- Checking the time
        if (self.model.local_time() == 0):

            # --- Checking if we should print the schedule
            if self.model.params["print_schedule"]:
                # --- Logging agent
                self.logging_schedule_of_agents()

            # --- Zeroing the schedule
            self.schedule_size = 0
            self.schedule.fill(0)

            # --- Checking if agent is at home
            self.schedule[0] = self.activity_type

    def _reset_today_segments(self, day_index: int) -> None:
        self.today_day_index = day_index
        self.today_segments = []

    def _record_today_segment(
        self,
        start_tick: int,
        end_tick: int,
        activity_type: int
    ) -> None:
        if self.model is None:
            return
        current_day = end_tick // 288
        if self.today_day_index != current_day:
            self._reset_today_segments(current_day)
        day_start = current_day * 288
        day_end = day_start + 288
        if end_tick <= day_start or start_tick >= day_end:
            return
        seg_start = max(start_tick, day_start)
        seg_end = min(end_tick, day_end)
        if seg_end < seg_start:
            return
        self.today_segments.append((seg_start, seg_end, activity_type))

    def get_today_activity_table(self) -> str:
        if self.model is None:
            return "(no data)"
        current_tick = self.model.runner.schedule.tick
        current_day = current_tick // 288
        if self.today_day_index != current_day:
            self._reset_today_segments(current_day)
        day_start = current_day * 288
        lines = []
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

    def _is_valid_building_choice(self, building_id: Any) -> bool:
        try:
            bid = int(building_id)
        except Exception:
            return False
        return bid not in (-1, 0) and bid in self.model.dB

    def _sample_infra_building(self, infra_type: int) -> Optional[int]:
        choice = self._safe_rng_choice(self.model.dInfra[self.id][infra_type])
        if choice is None:
            return None
        return int(choice)

    def _route_llm_activity_destination(
        self,
        target_activity: int
    ) -> Optional[int]:
        policy = activity_route_policy(target_activity)
        if policy == "own_home":
            home_id = self.buildings[1]
            if self._is_valid_building_choice(home_id):
                return int(home_id)
            return None

        if policy == "assigned_work":
            work_id = self.buildings[2]
            if self._is_valid_building_choice(work_id):
                return int(work_id)
            return None

        if policy == "education_special":
            if self.agent_type == 2:
                school_id = self.buildings[3]
                if self._is_valid_building_choice(school_id):
                    return int(school_id)
            infra_type = activity_route_infra_type(target_activity)
            if infra_type is None:
                return None
            return self._sample_infra_building(int(infra_type))

        if policy == "sample_infra_type":
            infra_type = activity_route_infra_type(target_activity)
            if infra_type is None:
                return None
            return self._sample_infra_building(int(infra_type))

        if policy == "residential_not_own_home":
            own_home = (
                int(self.buildings[1])
                if len(self.buildings) > 1
                else -1
            )
            candidates = [
                int(bid)
                for bid in self.model.dInfra[self.id][1]
                if self._is_valid_building_choice(bid) and int(bid) != own_home
            ]
            return self._safe_rng_choice(candidates)

        return None

    def _calibrated_typical_ticks(
        self,
        activity_type: int
    ) -> Optional[int]:
        if self.model is None:
            return None
        dwell_cfg = self.model.params.get("dwell_calibration")
        if not isinstance(dwell_cfg, dict):
            return None
        cohort_key = cohort_label_for_agent_type(
            self.agent_type,
            context=(
                f"Worker(agent_id={self.id}) "
                "dwell calibration cohort lookup"
            ),
        )
        cohort_cfg = dwell_cfg.get(cohort_key)
        if not isinstance(cohort_cfg, dict):
            return None
        typical_map = cohort_cfg.get("typical_dwell_minutes")
        if not isinstance(typical_map, dict):
            return None

        activity_key = {
            1: "home",
            2: "work",
            3: "restaurant",
            4: "school",
            5: "recreation",
            7: "errands",
        }.get(int(activity_type))
        lookup_keys = []
        if activity_key is not None:
            lookup_keys.append(activity_key)
        lookup_keys.extend([str(int(activity_type)), int(activity_type)])
        if int(activity_type) == 4:
            lookup_keys.append("work")
        if int(activity_type) == 2:
            lookup_keys.append("school")

        for key in lookup_keys:
            if key not in typical_map:
                continue
            try:
                minutes = float(typical_map[key])
            except Exception:
                continue
            if minutes <= 0:
                continue
            return max(1, int(round(minutes / 5.0)))
        return None

    def _typical_decay_ticks(self, activity_type: int) -> int:
        calibrated = self._calibrated_typical_ticks(activity_type)
        if calibrated is not None:
            return calibrated
        if self.model is None:
            return 1
        return max(1, getting_dwell_time(activity_type, self.model.params))

    def _refresh_need_decay(self) -> None:
        self.decay_activity_type = self.activity_type
        self.work_decay_per_tick = 0.0
        self.food_decay_per_tick = 0.0
        self.social_decay_per_tick = 0.0
        self.errands_decay_per_tick = 0.0
        self.rest_decay_per_tick = 0.0
        if self.model is None:
            return
        if self.activity_type == 1:
            typical_ticks = self._typical_decay_ticks(self.activity_type)
            self.rest_decay_per_tick = self.rest_need[0] / typical_ticks
        if self.activity_type in (2, 4):
            typical_ticks = self._typical_decay_ticks(self.activity_type)
            self.work_decay_per_tick = self.work_need[0] / typical_ticks
        elif self.activity_type == 3:
            typical_ticks = self._typical_decay_ticks(self.activity_type)
            self.food_decay_per_tick = self.food_need[0] / typical_ticks
        elif self.activity_type == 5:
            typical_ticks = self._typical_decay_ticks(self.activity_type)
            self.social_decay_per_tick = self.social_need[0] / typical_ticks
        elif self.activity_type == 7:
            typical_ticks = self._typical_decay_ticks(self.activity_type)
            self.errands_decay_per_tick = self.errands_need[0] / typical_ticks

    def _update_needs_memory_stream(self) -> None:
        # Only apply decay while dwelling at the satisfying activity.
        if self.ttravel == 0 and self.tdwell > 0:
            if self.decay_activity_type != self.activity_type:
                self._refresh_need_decay()
            # Rest need (home)
            if self.activity_type == 1:
                self.rest_need[0] = max(
                    0.0, self.rest_need[0] - self.rest_decay_per_tick
                )
            else:
                self.rest_need[0] += self.rest_need[1]
            # Work / School need
            if self.activity_type in (2, 4):
                self.work_need[0] = max(
                    0.0, self.work_need[0] - self.work_decay_per_tick
                )
            else:
                self.work_need[0] += self.work_need[1]
            # Food need
            if self.activity_type == 3:
                self.food_need[0] = max(
                    0.0, self.food_need[0] - self.food_decay_per_tick
                )
            else:
                self.food_need[0] += self.food_need[1]
            # Social need
            if self.activity_type == 5:
                self.social_need[0] = max(
                    0.0, self.social_need[0] - self.social_decay_per_tick
                )
            else:
                self.social_need[0] += self.social_need[1]
            # Errands need
            if self.activity_type == 7:
                self.errands_need[0] = max(
                    0.0, self.errands_need[0] - self.errands_decay_per_tick
                )
            else:
                self.errands_need[0] += self.errands_need[1]
        else:
            # Travel or waiting for a decision: all needs grow
            self.work_need[0] += self.work_need[1]
            self.food_need[0] += self.food_need[1]
            self.social_need[0] += self.social_need[1]
            self.errands_need[0] += self.errands_need[1]
            self.rest_need[0] += self.rest_need[1]

    def post_decision_updates(self) -> None:
        self._update_needs_and_daily()

    def walk(self, allow_decision: bool = True) -> bool:
        decision_needed = False
        # --- Checking if the dwell time is higher than zero
        if self.tdwell > 0:

            # --- Subtracting one
            self.tdwell -= 1

        elif self.ttravel > 0:

            # --- Subtracting one
            self.ttravel -= 1

            # --- Checking if the agent arrived ad their destination
            if (self.ttravel == 0):
                self._finish_travel()

        # --- Checking if the agent will need to make a choice and travel
        elif (self.tdwell == 0):

            if allow_decision:
                self.setting_schedule()
            else:
                decision_needed = True

        if not decision_needed:
            self._update_needs_and_daily()

        return decision_needed

    def setting_schedule(self):
        if self.decision_policy == 'needs_based':
            self._execute_needs_based_policy()
        # elif self.decision_policy == 'conversational_llm':
        #     self._execute_conversational_llm_policy()
        elif self.decision_policy == 'memory_stream_llm':
            self._execute_memory_stream_llm_policy()
        if (
            self.model is not None and
            self.ttravel == 0 and
            self.bto != -1
        ):
            # Arrival handled in walk/_finish_travel; nothing to do here
            return

    def _execute_needs_based_policy(self):
        # --- This only happens if we are in the day
        if (self.model.local_time() > 60) and (self.model.local_time() < 264):

            # --- Checking if agent has been to work / school
            if (self.work_need[0] > 1):
                target_activity = None
                target_building = None
                dwell_goal = None

                if self.agent_type == 1:
                    target_activity = 2
                    target_building = self.buildings[2]
                    dwell_goal = 2
                elif self.agent_type == 2:
                    target_activity = 4
                    target_building = self.buildings[3]
                    dwell_goal = 4
                elif self.agent_type == 3:
                    # Homemakers do not have required work/school commutes in
                    # the legacy needs-based path.
                    self.work_need[0] = 0.0
                else:
                    raise ValueError(
                        f"Unsupported agent_type={self.agent_type} in "
                        f"needs_based policy for agent_id={self.id}"
                    )

                if (
                    target_building not in (None, -1, 0)
                    and dwell_goal is not None
                    and (
                        self.model.local_time() +
                        getting_dwell_time(dwell_goal, self.model.params)
                    ) < 264
                ):
                    # log current stay before leaving
                    if (
                        self.model is not None
                        and getattr(self.model, "activity_log_file", None)
                    ):
                        self._log_activity_period(
                            self.model.runner.schedule.tick
                        )
                    self.bto = target_building
                    self._initiate_travel()
                    self.work_need[0] = 0.0
                    self.activity_type = target_activity

            # --- Checking if agent needs food
            elif (self.food_need[0] > 1):

                if (
                    self.model is not None
                    and getattr(self.model, "activity_log_file", None)
                ):
                    self._log_activity_period(self.model.runner.schedule.tick)

                if (
                    self.model.local_time() +
                    getting_dwell_time(3, self.model.params)
                ) < 264:
                    choice = self._safe_rng_choice(
                        self.model.dInfra[self.id][3]
                    )
                    if choice is None:
                        return
                    self.bto = choice
                    self._initiate_travel()
                    self.food_need[0] = 0.0
                    self.activity_type = 3

            # --- Checking if agent social need
            elif (self.social_need[0] > 1):

                if (
                    self.model is not None
                    and getattr(self.model, "activity_log_file", None)
                ):
                    self._log_activity_period(self.model.runner.schedule.tick)

                if (
                    self.model.local_time() +
                    getting_dwell_time(5, self.model.params)
                ) < 264:
                    choice = self._safe_rng_choice(
                        self.model.dInfra[self.id][5]
                    )
                    if choice is None:
                        return
                    self.bto = choice
                    self._initiate_travel()
                    self.social_need[0] = 0.0
                    self.activity_type = 5

            # --- Checking if agent errands need
            elif (self.errands_need[0] > 1):

                if (
                    self.model is not None
                    and getattr(self.model, "activity_log_file", None)
                ):
                    self._log_activity_period(self.model.runner.schedule.tick)

                if (
                    self.model.local_time() +
                    getting_dwell_time(7, self.model.params)
                ) < 264:
                    choice = self._safe_rng_choice(
                        self.model.dInfra[self.id][7]
                    )
                    if choice is None:
                        return
                    self.bto = choice
                    self._initiate_travel()
                    self.errands_need[0] = 0.0
                    self.activity_type = 7

            # --- Sending agent home
            elif (self.activity_type != 1):
                if (
                    self.model is not None
                    and getattr(self.model, "activity_log_file", None)
                ):
                    self._log_activity_period(self.model.runner.schedule.tick)
                self.bto = self.buildings[1]
                self._initiate_travel()
                self.activity_type = 1

        else:
            if (self.activity_type != 1):
                if (
                    self.model is not None
                    and getattr(self.model, "activity_log_file", None)
                ):
                    self._log_activity_period(self.model.runner.schedule.tick)
                self.bto = self.buildings[1]
                self._initiate_travel()
                self.activity_type = 1

    # def _execute_conversational_llm_policy(self):
    #     """ Executes the LLM-based decision-making logic. """
    #     if (
    #         self.memory_model is None or
    #         not isinstance(self.memory_model, ConversationalMemory)
    #     ):
    #         endpoint = self.llm_endpoint or self.model.params['llm_endpoint']
    #         self.memory_model = ConversationalMemory(
    #             self,
    #             endpoint,
    #             self.model.params['llm_model']
    #         )

    #     next_activity_type, stay_minutes = \
    #         self.memory_model.decide_next_action()
    #     self._apply_llm_decision(next_activity_type, stay_minutes)

    def _execute_memory_stream_llm_policy(self):
        if (
            self.memory_model is None or
            not isinstance(self.memory_model, MemoryStreamPolicy)
        ):
            endpoint = self.llm_endpoint or self.model.params['llm_endpoint']
            self.memory_model = MemoryStreamPolicy(
                self,
                endpoint,
                self.model.params['llm_model']
            )

        next_activity_type, stay_minutes = \
            self.memory_model.decide_next_action()
        self._apply_llm_decision(next_activity_type, stay_minutes)

    async def get_memory_stream_decision_async(
        self,
        llm_client=None
    ) -> tuple[int, Optional[int]]:
        if (
            self.memory_model is None or
            not isinstance(self.memory_model, MemoryStreamPolicy)
        ):
            endpoint = self.llm_endpoint or self.model.params['llm_endpoint']
            self.memory_model = MemoryStreamPolicy(
                self,
                endpoint,
                self.model.params['llm_model']
            )

        return await self.memory_model.decide_next_action_async(llm_client)

    async def ensure_day_plan_async(self, llm_client=None) -> None:
        if self.decision_policy != "memory_stream_llm":
            return
        if (
            self.memory_model is None or
            not isinstance(self.memory_model, MemoryStreamPolicy)
        ):
            endpoint = self.llm_endpoint or self.model.params['llm_endpoint']
            self.memory_model = MemoryStreamPolicy(
                self,
                endpoint,
                self.model.params['llm_model']
            )
        await self.memory_model.ensure_day_plan_async(llm_client)

    def _apply_llm_decision(
        self,
        next_activity_type: int,
        stay_minutes: Optional[int]
    ) -> None:
        stay_ticks = None
        if stay_minutes is not None:
            stay_ticks = max(1, int(round(stay_minutes / 5.0)))
        if stay_ticks is None:
            stay_ticks = LLM_DEFAULT_STAY_TICKS

        target_activity = int(next_activity_type)
        if target_activity not in VALID_ACTIVITY_TYPES:
            target_activity = self.activity_type
            stay_ticks = LLM_DEFAULT_STAY_TICKS

        # If the agent keeps the same activity at a valid current building,
        # keep the same destination building instead of re-sampling randomly.
        continue_same_location = (
            self.ttravel == 0
            and target_activity == self.activity_type
            and self.bfrom not in (-1, 0)
            and self.bfrom in self.model.dB
        )

        if continue_same_location:
            self.bto = self.bfrom
        else:
            dest_choice = self._route_llm_activity_destination(target_activity)
            if dest_choice is None:
                target_activity = self.activity_type
                self.bto = self.bfrom
                stay_ticks = LLM_DEFAULT_STAY_TICKS
            else:
                self.bto = int(dest_choice)

        current_tick = self.model.runner.schedule.tick
        leaving_current = not (
            target_activity == self.activity_type and self.bto == self.bfrom
        )
        if leaving_current:
            self._log_activity_period(current_tick)

        if not leaving_current:
            self.ttravel = 0
            self.tdwell = stay_ticks
            self.current_stay_ticks_total = stay_ticks
            self.pending_stay_ticks = None
        else:
            self.pending_stay_ticks = stay_ticks
            self._initiate_travel()

        self.activity_type = target_activity

    def logging_schedule_of_agents(self):
        day = int(self.model.runner.schedule.tick / 288)
        if (day > 0):
            list_of_variables = ("|").join(
                [
                    "%s" % data for data in [
                        day,
                        self.id,
                        self.schedule_size,
                        self.schedule
                    ]
                ]
            )
            self.model.file_schedules.write("%s\n" % list_of_variables)

    def adding_activity(self, btype):
        self.schedule_size += 1
        if self.schedule_size >= len(self.schedule):
            self.schedule = np.pad(
                self.schedule,
                (0, max(10, len(self.schedule))),
                mode="constant",
                constant_values=0,
            )
        self.schedule[self.schedule_size] = btype

    def save(self) -> Tuple:
        return (
            self.uid, self.pt.coordinates, self.schedule_size, self.schedule,
            self.buildings, self.bfrom, self.bto, self.ttravel, self.tdwell,
            self.work_need, self.food_need, self.social_need,
            self.errands_need, self.rest_need, self.activity_type,
            self.decision_policy,
            self.age, self.gender, self.llm_endpoint, self.agent_type,
            self.pending_stay_ticks, self.current_stay_ticks_total,
            self.activity_start_tick, self.total_travel_ticks,
            self.travel_origin, self.travel_destination,
            self.last_logged_position, self.current_position,
            self.today_day_index, self.today_segments,
            dict(self.agent_attrs),
            dict(self.day_plan_state),
            dict(self.day_plan_status),
        )


# --- Creating cache for the agents
agente_cache = {}
CURRENT_MODEL = None


# --- Creating function that restore the agent when it changes rank
def restore_agent(agent_data):
    uid = agent_data[0]
    if uid in agente_cache:
        agent = agente_cache[uid]
    else:
        pt = dpt(agent_data[1][0], agent_data[1][1], agent_data[1][2])
        agent = Worker(
            CURRENT_MODEL, uid[0], uid[2], pt,
            agent_data[2], agent_data[3],
            agent_data[4], agent_data[5], agent_data[6], agent_data[7],
            agent_data[8], agent_data[9], agent_data[10], agent_data[11],
            agent_data[12], agent_data[13], agent_data[14], agent_data[15],
            agent_data[16], agent_data[17], agent_data[18], agent_data[19]
        )
        agente_cache[uid] = agent

    agent.model = CURRENT_MODEL
    agent.pt = dpt(agent_data[1][0], agent_data[1][1], agent_data[1][2])
    agent.schedule_size = agent_data[2]
    agent.schedule = agent_data[3]
    agent.buildings = agent_data[4]
    agent.bfrom = agent_data[5]
    agent.bto = agent_data[6]
    agent.ttravel = agent_data[7]
    agent.tdwell = agent_data[8]
    agent.work_need = agent_data[9]
    agent.food_need = agent_data[10]
    agent.social_need = agent_data[11]
    agent.errands_need = agent_data[12]
    agent.rest_need = agent_data[13]
    agent.activity_type = agent_data[14]
    agent.decision_policy = agent_data[15]
    agent.age = agent_data[16]
    agent.gender = agent_data[17]
    agent.llm_endpoint = agent_data[18]
    agent.agent_type = parse_agent_type(
        agent_data[19],
        context=f"restore_agent(agent_id={uid[0]}) agent_type",
    )
    agent.pending_stay_ticks = agent_data[20]
    agent.current_stay_ticks_total = agent_data[21]
    agent.activity_start_tick = agent_data[22]
    agent.total_travel_ticks = agent_data[23]
    agent.travel_origin = (
        tuple(agent_data[24])
        if agent_data[24] is not None else agent.travel_origin
    )
    agent.travel_destination = (
        tuple(agent_data[25])
        if agent_data[25] is not None else agent.travel_destination
    )
    agent.last_logged_position = (
        tuple(agent_data[26]) if agent_data[26] is not None else None
    )
    agent.current_position = (
        tuple(agent_data[27])
        if agent_data[27] is not None else agent.current_position
    )
    if len(agent_data) > 28:
        agent.today_day_index = agent_data[28]
    if len(agent_data) > 29:
        agent.today_segments = agent_data[29]
    if len(agent_data) > 30:
        agent.agent_attrs = (
            dict(agent_data[30]) if agent_data[30] is not None else {}
        )
    elif not hasattr(agent, "agent_attrs"):
        agent.agent_attrs = {}
    if len(agent_data) > 31:
        agent.day_plan_state = (
            dict(agent_data[31]) if agent_data[31] is not None else {}
        )
    elif not hasattr(agent, "day_plan_state"):
        agent.day_plan_state = {}
    if len(agent_data) > 32:
        agent.day_plan_status = (
            dict(agent_data[32]) if agent_data[32] is not None else {}
        )
    elif not hasattr(agent, "day_plan_status"):
        agent.day_plan_status = {}
    return agent

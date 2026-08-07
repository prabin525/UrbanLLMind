"""
Core Model Logic for the Urban Agent Simulation.
Manages the simulation context, grid space, agent population,
and the main time-step loop (`step`). Handles MPI synchronization
and async LLM client batching.
"""
from typing import Dict, Optional
import asyncio
import threading
from datetime import (
    datetime,
    # timedelta
)
from mpi4py import MPI
import numpy as np
from repast4py import schedule, context as ctx, space, random
from repast4py.space import DiscretePoint as dpt, BorderType, OccupancyType
import pandas as pd
import os
import collections

from mini_world.agent import Worker, restore_agent
import mini_world.agent as agent_module
from mini_world.agent_types import (
    cohort_label_for_agent_type,
    validate_agent_type_series,
)
from mini_world.agent_attrs_loader import (
    default_agent_attrs,
    load_optional_agent_attrs,
)


class Model:
    """
    Repast4Py Model class encapsulating the simulation state.

    Attributes:
        comm: MPI communicator.
        rank: MPI rank.
        context: SharedContext for agents.
        grid: SharedGrid for spatial movement.
        agents: Local list of agents.
        runner: Schedule runner.
    """
    def __init__(self, comm: MPI.Intracomm, params: Dict):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.out_folder_name = params.get(
            "output_folder",
            "output_files_teste"
        )
        agent_module.CURRENT_MODEL = self
        # Ensure attribute exists before any async client setup.
        self.llm_endpoints = []

        # Persistent asyncio loop for async LLM batching
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop,
            daemon=True
        )
        self._loop_thread.start()
        self._async_llm_client = None

        self.runner = schedule.init_schedule_runner(self.comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop_at'])
        self.runner.schedule_end_event(self.at_end)

        self.input_data_folder = params.get("input_data_folder", "InputData")

        input_agents = pd.read_csv(
            os.path.join(self.input_data_folder, "input_agents.txt"),
            sep="\t"
        )
        input_buildings = pd.read_csv(
            os.path.join(self.input_data_folder, "input_buildings.txt"),
            sep="\t"
        )
        input_infra = pd.read_csv(
            os.path.join(self.input_data_folder, "input_agents_infra_zipf.csv")
        )
        attrs_result = load_optional_agent_attrs(
            self.input_data_folder,
            input_agents["agent_id"].tolist(),
        )
        self.agent_attrs_by_id = attrs_result.attrs_by_agent_id
        self.default_agent_attrs = default_agent_attrs()
        if self.rank == 0:
            print(attrs_result.summary_line())

        env_endpoints = os.getenv("VLLM_ENDPOINTS")
        if env_endpoints:
            endpoints = [
                ep.strip()
                for ep in env_endpoints.split(",")
                if ep.strip()
            ]
        else:
            env_llm_url = os.getenv("VLLM_URL")
            if env_llm_url:
                params = dict(params)
                params["llm_endpoint"] = env_llm_url
            endpoints = []

        # Always store an independent copy so the caller's
        # dict is never mutated.
        self.params = dict(params)
        self.create_activity_table = self.params.get(
            "create_activity_table",
            False
        )
        self.survey_start_date = datetime.strptime(
            self.params['survey_start_date'],
            '%Y/%m/%d'
        )
        if self.create_activity_table:
            filename = os.path.join(
                self.out_folder_name,
                f"activity_log_rank{self.rank}.csv"
            )
            self.activity_log_file = open(filename, "w")
            self.activity_log_file.write(
                "agent_id,date,location_type,arrival_time,departure_time\n"
            )
        else:
            self.activity_log_file = None
        if endpoints:
            self.llm_endpoints = endpoints
        else:
            default_endpoint = self.params.get("llm_endpoint")
            self.llm_endpoints = [default_endpoint] if default_endpoint else []
        if self.rank == 0:
            print("ENV VLLM_ENDPOINTS:", os.getenv("VLLM_ENDPOINTS"))
            print("Model using LLM endpoints:", self.llm_endpoints)

        # Create one async OpenAI client per endpoint on the loop thread
        self._async_llm_clients = asyncio.run_coroutine_threadsafe(
            self._create_async_llm_clients(),
            self._loop
        ).result()

        self.x_bbox_min = int(min(input_buildings['x_centroid'])) - 1
        self.x_bbox_max = int(max(input_buildings['x_centroid']))
        self.y_bbox_min = int(min(input_buildings['y_centroid'])) - 1
        self.y_bbox_max = int(max(input_buildings['y_centroid']))

        x_extent = (self.x_bbox_max - self.x_bbox_min) + 1
        y_extent = (self.y_bbox_max - self.y_bbox_min) + 1

        self.context = ctx.SharedContext(self.comm)
        box = space.BoundingBox(
            self.x_bbox_min, x_extent, self.y_bbox_min, y_extent, 0, 0
        )
        self.grid = space.SharedGrid(
            'grid', bounds=box, borders=BorderType.Sticky,
            occupancy=OccupancyType.Multiple, buffer_size=0, comm=self.comm
        )
        self.context.add_projection(self.grid)

        self.local_bounds = self.grid.get_local_bounds()

        self.loading_buildings(input_buildings, input_infra)

        random.init(params["random_seed"] + self.rank)
        self.rng = random.default_rng

        self.creating_agents_from_file(input_agents)

        try:
            self.agents = [i for i in self.context.agents(Worker.TYPE)]
        except Exception:
            self.agents = []

        if (self.params["print_schedule"]):
            self.creating_output_files_schedule()

        self.file_activity = open(
            "%s%s%s" % (
                self.out_folder_name,
                os.sep,
                "agent_activities_rank%s.csv" % self.rank
            ),
            "w"
        )
        self.file_activity.write("tick,agent_id,building_id,activity\n")

        # Position logging disabled to mirror MinimalModel_v4 arrival-only logs

    def creating_output_files_schedule(self):
        self.file_schedules = open(
            os.path.join(
                self.out_folder_name, f"tabular_schedule_rank{self.rank}.txt"
            ),
            "w"
        )
        list_of_variables = ("|").join(
            ['day', 'agent_id', "schedule_size", "schedule_places"]
        )
        self.file_schedules.write(f"{list_of_variables}\n")

    def local_time(self):
        return self.runner.schedule.tick % 288

    def _cohort_label(self, agent_type: int) -> str:
        return cohort_label_for_agent_type(
            agent_type,
            context="Model._cohort_label(agent_type)",
        )

    def _get_need_calibration(
        self,
        agent_type: int
    ) -> Optional[Dict[str, Dict[str, float]]]:
        cfg = self.params.get("need_calibration")
        if not isinstance(cfg, dict):
            return None
        cohort_cfg = cfg.get(self._cohort_label(agent_type))
        if not isinstance(cohort_cfg, dict):
            return None
        initial = cohort_cfg.get("initial")
        growth = cohort_cfg.get("growth_per_tick")
        if not isinstance(initial, dict) or not isinstance(growth, dict):
            return None
        jitter_cfg = cohort_cfg.get("jitter_std_fraction", {})
        if not isinstance(jitter_cfg, dict):
            jitter_cfg = {}
        jitter_init = jitter_cfg.get("initial", {})
        if not isinstance(jitter_init, dict):
            jitter_init = {}
        jitter_growth = jitter_cfg.get("growth_per_tick", {})
        if not isinstance(jitter_growth, dict):
            jitter_growth = {}

        parsed_initial: Dict[str, float] = {}
        parsed_growth: Dict[str, float] = {}
        parsed_jitter_initial: Dict[str, float] = {}
        parsed_jitter_growth: Dict[str, float] = {}
        for key in ("work", "food", "social", "errands", "rest"):
            try:
                parsed_initial[key] = float(initial[key])
                parsed_growth[key] = float(growth[key])
                parsed_jitter_initial[key] = max(
                    0.0, float(jitter_init.get(key, 0.0))
                )
                parsed_jitter_growth[key] = max(
                    0.0, float(jitter_growth.get(key, 0.0))
                )
            except Exception:
                return None
        return {
            "initial": parsed_initial,
            "growth_per_tick": parsed_growth,
            "jitter_std_fraction": {
                "initial": parsed_jitter_initial,
                "growth_per_tick": parsed_jitter_growth,
            },
        }

    def _sample_with_relative_jitter(
        self,
        mean_value: float,
        jitter_fraction: float,
        min_value: float,
    ) -> float:
        if jitter_fraction <= 0:
            return max(min_value, mean_value)
        std = abs(mean_value) * jitter_fraction
        sampled = float(self.rng.normal(mean_value, std))
        return max(min_value, sampled)

    def creating_agents_from_file(self, agents):
        has_decision_policy_col = "decision_policy" in agents.columns
        if "agent_type" not in agents.columns:
            raise ValueError(
                "input_agents.txt is missing required column 'agent_type'."
            )
        agent_types = validate_agent_type_series(
            agents["agent_type"],
            context="input_agents.txt agent_type",
            allow_na=False,
        )
        if self.rank == 0:
            for i in range(len(agents)):
                schedule_size = 0
                schedule = np.zeros(20, dtype=int)
                buildings = np.zeros(4, dtype=int)
                bfrom = agents["home"][i]
                bto = -1
                ttravel = -1
                tdwell = int((7 * 60) / 5)
                work_need = np.zeros(2, dtype=float)
                food_need = np.zeros(2, dtype=float)
                social_need = np.zeros(2, dtype=float)
                errands_need = np.zeros(2, dtype=float)
                rest_need = np.zeros(2, dtype=float)
                activity_type = 1

                x = self.dB[agents["home"][i]]["x"]
                y = self.dB[agents["home"][i]]["y"]

                buildings[1] = agents["home"][i]
                buildings[2] = agents["work"][i]
                buildings[3] = agents["school"][i]

                agent_type = int(agent_types.iloc[i])
                need_calibration = self._get_need_calibration(agent_type)
                if need_calibration is not None:
                    init = need_calibration["initial"]
                    growth = need_calibration["growth_per_tick"]
                    jitter = need_calibration.get("jitter_std_fraction", {})
                    jitter_init = jitter.get("initial", {})
                    jitter_growth = jitter.get("growth_per_tick", {})

                    need_vars = {
                        "work": work_need, "food": food_need,
                        "social": social_need, "errands": errands_need,
                        "rest": rest_need,
                    }
                    for key, need_arr in need_vars.items():
                        need_arr[0] = self._sample_with_relative_jitter(
                            init[key], jitter_init.get(key, 0.0), 1e-6
                        )
                        need_arr[1] = self._sample_with_relative_jitter(
                            growth[key], jitter_growth.get(key, 0.0), 1e-9
                        )
                else:
                    base_growth = 1.0 / 287.0
                    # Jitter factors
                    primary_factor = 1.0
                    food_factor = self.rng.uniform(0.00, 0.7)
                    social_factor = self.rng.uniform(0.00, 0.7)
                    errands_factor = self.rng.uniform(0.00, 0.7)
                    rest_factor = self.rng.uniform(0.9, 1.2)

                    # For students (agent_type 2), reuse
                    # work_need as school_need
                    if agent_type == 2:
                        work_need[0] = 1.1 * primary_factor
                        work_need[1] = base_growth * primary_factor
                    else:
                        work_need[0] = 1.1 * primary_factor
                        work_need[1] = base_growth * primary_factor

                    food_need[0] = 1.1 * food_factor
                    food_need[1] = base_growth * food_factor
                    social_need[0] = 1.1 * social_factor
                    social_need[1] = base_growth * social_factor
                    errands_need[0] = 1.1 * errands_factor
                    errands_need[1] = base_growth * errands_factor
                    rest_need[0] = 1.1 * rest_factor
                    rest_need[1] = base_growth * rest_factor

                endpoint = None
                if self.llm_endpoints:
                    endpoint = self.llm_endpoints[
                        i % len(self.llm_endpoints)
                    ]
                agent_id = int(agents["agent_id"][i])
                agent_attrs = self.agent_attrs_by_id.get(agent_id)
                if agent_attrs is None:
                    agent_attrs = dict(self.default_agent_attrs)
                else:
                    agent_attrs = dict(agent_attrs)
                decision_policy = "memory_stream_llm"
                if has_decision_policy_col:
                    raw_policy = agents["decision_policy"][i]
                    if pd.notna(raw_policy):
                        parsed_policy = str(raw_policy).strip()
                        if parsed_policy:
                            decision_policy = parsed_policy

                agente = Worker(
                    self, agent_id, self.rank, dpt(x, y, 0),
                    schedule_size, schedule, buildings, bfrom, bto,
                    ttravel, tdwell, work_need, food_need, social_need,
                    errands_need, rest_need, activity_type,
                    decision_policy, agents["age"][i],
                    agents["gender"][i], endpoint, agent_type,
                    agent_attrs
                )

                self.context.add(agente)
                self.grid.move(agente, dpt(x, y, 0))

        self.context.synchronize(restore_agent)

    def loading_buildings(self, buildings, infra):
        self.dB = {}
        for i in range(len(buildings)):
            self.dB[buildings["building_id"][i]] = {
                "x": int(buildings["x_centroid"][i]),
                "y": int(buildings["y_centroid"][i]),
                "btype": buildings["building_type"][i]
            }

        self.dInfra = collections.defaultdict(
            lambda: collections.defaultdict(lambda: [])
        )
        for i in range(len(infra)):
            self.dInfra[
                infra["agent_id"][i]
            ][infra["building_type"][i]].append(infra["building_id"][i])

    def step(self):
        llm_decision_agents = []
        reflection_agents = []
        for agent in self.context.agents():
            if agent.decision_policy == "memory_stream_llm":
                needs_decision = agent.walk(allow_decision=False)
                if needs_decision:
                    llm_decision_agents.append(agent)
                if (
                    agent.memory_model is not None
                    and getattr(
                        agent.memory_model,
                        "pending_reflection_force",
                        False
                    )
                ):
                    reflection_agents.append(agent)
            else:
                agent.walk()

        if reflection_agents:
            future = asyncio.run_coroutine_threadsafe(
                self._gather_reflections(reflection_agents),
                self._loop
            )
            future.result()

        if llm_decision_agents:
            future = asyncio.run_coroutine_threadsafe(
                self._gather_day_plans(llm_decision_agents),
                self._loop
            )
            future.result()
            future = asyncio.run_coroutine_threadsafe(
                self._gather_llm_decisions(llm_decision_agents),
                self._loop
            )
            decisions = future.result()
            for agent, decision in decisions:
                next_activity_type, stay_minutes = decision
                agent._apply_llm_decision(next_activity_type, stay_minutes)
            for agent in llm_decision_agents:
                agent.post_decision_updates()

    async def _gather_reflections(self, agents):
        tasks = []
        for agent in agents:
            tasks.append(agent.memory_model.run_pending_reflection_async())
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    def _llm_client_for_agent(self, agent):
        endpoint = agent.llm_endpoint or self.params.get("llm_endpoint")
        llm_client = None
        if endpoint and self._async_llm_clients:
            llm_client = self._async_llm_clients.get(endpoint)
            if llm_client is None and self._async_llm_clients:
                # Fallback to any available client if endpoint key missing
                llm_client = next(iter(self._async_llm_clients.values()))
        return llm_client

    async def _gather_day_plans(self, agents):
        tasks = []
        for agent in agents:
            llm_client = self._llm_client_for_agent(agent)
            tasks.append(agent.ensure_day_plan_async(llm_client))
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _gather_llm_decisions(self, agents):
        tasks = []
        for agent in agents:
            llm_client = self._llm_client_for_agent(agent)
            tasks.append(agent.get_memory_stream_decision_async(llm_client))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        decisions = []
        for agent, result in zip(agents, results):
            if isinstance(result, Exception):
                decisions.append((agent, (1, 15)))
            else:
                decisions.append((agent, result))
        return decisions

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _create_async_llm_clients(self):
        import openai

        endpoints = list(self.llm_endpoints) if self.llm_endpoints else []
        if not endpoints:
            fallback = self.params.get("llm_endpoint")
            if fallback:
                endpoints = [fallback]
        clients = {}
        for endpoint in endpoints:
            clients[endpoint] = openai.AsyncOpenAI(
                base_url=endpoint,
                api_key="sk-no-key-required"
            )
        return clients

    def at_end(self):
        self.comm.Barrier()
        if self.create_activity_table and self.activity_log_file is not None:
            end_tick = self.runner.schedule.tick
            for agent in self.context.agents():
                agent._log_activity_period(end_tick)
            self.activity_log_file.close()
        if hasattr(self, 'file_schedules') and self.file_schedules:
            self.file_schedules.close()
        if hasattr(self, 'file_activity') and self.file_activity:
            self.file_activity.close()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=5)
            try:
                self._loop.close()
            except RuntimeError:
                pass

    def start(self):
        self.comm.Barrier()
        self.runner.execute()

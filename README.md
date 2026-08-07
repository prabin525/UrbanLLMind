# UrbanLLMind

UrbanLLMind is a generative agent-based simulation framework for urban
mobility. It combines a synthetic city and population, profile-grounded LLM
agents, persistent memory, and distributed open-weight inference to simulate
multi-day mobility behavior and responses to scenario interventions.

The simulation is implemented with Repast4Py and MPI. At each decision point,
an agent uses its profile, current context, daily plan, and retrieved memories
to select its next destination-purpose category and intended stay duration.
The simulation engine validates that decision, resolves it to a physical
location, executes travel and dwelling, and records the resulting activity.

## Repository Structure

- `src/mini_world/`: core simulation package
- `src/mini_world/prompts/`: LLM prompt templates
- `analysis/`: evaluation scripts and notebooks
- `scripts/`: input preparation and prompt-inspection utilities
- `training_data_generator/`: survey-grounded supervision generation
- `train/`: adaptation and inference configurations
- `slurm/`: distributed simulation and model-serving templates
- `input_GABM*.yaml`: simulation and ablation configurations

## Core Runtime

- `src/mini_world/main.py`: command-line entrypoint and model initialization
- `src/mini_world/model.py`: simulation lifecycle, MPI coordination, and LLM
  request batching
- `src/mini_world/agent.py`: agent state, travel, dwell execution, and activity
  logging
- `src/mini_world/memory_stream_async.py`: daily planning, step decisions,
  reflection, memory retrieval, and asynchronous LLM interaction
- `src/mini_world/day_planner.py`: day-planning utilities
- `src/mini_world/activity_taxonomy.py`: activity vocabulary and mappings
- `src/mini_world/prompts/memory_stream.yaml`: active memory-stream prompts

## Decision Stack

UrbanLLMind models behavior through three connected stages:

1. **Daily planning:** each agent forms a rough plan grounded in its profile,
   role, and current context.
2. **Step-level decisions:** when a decision is required, the agent selects its
   next destination-purpose category and intended stay duration.
3. **Reflection and memory:** daily reflections and prior observations are
   stored and retrieved to support longitudinal behavioral continuity.

LLM calls are selective and batched. Only agents requiring a planning,
decision, or reflection operation at the current simulation tick are sent to
the configured OpenAI-compatible inference endpoints.

## Framework Capabilities

- profile-grounded synthetic urban mobility simulation
- daily planning and step-level activity decisions
- persistent memory retrieval and end-of-day reflection
- multi-day behavioral and persona-continuity analysis
- scenario interventions such as severe weather and travel disruption
- distributed simulation with MPI and distributed open-weight inference
- survey-grounded supervision generation and model adaptation
- component ablations for planning, memory, and reflection

## Data Requirements

Simulation inputs and large datasets are maintained separately from the source
repository. Depending on the workflow, UrbanLLMind expects:

- synthetic agent and population attributes
- building and destination-infrastructure records
- agent-to-home, work, and education assignments
- NHTS travel-survey files for validation and adaptation workflows
- optional generated supervision and model checkpoints

Simulation configurations reference these files through `input_data_folder`
and related paths. Local data is conventionally placed under `Inputs/` and
`dataset/`; these directories are excluded from version control because the
underlying datasets and generated artifacts can be large.

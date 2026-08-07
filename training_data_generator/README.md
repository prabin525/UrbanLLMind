# Async NHTS Training Data Generator

This module builds supervised fine-tuning data for the simulator from NHTS 2017 person-days.

Each sampled NHTS person-day is replayed through the simulator prompt contract and expanded into:

- `1` day-planner example
- `B` decision examples, where `B` is the number of dwell blocks in the reconstructed day
- `1` reflection example

The output is a GPT-OSS-style JSONL file where each row contains `messages` plus metadata for one prompt call.

## What This Generator Does

The generator uses three layers:

1. NHTS as the behavioral source of truth.
   - `perpub.csv` provides the person-day and demographic context.
   - `trippub.csv` provides the observed trip chain used to reconstruct the day.

2. The simulator prompt contract as the visible training input.
   - The shared system prompt and user prompts are rendered to match the current simulator prompt style.
   - The visible assistant `content` is kept runtime-compatible.

3. A teacher LLM as the explanation layer.
   - The teacher sees the runtime prompts plus hidden gold constraints.
   - It returns:
     - hidden `thinking`
     - visible `content`

The hidden `thinking` is included because the dataset is intended for GPT-OSS-style reasoning SFT. The simulator itself does not consume this field at runtime.

## Age Restriction

The generator supports an optional hard minimum-age filter at sampling time.

By default it is off:

- `ENABLE_MIN_AGE_FILTER = False`

When enabled, it uses:

- `MIN_AGE = 16`

When `ENABLE_MIN_AGE_FILTER = True`, person-days for respondents younger than 16 are excluded before replay and teacher generation.

The filter uses the same age derivation as the prompt profile:

- prefer `R_AGE_IMP`
- fallback to `R_AGE`

There is also a defensive check during sample construction, so under-16 person-days are rejected even if the loader filter is bypassed later while the flag is enabled.

## End-to-End Flow

For one sampled person-day, the pipeline is:

1. Load one NHTS person-day from `perpub.csv`.
2. Join all matching `trippub.csv` rows by:
   - `HOUSEID`
   - `PERSONID`
   - `TDAYDATE`
   - `TRAVDAY`
3. Derive the agent profile:
   - age
   - gender
   - role
   - household/context attributes
   - CBSA and prompt location
4. Reconstruct dwell blocks from the trip chain.
   - Example: `Home -> Work -> Recreational -> Home`
   - Person-days with no matching trips are excluded before replay.
5. Render the same prompt sequence the simulator uses:
   - shared system prompt
   - day planner
   - stepwise decisions
   - end-of-day reflection
6. Replay the day chronologically.
   - The planner is generated first.
   - The generated planner text is carried into later decision prompts.
   - Observation and decision memories are generated deterministically and carried into later prompts.
7. Teacher-force the output.
   - Planner: guided by the gold day outline.
   - Decision: constrained to the gold `next_activity_type` and `stay_minutes`.
   - Reflection: generated from the day’s accumulated memories.
8. Validate each output.
   - Planner must be a short paragraph.
   - Decision must be valid JSON and match the gold labels exactly.
   - Reflection must be one compact line.
9. Export one JSONL row per prompt call.

## Files

- `/Users/prb977/Project/MMv4/training_data_generator/config.py`
  - top-level configuration
- `/Users/prb977/Project/MMv4/training_data_generator/main.py`
  - async orchestration and progress reporting
- `/Users/prb977/Project/MMv4/training_data_generator/nhts_loader.py`
  - NHTS loading, sampling, date synthesis, dwell-block reconstruction
- `/Users/prb977/Project/MMv4/training_data_generator/replay_engine.py`
  - day replay and row assembly
- `/Users/prb977/Project/MMv4/training_data_generator/runtime_contract.py`
  - sim-style prompt and memory rendering
- `/Users/prb977/Project/MMv4/training_data_generator/teacher_client.py`
  - async OpenAI-compatible teacher client and validation
- `/Users/prb977/Project/MMv4/training_data_generator/writer.py`
  - JSONL output and error logging
- `/Users/prb977/Project/MMv4/training_data_generator/template.slurm`
  - simple cluster submission template

## Inputs

By default the generator reads:

- `/Users/prb977/Project/MMv4/dataset/NHTS_2017_csv/perpub.csv`
- `/Users/prb977/Project/MMv4/dataset/NHTS_2017_csv/trippub.csv`
- `/Users/prb977/Project/MMv4/dataset/NHTS_2017_csv/cities_info.csv`

## Prompt Location Name

Unless `LOCATION_NAME_OVERRIDE` is set, the prompt location name is derived from `CBSA_Title` in `cities_info.csv` as:

- primary city before the first `-`
- primary state abbreviation before the first `-` after the comma

Examples:

- `San Francisco-Oakland-Hayward, CA` -> `San Francisco, CA`
- `Los Angeles-Long Beach-Anaheim, CA` -> `Los Angeles, CA`
- `New York-Newark-Jersey City, NY-NJ-PA` -> `New York, NY`

## Day and Date Handling

NHTS `TRAVDAY` is treated as a weekday code, not a day-of-month:

- `1 = Sunday`
- `2 = Monday`
- `3 = Tuesday`
- `4 = Wednesday`
- `5 = Thursday`
- `6 = Friday`
- `7 = Saturday`

`TDAYDATE` is treated as a `YYYYMM` month key.

Because the public file does not provide a full exact date, the generator synthesizes a deterministic date within that month by selecting the first calendar date in `YYYYMM` that matches the `TRAVDAY` weekday code.

Examples:

- `TDAYDATE=201610`, `TRAVDAY=1` becomes the first Sunday in October 2016
- `TDAYDATE=201611`, `TRAVDAY=1` becomes the first Sunday in November 2016

This keeps the weekday correct while still giving the prompts a deterministic date string.

## Activity Reconstruction

The generator maps NHTS trip purposes into the simulator’s 10 activity categories using the project’s current taxonomy mapping.

If a person-day has no matching trips in `trippub.csv`, it is excluded from sampling.

For sampled person-days that do have trips, the generator reconstructs dwell blocks from the observed trip chain and trip times.

### Zero-Trip Person-Days in the Current NHTS Files

Using the same person-day key and deduping logic as the loader:

- person-day key: `HOUSEID`, `PERSONID`, `TDAYDATE`, `TRAVDAY`
- `perpub.csv` is deduped to one row per key
- a zero-trip day is a `perpub` person-day with no matching `trippub` rows for that same key

On the current local NHTS files in this repo:

- nationwide raw deduped person-days: `264,234`
- nationwide excluded zero-trip person-days: `45,040`
  - share excluded: `17.05%`
- nationwide remaining sampleable person-days after dropping zero-trip days: `219,194`
- nationwide remaining sampleable person-days with `ENABLE_MIN_AGE_FILTER = True` and `MIN_AGE = 16`: `195,863`

- San Francisco CBSA `41860` raw deduped person-days: `4,688`
- San Francisco CBSA `41860` excluded zero-trip person-days: `661`
  - share excluded: `14.10%`
- San Francisco CBSA `41860` remaining sampleable person-days after dropping zero-trip days: `4,027`
- San Francisco CBSA `41860` remaining sampleable person-days with `ENABLE_MIN_AGE_FILTER = True` and `MIN_AGE = 16`: `3,603`

These zero-trip days are now dropped. The generator does not convert them into a full-day `Home` dwell block.

## Role Assignment

The generator assigns the top-level role from NHTS using a simple three-way rule:

- `worker` if `WORKER == 1`
- `student` if `WORKER != 1` and `SCHTYP in {1, 2, 3}`
- `homemaker` otherwise

In other words:

- worker status takes priority over school status
- any in-school non-worker becomes `student`
- everyone else becomes `homemaker`

This logic is implemented in `/Users/prb977/Project/MMv4/training_data_generator/role_and_attrs.py`.

### Role Counts in the Current Sampleable Pool

Using the same deduped person-day table the loader uses:

- person-day key: `HOUSEID`, `PERSONID`, `TDAYDATE`, `TRAVDAY`
- counts below are person-day counts, not unique-person counts
- zero-trip person-days are excluded from these counts because the generator drops them before replay

Nationwide:

- total sampleable person-days: `219,194`
- `worker`: `115,980` (`52.91%`)
- `student`: `26,340` (`12.02%`)
- `homemaker`: `76,874` (`35.07%`)
- with `ENABLE_MIN_AGE_FILTER = True` and `MIN_AGE = 16` total sampleable person-days: `195,863`
- with `ENABLE_MIN_AGE_FILTER = True` and `MIN_AGE = 16` `worker`: `115,971` (`59.21%`)
- with `ENABLE_MIN_AGE_FILTER = True` and `MIN_AGE = 16` `student`: `3,097` (`1.58%`)
- with `ENABLE_MIN_AGE_FILTER = True` and `MIN_AGE = 16` `homemaker`: `76,795` (`39.21%`)

San Francisco CBSA `41860`:

- total sampleable person-days: `4,027`
- `worker`: `2,428` (`60.29%`)
- `student`: `472` (`11.72%`)
- `homemaker`: `1,127` (`27.99%`)
- with `ENABLE_MIN_AGE_FILTER = True` and `MIN_AGE = 16` total sampleable person-days: `3,603`
- with `ENABLE_MIN_AGE_FILTER = True` and `MIN_AGE = 16` `worker`: `2,428` (`67.39%`)
- with `ENABLE_MIN_AGE_FILTER = True` and `MIN_AGE = 16` `student`: `50` (`1.39%`)
- with `ENABLE_MIN_AGE_FILTER = True` and `MIN_AGE = 16` `homemaker`: `1,125` (`31.22%`)

## Prompt and Memory Behavior

Within each sampled day:

- the planner always runs first
- planner output is stored as `day_plan_text`
- each decision prompt includes:
  - current context
  - the generated daily plan
  - today’s activity table
  - selected memories
- before each decision, the replay engine adds the same style of observation memory the sim uses
- after each decision, the replay engine adds the same style of decision memory the sim uses
- the reflection prompt is built from the same-day non-reflection memories

This means later prompts within the same day can refer back to what was already generated earlier in the replay.

## Output Format

The default output file is:

- `/Users/prb977/Project/MMv4/training_data_generator/outputs/nhts_gptoss_reasoning.jsonl`

The default error log is:

- `/Users/prb977/Project/MMv4/training_data_generator/outputs/nhts_gptoss_errors.jsonl`

Each JSONL row contains:

- `task_type`
- `split`
- `sample_index`
- `sample_day_id`
- `house_id`
- `person_id`
- `person_key`
- `cbsa_code`
- `cbsa_title`
- `prompt_location_name`
- `agent_role`
- `age`
- `gender`
- `step_index`
- `day_plan_text`
- `gold_next_activity_type` and `gold_stay_minutes` for decision rows
- `messages`

`messages` always has three turns:

1. `system`
2. `user`
3. `assistant`

The assistant turn contains:

- `content`: the visible sim-compatible response
- `thinking`: the hidden reasoning trace for GPT-OSS-style SFT

Important:

- user messages in the raw exported JSONL do not contain a `thinking` field
- assistant messages do contain `thinking`
- this is intentional for training export and does not exactly match the raw sim runtime artifact shape

## Async Execution Model

One sampled person-day is one async job.

The generator:

- creates one shared async OpenAI-compatible client
- queues all sampled person-days
- starts `ASYNC_AGENT_CONCURRENCY` workers
- processes each sampled day sequentially within a worker:
  - planner
  - decisions
  - reflection
- writes only complete successful days to the JSONL output
- writes failures to the error JSONL

The terminal progress line shows:

- completed agents
- active agents
- successful agents
- failed agents

## Running Locally

From the repo root:

```bash
python -m training_data_generator.main
```

The module reads configuration directly from `/Users/prb977/Project/MMv4/training_data_generator/config.py`.

## Running on Slurm

There is a minimal submission template at:

- `/Users/prb977/Project/MMv4/training_data_generator/template.slurm`

It runs:

```bash
python -m training_data_generator.main
```

Adjust the working directory, partition, memory, and task settings to match your cluster.

## Main Configuration Knobs

The main settings live near the top of `/Users/prb977/Project/MMv4/training_data_generator/config.py`.

Sampling:

- `N_PERSON_DAYS`
- `CBSA_FILTER`
- `LOCATION_NAME_OVERRIDE`
- `ENABLE_MIN_AGE_FILTER`
- `MIN_AGE`
- `RANDOM_SEED`

`N_PERSON_DAYS` controls how many valid sampleable person-days are generated.

- if it is an integer, the generator samples that many person-days
- if it is `None`, the generator uses all valid sampleable person-days after filters

At startup, the generator prints the resolved number of sampled person-days it is about to process.

Concurrency:

- `ASYNC_AGENT_CONCURRENCY`

I/O:

- `PERPUB_PATH`
- `TRIPPUB_PATH`
- `CITIES_INFO_PATH`
- `OUTPUT_JSONL_PATH`
- `ERROR_LOG_PATH`

Teacher generation:

- `OPENAI_API_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `MAX_COMPLETION_RETRIES`
- `OPENAI_TIMEOUT_SECONDS`
- `ENABLE_REQUEST_RATE_LIMITER`
- `MAX_REQUESTS_PER_MINUTE`
- `RATE_LIMIT_WINDOW_SECONDS`
- `DAY_PLANNER_TEMPERATURE`
- `DECISION_TEMPERATURE`
- `REFLECTION_TEMPERATURE`

The request rate limiter is global to the generator process. If enabled, every teacher request is paced through a shared limiter before being sent to the provider. This is useful for providers with per-minute quotas such as Gemini.

Dataset split:

- `TRAIN_RATIO`
- `VAL_RATIO`
- `TEST_RATIO`

## Changing the API and Model

To point the generator at a different OpenAI-compatible provider or model, edit these three constants in `/Users/prb977/Project/MMv4/training_data_generator/config.py`:

```python
OPENAI_API_BASE_URL = "..."
OPENAI_API_KEY = "..."
OPENAI_MODEL = "..."
```

### Example: OpenAI API

```python
OPENAI_API_BASE_URL = "https://api.openai.com/v1/"
OPENAI_API_KEY = "YOUR_OPENAI_KEY"
OPENAI_MODEL = "gpt-5-mini"
```

### Example: Gemini OpenAI-compatible endpoint

```python
OPENAI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENAI_API_KEY = "YOUR_GEMINI_KEY"
OPENAI_MODEL = "gemini-3.1-pro-preview"
```

### Example: Any other OpenAI-compatible endpoint

```python
OPENAI_API_BASE_URL = "https://your-provider.example.com/v1/"
OPENAI_API_KEY = "YOUR_PROVIDER_KEY"
OPENAI_MODEL = "your-model-name"
```

After editing those values, run the same command:

```bash
python -m training_data_generator.main
```

Nothing else in the generator needs to change as long as the endpoint supports the OpenAI chat-completions interface used by `/Users/prb977/Project/MMv4/training_data_generator/teacher_client.py`.

## Practical Notes

- If `CBSA_FILTER` is `"41860"`, sampling is restricted to San Francisco.
- If `CBSA_FILTER` is `None` or `"ALL"`, sampling is nationwide.
- If `LOCATION_NAME_OVERRIDE` is set, it replaces the city name used in prompts.
- If there are not enough valid person-days after filtering, the generator fails fast.
- If a teacher response cannot be validated after retries, that whole sampled day is skipped and logged to the error file.

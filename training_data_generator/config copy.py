from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# OpenAI-compatible endpoint configuration.
OPENAI_API_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
OPENAI_API_KEY = "AIz"
OPENAI_MODEL = "gemini-3.1-pro-preview"

# Sampling configuration.
N_PERSON_DAYS: int | None = None
# N_PERSON_DAYS: int | None = 10000
CBSA_FILTER: str | None = "41860"  # Set to None or "ALL" for nationwide.
# CBSA_FILTER: str | None = None  # Set to None or "ALL" for nationwide.
LOCATION_NAME_OVERRIDE: str | None = None
ENABLE_MIN_AGE_FILTER = False
MIN_AGE = 16
ASYNC_AGENT_CONCURRENCY = 100
RANDOM_SEED = 42

# I/O configuration.
PERPUB_PATH = REPO_ROOT / "dataset" / "NHTS_2017_csv" / "perpub.csv"
TRIPPUB_PATH = REPO_ROOT / "dataset" / "NHTS_2017_csv" / "trippub.csv"
CITIES_INFO_PATH = REPO_ROOT / "dataset" / "NHTS_2017_csv" / "cities_info.csv"
OUTPUT_JSONL_PATH = (
    REPO_ROOT
    / "training_data_generator"
    / "outputs"
    / "nhts_gptoss_reasoning_v3-sf.jsonl"
)
# OUTPUT_JSONL_PATH = (
#     REPO_ROOT
#     / "training_data_generator"
#     / "outputs"
#     / "nhts_gptoss_reasoning_v3-.jsonl"
# )
ERROR_LOG_PATH = (
    REPO_ROOT
    / "training_data_generator"
    / "outputs"
    / "nhts_gptoss_errors.jsonl"
)

# Dataset split configuration.
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

# Teacher request configuration.
MAX_COMPLETION_RETRIES = 3
OPENAI_TIMEOUT_SECONDS = 300.0
DAY_PLANNER_TEMPERATURE = 0.4
DECISION_TEMPERATURE = 0.3
REFLECTION_TEMPERATURE = 0.4

# Memory retrieval configuration. Mirrors the runtime defaults.
MAX_CONTEXT_MEMORIES = 6
RECENCY_WEIGHT = 1.5
IMPORTANCE_WEIGHT = 1.0
RELEVANCE_WEIGHT = 1.2


@dataclass(frozen=True)
class GeneratorConfig:
    openai_api_base_url: str | None
    openai_api_key: str
    openai_model: str
    n_person_days: int | None
    cbsa_filter: str | None
    location_name_override: str | None
    enable_min_age_filter: bool
    min_age: int
    async_agent_concurrency: int
    random_seed: int
    perpub_path: Path
    trippub_path: Path
    cities_info_path: Path
    output_jsonl_path: Path
    error_log_path: Path
    train_ratio: float
    val_ratio: float
    test_ratio: float
    max_completion_retries: int
    openai_timeout_seconds: float
    day_planner_temperature: float
    decision_temperature: float
    reflection_temperature: float
    max_context_memories: int
    recency_weight: float
    importance_weight: float
    relevance_weight: float

    def validate(self) -> None:
        if self.n_person_days is not None and self.n_person_days <= 0:
            raise ValueError("n_person_days must be positive")
        if self.async_agent_concurrency <= 0:
            raise ValueError("async_agent_concurrency must be positive")
        if self.min_age < 0:
            raise ValueError("min_age must be non-negative")
        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-9:
            raise ValueError(
                "train_ratio + val_ratio + test_ratio must equal 1.0"
            )
        if self.max_completion_retries <= 0:
            raise ValueError("max_completion_retries must be positive")


def get_config() -> GeneratorConfig:
    config = GeneratorConfig(
        openai_api_base_url=OPENAI_API_BASE_URL,
        openai_api_key=OPENAI_API_KEY,
        openai_model=OPENAI_MODEL,
        n_person_days=N_PERSON_DAYS,
        cbsa_filter=CBSA_FILTER,
        location_name_override=LOCATION_NAME_OVERRIDE,
        enable_min_age_filter=ENABLE_MIN_AGE_FILTER,
        min_age=MIN_AGE,
        async_agent_concurrency=ASYNC_AGENT_CONCURRENCY,
        random_seed=RANDOM_SEED,
        perpub_path=Path(PERPUB_PATH),
        trippub_path=Path(TRIPPUB_PATH),
        cities_info_path=Path(CITIES_INFO_PATH),
        output_jsonl_path=Path(OUTPUT_JSONL_PATH),
        error_log_path=Path(ERROR_LOG_PATH),
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        max_completion_retries=MAX_COMPLETION_RETRIES,
        openai_timeout_seconds=OPENAI_TIMEOUT_SECONDS,
        day_planner_temperature=DAY_PLANNER_TEMPERATURE,
        decision_temperature=DECISION_TEMPERATURE,
        reflection_temperature=REFLECTION_TEMPERATURE,
        max_context_memories=MAX_CONTEXT_MEMORIES,
        recency_weight=RECENCY_WEIGHT,
        importance_weight=IMPORTANCE_WEIGHT,
        relevance_weight=RELEVANCE_WEIGHT,
    )
    config.validate()
    return config

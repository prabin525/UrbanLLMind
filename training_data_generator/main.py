from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, List, Sequence

from .config import GeneratorConfig, get_config
from .nhts_loader import SampledPersonDay, load_sampled_person_days
from .replay_engine import replay_sample_day
from .teacher_client import TeacherClient, TeacherGenerationError
from .writer import error_writer_loop, writer_loop


def _render_progress_line(stats: Dict[str, Any]) -> str:
    total = max(int(stats.get("requested_person_days", 0)), 1)
    successful = int(stats.get("successful_person_days", 0))
    failed = int(stats.get("failed_person_days", 0))
    completed = successful + failed
    active = int(stats.get("active_person_days", 0))
    width = 20
    filled = min(width, int(width * completed / total))
    bar = "#" * filled + "-" * (width - filled)
    return (
        f"[{bar}] {completed}/{total} agents"
        f" | active {active}"
        f" | ok {successful}"
        f" | failed {failed}"
    )


def _emit_progress(stats: Dict[str, Any], *, finished: bool = False) -> None:
    line = _render_progress_line(stats)
    if sys.stdout.isatty():
        end = "\n" if finished else "\r"
        sys.stdout.write(line + end)
        sys.stdout.flush()
        return
    if finished:
        print(line)


async def generate_dataset(
    config: GeneratorConfig,
    *,
    samples: Sequence[SampledPersonDay] | None = None,
    teacher_client: TeacherClient | None = None,
) -> Dict[str, Any]:
    config.validate()
    sampled_days = list(samples) if samples is not None else load_sampled_person_days(config)
    if config.n_person_days is None:
        print(
            "Preparing "
            f"{len(sampled_days)} sampled person-days for generation "
            "(all available after filters)."
        )
    else:
        print(
            "Preparing "
            f"{len(sampled_days)} sampled person-days for generation."
        )
    owns_teacher_client = teacher_client is None
    client = teacher_client or TeacherClient(config)

    day_queue: asyncio.Queue[SampledPersonDay | None] = asyncio.Queue()
    writer_queue: asyncio.Queue[List[Dict[str, Any]] | None] = asyncio.Queue()
    error_queue: asyncio.Queue[List[Dict[str, Any]] | None] = asyncio.Queue()

    for sample in sampled_days:
        await day_queue.put(sample)
    for _ in range(config.async_agent_concurrency):
        await day_queue.put(None)

    stats = {
        "requested_person_days": len(sampled_days),
        "successful_person_days": 0,
        "failed_person_days": 0,
        "active_person_days": 0,
        "rows_written": 0,
        "error_log_path": str(config.error_log_path),
    }

    writer_task = asyncio.create_task(
        writer_loop(config.output_jsonl_path, writer_queue)
    )
    error_writer_task = asyncio.create_task(
        error_writer_loop(config.error_log_path, error_queue)
    )
    _emit_progress(stats)

    async def worker() -> None:
        while True:
            sample = await day_queue.get()
            try:
                if sample is None:
                    return
                stats["active_person_days"] += 1
                _emit_progress(stats)
                try:
                    rows = await replay_sample_day(
                        sample=sample,
                        config=config,
                        teacher_client=client,
                    )
                except Exception as exc:
                    stats["failed_person_days"] += 1
                    error_row = {
                        "sample_index": sample.sample_index,
                        "sample_day_id": sample.sample_day_id,
                        "house_id": sample.house_id,
                        "person_id": sample.person_id,
                        "person_key": sample.person_key,
                        "cbsa_code": sample.cbsa_code,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    if isinstance(exc, TeacherGenerationError):
                        error_row["task_type"] = exc.task_type
                        error_row["raw_outer_payload_text"] = exc.raw_outer_payload_text
                        error_row["raw_thinking"] = exc.raw_thinking
                        error_row["raw_content"] = exc.raw_content
                    await error_queue.put(
                        [
                            error_row
                        ]
                    )
                    continue
                stats["successful_person_days"] += 1
                stats["rows_written"] += len(rows)
                await writer_queue.put(rows)
            finally:
                if sample is not None:
                    stats["active_person_days"] = max(
                        0, int(stats["active_person_days"]) - 1
                    )
                    _emit_progress(stats)
                day_queue.task_done()

    workers = [
        asyncio.create_task(worker())
        for _ in range(config.async_agent_concurrency)
    ]
    await day_queue.join()
    await asyncio.gather(*workers)
    await writer_queue.put(None)
    await error_queue.put(None)
    await writer_queue.join()
    await error_queue.join()
    await writer_task
    await error_writer_task
    _emit_progress(stats, finished=True)

    if owns_teacher_client:
        await client.close()
    return stats


def run() -> Dict[str, Any]:
    config = get_config()
    return asyncio.run(generate_dataset(config))


if __name__ == "__main__":
    summary = run()
    print(summary)

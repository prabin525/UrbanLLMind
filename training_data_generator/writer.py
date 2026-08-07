from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence


async def _jsonl_writer_loop(
    output_path: Path,
    queue,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        while True:
            bundle = await queue.get()
            try:
                if bundle is None:
                    return
                for row in bundle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
            finally:
                queue.task_done()


async def writer_loop(
    output_path: Path,
    queue,
) -> None:
    await _jsonl_writer_loop(output_path, queue)


async def error_writer_loop(
    output_path: Path,
    queue,
) -> None:
    await _jsonl_writer_loop(output_path, queue)

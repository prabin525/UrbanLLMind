import os
import json
from typing import Any


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _log(payload: dict[str, Any]) -> None:
    if _rank() == 0:
        print(json.dumps(payload, indent=2), flush=True)



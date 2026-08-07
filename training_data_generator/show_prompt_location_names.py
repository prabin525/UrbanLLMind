from __future__ import annotations

import argparse

import pandas as pd

from .config import get_config
from .nhts_loader import _metro_title_to_location_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show derived prompt location names from cities_info.csv."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of rows to print. Use 0 to print all rows.",
    )
    args = parser.parse_args()

    config = get_config()
    frame = pd.read_csv(
        config.cities_info_path,
        usecols=["CBSA", "CBSA_Title"],
        low_memory=False,
    ).copy()
    frame["CBSA"] = frame["CBSA"].astype(str).str.strip()
    frame["CBSA_Title"] = frame["CBSA_Title"].astype(str).str.strip()
    frame["prompt_location_name"] = frame["CBSA_Title"].map(
        _metro_title_to_location_name
    )

    limit = None if args.limit == 0 else max(args.limit, 0)
    if limit is not None:
        frame = frame.head(limit)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()

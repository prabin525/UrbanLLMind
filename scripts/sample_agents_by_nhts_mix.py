#!/usr/bin/env python3
"""
Create a sampled simulation input folder whose agent_type mix matches a
processed NHTS reference distribution.

Design choices (intentional):
1. Uses processed data only (no CBSA/raw-trips logic here).
2. Samples without replacement (no repeated agents).
3. Never upsamples. If requested N or per-type quota exceeds available
   source agents, the script fails.
4. Keeps original agent_ids for sampled agents.
"""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


VALID_AGENT_TYPES = (1, 2, 3)
REQUIRED_AGENT_COLUMNS = ("agent_id", "agent_type")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample an input_agents folder to match processed NHTS agent_type "
            "ratios (types 1/2/3)."
        )
    )
    parser.add_argument(
        "--source-input-folder",
        default="Inputs/input_agents",
        help="Source folder with input_agents.txt and related input files.",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Output folder to write the sampled input set.",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        required=True,
        help="Number of agents to sample.",
    )
    parser.add_argument(
        "--reference-processed",
        default="dataset/NHTS_2017_csv/processed_data/sf",
        help=(
            "Processed NHTS pickle path used to compute type distribution. "
            "Expected to include an 'agent_type' column."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    return parser.parse_args()


def _require_columns(
    df: pd.DataFrame, required: Iterable[str], context: str
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns {missing}")


def _read_source_agents(source_folder: Path) -> pd.DataFrame:
    agent_path = source_folder / "input_agents.txt"
    if not agent_path.exists():
        raise FileNotFoundError(f"Missing source file: {agent_path}")
    agents = pd.read_csv(agent_path, sep="\t")
    _require_columns(agents, REQUIRED_AGENT_COLUMNS, str(agent_path))
    agents["agent_id"] = pd.to_numeric(
        agents["agent_id"], errors="raise"
    ).astype(int)
    agents["agent_type"] = pd.to_numeric(
        agents["agent_type"], errors="raise"
    ).astype(int)
    invalid = sorted(
        set(
            agents.loc[
                ~agents["agent_type"].isin(VALID_AGENT_TYPES), "agent_type"
            ]
        )
    )
    if invalid:
        raise ValueError(
            f"{agent_path}: invalid agent_type values {invalid}; "
            f"expected only {list(VALID_AGENT_TYPES)}."
        )
    if agents["agent_id"].duplicated().any():
        dupes = sorted(
            agents.loc[agents["agent_id"].duplicated(), "agent_id"]
            .unique()
            .tolist()
        )[:10]
        raise ValueError(
            f"{agent_path}: duplicate agent_id values (sample) {dupes}"
        )
    return agents


def _read_reference_mix(reference_path: Path) -> Dict[int, float]:
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Missing reference processed file: {reference_path}"
        )
    try:
        ref = pd.read_pickle(reference_path)
    except Exception as exc:
        script_hint = "scripts/process_nhts_original_data.py"
        raise RuntimeError(
            f"Failed to read processed reference pickle at {reference_path}. "
            f"Regenerate it with {script_hint} in the same "
            "Python environment and retry."
        ) from exc
    _require_columns(ref, ("agent_type",), str(reference_path))
    ref_types = (
        pd.to_numeric(ref["agent_type"], errors="coerce").dropna().astype(int)
    )
    counts = (
        ref_types[ref_types.isin(VALID_AGENT_TYPES)].value_counts().to_dict()
    )
    total = int(sum(counts.values()))
    if total <= 0:
        valid_types = list(VALID_AGENT_TYPES)
        raise ValueError(
            f"{reference_path}: no rows with agent_type in {valid_types}"
        )
    return {
        t: float(counts.get(t, 0)) / float(total) for t in VALID_AGENT_TYPES
    }


def _allocate_quotas(
    n_agents: int, proportions: Dict[int, float]
) -> Dict[int, int]:
    raw = {t: n_agents * proportions[t] for t in VALID_AGENT_TYPES}
    quotas = {t: int(math.floor(raw[t])) for t in VALID_AGENT_TYPES}
    remaining = n_agents - sum(quotas.values())
    if remaining > 0:
        order = sorted(
            VALID_AGENT_TYPES,
            key=lambda t: ((raw[t] - quotas[t]), proportions[t], -t),
            reverse=True,
        )
        for i in range(remaining):
            quotas[order[i % len(order)]] += 1
    return quotas


def _sample_agent_ids(
    source_agents: pd.DataFrame,
    quotas: Dict[int, int],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected_ids = []
    for agent_type in VALID_AGENT_TYPES:
        available = source_agents.loc[
            source_agents["agent_type"] == agent_type, "agent_id"
        ].to_numpy()
        needed = int(quotas.get(agent_type, 0))
        if needed > len(available):
            available_count = len(available)
            raise ValueError(
                "Cannot sample without replacement for "
                f"agent_type={agent_type}: "
                f"requested {needed}, available {available_count}."
            )
        if needed == 0:
            continue
        sampled = rng.choice(available, size=needed, replace=False)
        selected_ids.extend(sampled.tolist())
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError(
            "Sampling produced duplicate agent_ids, which is invalid."
        )
    if len(selected_ids) != int(sum(quotas.values())):
        raise RuntimeError("Sampled agent count mismatch with quota total.")
    return np.array(selected_ids, dtype=int)


def _write_filtered_agents(
    source_agents: pd.DataFrame,
    selected_ids: np.ndarray,
    out_folder: Path,
) -> pd.DataFrame:
    selected_set = set(selected_ids.tolist())
    sampled_agents = source_agents[
        source_agents["agent_id"].isin(selected_set)
    ].copy()
    sampled_agents = sampled_agents.sort_values("agent_id")
    if len(sampled_agents) != len(selected_ids):
        raise RuntimeError("Filtered sampled agent table size mismatch.")
    out_path = out_folder / "input_agents.txt"
    sampled_agents.to_csv(out_path, sep="\t", index=False)
    return sampled_agents


def _copy_and_filter_supporting_files(
    source_folder: Path,
    out_folder: Path,
    selected_ids: np.ndarray,
) -> None:
    selected_set = set(selected_ids.tolist())
    for path in source_folder.iterdir():
        if not path.is_file():
            continue
        if path.name == "input_agents.txt":
            continue

        out_path = out_folder / path.name
        if path.name == "input_agents_infra_zipf.csv":
            infra = pd.read_csv(path)
            _require_columns(infra, ("agent_id",), str(path))
            infra["agent_id"] = pd.to_numeric(
                infra["agent_id"], errors="raise"
            ).astype(int)
            infra_filtered = infra[infra["agent_id"].isin(selected_set)].copy()
            covered_ids = set(infra_filtered["agent_id"].unique().tolist())
            missing = sorted(selected_set - covered_ids)
            if missing:
                preview = missing[:10]
                raise ValueError(
                    f"{path}: missing infrastructure rows for "
                    "sampled agent_ids "
                    f"(sample) {preview}."
                )
            infra_filtered.to_csv(out_path, index=False)
            continue

        if path.name == "input_agent_attrs.csv":
            attrs = pd.read_csv(path)
            if "agent_id" in attrs.columns:
                attrs["agent_id"] = pd.to_numeric(
                    attrs["agent_id"], errors="coerce"
                ).astype("Int64")
                attrs = attrs[attrs["agent_id"].isin(selected_set)].copy()
            attrs.to_csv(out_path, index=False)
            continue

        shutil.copy2(path, out_path)


def _print_report(
    n_agents: int,
    source_agents: pd.DataFrame,
    reference_mix: Dict[int, float],
    quotas: Dict[int, int],
    sampled_agents: pd.DataFrame,
    output_folder: Path,
) -> None:
    source_counts = (
        source_agents["agent_type"].value_counts().sort_index().to_dict()
    )
    sampled_counts = (
        sampled_agents["agent_type"].value_counts().sort_index().to_dict()
    )
    print("Sampling complete.")
    print(f"Output folder: {output_folder}")
    print(f"Requested n_agents: {n_agents}")
    print("Reference proportions (types 1/2/3):")
    for t in VALID_AGENT_TYPES:
        print(f"  type {t}: {reference_mix[t]:.6f}")
    print("Allocated quotas:")
    for t in VALID_AGENT_TYPES:
        print(f"  type {t}: {quotas.get(t, 0)}")
    print("Source availability:")
    for t in VALID_AGENT_TYPES:
        print(f"  type {t}: {int(source_counts.get(t, 0))}")
    print("Sampled counts:")
    for t in VALID_AGENT_TYPES:
        print(f"  type {t}: {int(sampled_counts.get(t, 0))}")


def main() -> None:
    args = parse_args()
    n_agents = int(args.n_agents)
    if n_agents <= 0:
        raise ValueError("--n-agents must be > 0")

    source_folder = Path(args.source_input_folder)
    out_folder = Path(args.output_folder)
    reference_path = Path(args.reference_processed)

    source_agents = _read_source_agents(source_folder)
    if n_agents > len(source_agents):
        source_count = len(source_agents)
        raise ValueError(
            f"Cannot sample {n_agents} agents from source size {source_count} "
            "without replacement."
        )
    reference_mix = _read_reference_mix(reference_path)
    quotas = _allocate_quotas(n_agents, reference_mix)
    if sum(quotas.values()) != n_agents:
        raise RuntimeError(
            "Quota allocation bug: quota sum != requested n_agents."
        )

    selected_ids = _sample_agent_ids(source_agents, quotas, args.seed)

    out_folder.mkdir(parents=True, exist_ok=True)
    sampled_agents = _write_filtered_agents(
        source_agents, selected_ids, out_folder
    )
    _copy_and_filter_supporting_files(source_folder, out_folder, selected_ids)
    _print_report(
        n_agents=n_agents,
        source_agents=source_agents,
        reference_mix=reference_mix,
        quotas=quotas,
        sampled_agents=sampled_agents,
        output_folder=out_folder,
    )


if __name__ == "__main__":
    main()

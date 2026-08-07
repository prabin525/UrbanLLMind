import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import ast
# import matplotlib.dates as mdates
# import matplotlib.patches as mpatches
from datetime import (
    datetime,
    # time,
    # timedelta
)
from pathlib import Path
import glob  # NOQA: F401
from typing import Any, Dict, Optional, Tuple
# from matplotlib.cm import get_cmap
# import ast
# import random
import scipy.cluster.hierarchy as sch

try:
    from constants import (
        NHTS_TO_SIM_ACTIVITY,
        NHTS_WHYTO_TO_NEW_ACTION,
        NEW_ACTION_TO_LEGACY6,
        NEW_ACTION_TO_POL,
        SIM_LOG_LABEL_TO_NEW_ACTION,
    )
except ImportError:  # pragma: no cover - fallback for module-style execution
    from analysis.constants import (
        NHTS_TO_SIM_ACTIVITY,
        NHTS_WHYTO_TO_NEW_ACTION,
        NEW_ACTION_TO_LEGACY6,
        NEW_ACTION_TO_POL,
        SIM_LOG_LABEL_TO_NEW_ACTION,
    )

try:
    from mini_world.agent_types import (
        AGENT_TYPE_BY_COHORT_LABEL,
        COHORT_LABEL_BY_TYPE,
        VALID_AGENT_TYPES,
        parse_agent_type,
        validate_agent_type_series,
    )
except ImportError:  # pragma: no cover - fallback for module-style execution
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from mini_world.agent_types import (  # type: ignore[import-not-found]
        AGENT_TYPE_BY_COHORT_LABEL,
        COHORT_LABEL_BY_TYPE,
        VALID_AGENT_TYPES,
        parse_agent_type,
        validate_agent_type_series,
    )


# PROJECT_DIR = '/Users/prb977/Project/travel_survey_llm'
PROJECT_DIR = '/Users/prb977/Project/MMv4'

COHORT_TO_AGENT_TYPE = {
    "worker": 1,
    "student": 2,
    "homemaker": 3,
}
AGENT_TYPE_TO_COHORT = {
    1: "worker",
    2: "student",
    3: "homemaker",
}

VALID_TARGET_TAXONOMIES = {"legacy6", "new10"}


def _validate_target_taxonomy(target_taxonomy: str) -> str:
    taxonomy = str(target_taxonomy or "legacy6").strip().lower()
    if taxonomy not in VALID_TARGET_TAXONOMIES:
        raise ValueError(
            f"Unsupported target_taxonomy '{target_taxonomy}'. "
            f"Expected one of {sorted(VALID_TARGET_TAXONOMIES)}."
        )
    return taxonomy


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _map_nhts_code_to_target(code: int, target_taxonomy: str) -> int:
    taxonomy = _validate_target_taxonomy(target_taxonomy)
    raw = _safe_int(code, -9)
    if taxonomy == "new10":
        return int(NHTS_WHYTO_TO_NEW_ACTION.get(raw, 9))
    return int(NHTS_TO_SIM_ACTIVITY.get(raw, 7))


def _map_nhts_sequence_to_target(
    seq_like: Any, target_taxonomy: str = "legacy6"
) -> list[int]:
    seq = _coerce_list(seq_like)
    out: list[int] = []
    taxonomy = _validate_target_taxonomy(target_taxonomy)
    for item in seq:
        out.append(_map_nhts_code_to_target(_safe_int(item, -9), taxonomy))
    return out


def _map_sim_log_label_to_new_action(label: Any) -> int:
    if label is None:
        return 9
    text = str(label).strip().lower()
    if not text or text in {"nan", "none"}:
        return 9
    return int(SIM_LOG_LABEL_TO_NEW_ACTION.get(text, 9))


def _map_new_action_to_target(
    code: int, target_taxonomy: str = "legacy6"
) -> int:
    taxonomy = _validate_target_taxonomy(target_taxonomy)
    action = _safe_int(code, 9)
    if taxonomy == "new10":
        return int(action)
    return int(NEW_ACTION_TO_LEGACY6.get(action, 7))


def _map_sim_log_label_to_target(
    label: Any, target_taxonomy: str = "legacy6"
) -> int:
    new_action = _map_sim_log_label_to_new_action(label)
    return _map_new_action_to_target(new_action, target_taxonomy)


def _map_pol_sequence_to_target(
    seq_like: Any, target_taxonomy: str = "legacy6"
) -> list[int]:
    taxonomy = _validate_target_taxonomy(target_taxonomy)
    seq = _coerce_int_sequence(seq_like)
    if taxonomy == "new10":
        raise ValueError(
            "POL is not supported for target_taxonomy='new10' in this phase"
        )
    out: list[int] = []
    for item in seq:
        out.append(int(NEW_ACTION_TO_POL.get(item, item)))
    return out


def travel_time(combined_time):
    travel_times = []
    travel_time = 0
    pet = None
    for each in combined_time:
        s, e = each.split('-')
        st = datetime.strptime(s, '%H:%M')
        et = datetime.strptime(e, '%H:%M')
        if pet is None:
            # pst = st
            pet = et
            continue
        else:
            travel_time += (st - pet).seconds
            travel_times.append((st - pet).seconds)
            # pst = st
            pet = et
    return travel_time


def travel_times(combined_time):
    travel_times = []
    travel_time = 0
    pet = None
    for each in combined_time:
        s, e = each.split('-')
        # e = e.strip('(next day))')
        st = datetime.strptime(s, '%H:%M')
        et = datetime.strptime(e, '%H:%M')
        if pet is None:
            # pst = st
            pet = et
            continue
        else:
            travel_time += (st - pet).seconds
            travel_times.append((st - pet).seconds)
            # pst = st
            pet = et
    return travel_times


def get_digraph(x, key='loc_type'):
    G = nx.DiGraph()
    nodes = x[key]
    G.add_nodes_from(nodes)
    for i in range(len(nodes) - 1):
        G.add_edge(nodes[i], nodes[i + 1])
    for i in range(len(x[key])-1):
        G.add_edge(x[key][i], x[key][i+1])
    return G


def get_single_day_values(x):
    time_values = x.tick
    location_values = x.building_type
    max_value = max(time_values)
    num_sublists = max_value // 288 + 1
    grouped_lists = [[] for _ in range(num_sublists)]

    for value in time_values:
        sublist_index = value // 288
        grouped_lists[sublist_index].append(value)

    grouped_location_lists = [[] for _ in range(num_sublists)]

    for i, value in enumerate(location_values):
        sublist_index = time_values[i] // 288
        grouped_location_lists[sublist_index].append(value)

    for i, each in enumerate(grouped_location_lists):
        if i == 0:
            continue
        each.insert(0, 'H')

    rand_val = np.random.randint(0, len(grouped_lists))
    return grouped_location_lists[rand_val]


def convert_to_numeric_loc_types(x):
    loc_types = []
    for each in x:
        if each == 'H':
            loc_types.append(1)
        elif each == 'W':
            loc_types.append(2)
        elif each == 'Res':
            loc_types.append(3)
        elif each == 'Sch':
            loc_types.append(4)
        elif each == 'Rec':
            loc_types.append(5)
        else:
            loc_types.append(7)
    return loc_types


# def get_new_loc_type_for_original_data(row, key='loc_type'):
#     mappings = {
#         1: 1,
#         2: 1,
#         3: 2,
#         4: 2,
#         5: 3,
#         6: 4,
#         7: 4,
#         8: 5,
#         9: 6,
#         10: 6,
#         11: 7,
#         12: 7,
#         13: 8,
#         14: 11,
#         15: 9,
#         16: 9,
#         17: 10,
#         18: 10,
#         19: 3,
#         97: 11,
#     }
#     final_vals = []
#     for each in row[key]:
#         if each in mappings.keys():
#             final_vals.append(mappings[each])
#         else:
#             final_vals.append(11)
#     return final_vals


def get_sim_loc_type_for_original_data(row, key='loc_type'):
    # Backward-compatible wrapper retained for legacy callers.
    return _map_nhts_sequence_to_target(row[key], target_taxonomy="legacy6")


def get_target_loc_type_for_original_data(
    row,
    key='loc_type',
    target_taxonomy: str = "legacy6",
):
    return _map_nhts_sequence_to_target(
        row[key],
        target_taxonomy=target_taxonomy
    )


def get_actual_survey_data(
        file_loc,
        # new_loc_type=True,
        sim_loc_type=True,
        target_taxonomy: str = "legacy6",
):
    target_taxonomy = _validate_target_taxonomy(target_taxonomy)
    data_t = pd.read_pickle(open(file_loc, 'rb'))
    # if new_loc_type is True:
    #     data_t['loc_type'] = data_t.apply(
    #         lambda row: get_new_loc_type_for_original_data(row, 'loc_type'),
    #         axis=1
    #     )
    # elif sim_loc_type is True:
    if sim_loc_type is True:
        data_t['loc_type'] = data_t.apply(
            lambda row: get_target_loc_type_for_original_data(
                row,
                'loc_type',
                target_taxonomy=target_taxonomy,
            ),
            axis=1
        )
    data_t['graph'] = data_t.apply(lambda x: get_digraph(x), axis=1)
    data_t['TDAYDATE'] = pd.to_datetime(data_t['TDAYDATE'], format='%Y%m')
    data_t = data_t.rename(columns={'TDAYDATE': 'survey_date'})
    values_count = data_t['loc_type'].value_counts()
    relative_frequencies = values_count/data_t.shape[0]
    rank_relative_freq_list = list(enumerate(relative_frequencies.items(), 1))
    ranks, relative_freqs = zip(
        *[
            (rank, relative_freq)
            for rank, (category, relative_freq) in rank_relative_freq_list
        ]
    )
    return data_t, ranks, relative_freqs


def get_actual_survey_data_info(
    file_loc,
    name,
    # new_loc_type=True,
    sim_loc_type=True,
    target_taxonomy: str = "legacy6",
):
    print(name)
    print('*'*50)
    data_t, ranks, relative_freqs = get_actual_survey_data(
        f'{PROJECT_DIR}/{file_loc}',
        # new_loc_type,
        sim_loc_type,
        target_taxonomy=target_taxonomy,
    )
    print(f"Average Location: {data_t['location'].mean()}")
    print(f"Average Location: {data_t['location'].median()}")
    print(f'Number of samples: {data_t.shape[0]}')
    print(f'Travel time (Hrs): {data_t.travel_time.mean()/(60*60)}')
    print('*'*50)
    return data_t, ranks, relative_freqs


def assign_proper_loc_type(row, target_taxonomy: str = "legacy6"):
    return _map_sim_log_label_to_target(
        row.get('location_type'),
        target_taxonomy=target_taxonomy,
    )


def get_generated_survey_data_no_preprocess(
        folder_loc,
        target_taxonomy: str = "legacy6",
):
    target_taxonomy = _validate_target_taxonomy(target_taxonomy)
    dfs = []
    for r in range(4):
        df = pd.read_csv(f"{folder_loc}/activity_log_rank{r}.csv")
        df["rank"] = r
        dfs.append(df)

    data_gen = pd.concat(dfs, ignore_index=True, sort=False)
    # data_gen = pd.read_csv(folder_loc+'/activity_log_rank0.csv')
    data_gen = data_gen[
        ~data_gen.agent_id.isin(
            data_gen.loc[data_gen.location_type.isna()].agent_id
        )
    ]
    data_gen['loc_type2'] = data_gen.apply(
        lambda row: assign_proper_loc_type(
            row,
            target_taxonomy=target_taxonomy,
        ),
        axis=1
    )

    t = data_gen.copy()
    t['combined_time'] = t["arrival_time"] + '-' + t["departure_time"]
    t1 = t.groupby(['agent_id'])['combined_time'].apply(list).reset_index()
    t2 = t.groupby(['agent_id'])['loc_type2'].apply(list).reset_index()
    t3 = t.groupby(['agent_id'])['date'].first().reset_index()
    # t4 = t.groupby(['agent_id'])['sex'].first().reset_index()
    # t5 = t.groupby(['agent_id'])['age'].first().reset_index()
    t = t1.merge(t2, on='agent_id', how='inner')
    t = t.merge(t3, on='agent_id', how='inner')
    # t = t.merge(t4, on='agent_id', how='inner')
    # t = t.merge(t5, on='agent_id', how='inner')
    t['date'] = pd.to_datetime(t['date'], format='%Y-%m-%d')
    t['travel_times'] = t.apply(
        lambda x: travel_times(x['combined_time']),
        axis=1
    )
    t['travel_time'] = t.apply(
        lambda x: travel_time(x['combined_time']),
        axis=1
    )
    t['graph'] = t.apply(lambda x: get_digraph(x, 'loc_type2'), axis=1)

    t['loc_type_new'] = t['loc_type2']

    t['loc_type'] = t['loc_type_new']
    t['location'] = t['loc_type'].apply(len)
    values_count = t['loc_type2'].value_counts()
    relative_frequencies = values_count/t.shape[0]
    rank_relative_freq_list = list(enumerate(relative_frequencies.items(), 1))
    ranks, relative_freqs = zip(
        *[
            (rank, relative_freq)
            for rank, (category, relative_freq) in rank_relative_freq_list
        ]
    )

    return t, ranks, relative_freqs


def _resolve_path(path_like):
    path = Path(path_like)
    if not path.is_absolute():
        path = Path(PROJECT_DIR) / path_like
    return path


def load_activity_log(folder_loc):
    folder_path = _resolve_path(folder_loc)
    files = sorted(folder_path.glob("activity_log_rank*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No activity_log_rank*.csv files found in {folder_path}"
        )
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True, sort=False)


def load_agent_attributes(agent_file):
    agent_path = _resolve_path(agent_file)
    return pd.read_csv(agent_path, sep="\t")


def attach_agent_attributes(activity_df, agent_file):
    attrs = load_agent_attributes(agent_file)
    return activity_df.merge(attrs, on="agent_id", how="left")


def _time_to_minutes(value):
    h, m = value.split(":")
    return int(h) * 60 + int(m)


def build_agent_day_sequences(
    activity_df,
    target_taxonomy: str = "legacy6",
):
    target_taxonomy = _validate_target_taxonomy(target_taxonomy)
    data_gen = activity_df.copy()
    data_gen = data_gen[
        ~data_gen.agent_id.isin(
            data_gen.loc[data_gen.location_type.isna()].agent_id
        )
    ]
    data_gen["loc_type2"] = data_gen.apply(
        lambda row: assign_proper_loc_type(
            row,
            target_taxonomy=target_taxonomy,
        ),
        axis=1
    )
    data_gen["date"] = pd.to_datetime(data_gen["date"], format="%Y-%m-%d")
    data_gen["arrival_minutes"] = data_gen["arrival_time"].apply(
        _time_to_minutes
    )
    data_gen = data_gen.sort_values(
        ["agent_id", "date", "arrival_minutes"]
    )
    data_gen["combined_time"] = (
        data_gen["arrival_time"] + "-" + data_gen["departure_time"]
    )
    grouped = data_gen.groupby(["agent_id", "date"])
    t1 = grouped["combined_time"].apply(list).reset_index()
    t2 = grouped["loc_type2"].apply(list).reset_index()
    t = t1.merge(t2, on=["agent_id", "date"], how="inner")
    if "agent_type" in data_gen.columns:
        t3 = grouped["agent_type"].first().reset_index()
        t = t.merge(t3, on=["agent_id", "date"], how="left")
    t["travel_times"] = t.apply(
        lambda x: travel_times(x["combined_time"]),
        axis=1
    )
    t["travel_time"] = t.apply(
        lambda x: travel_time(x["combined_time"]),
        axis=1
    )
    t["graph"] = t.apply(lambda x: get_digraph(x, "loc_type2"), axis=1)
    t["loc_type_new"] = t["loc_type2"]
    t["loc_type"] = t["loc_type_new"]
    t["location"] = t["loc_type"].apply(len)
    return t


def select_agent_days(t, mode="all", seed=42):
    if mode == "all":
        return t

    ordered = t.sort_values(["agent_id", "date"]).copy()
    grouped = ordered.groupby("agent_id", sort=False)

    # Use nth to preserve whole rows (groupby.first/last is column-wise).
    if mode == "first":
        return grouped.nth(0).reset_index(drop=True)
    if mode == "last":
        return grouped.nth(-1).reset_index(drop=True)
    if mode == "second":
        return grouped.nth(1).reset_index(drop=True)
    if mode == "third":
        return grouped.nth(2).reset_index(drop=True)
    if mode == "fourth":
        return grouped.nth(3).reset_index(drop=True)
    if mode == "sixth":
        return grouped.nth(5).reset_index(drop=True)
    if mode == "random":
        # Deterministic per-run randomness while still varying across agents.
        rng = np.random.default_rng(seed)
        picks = []
        for _, g in grouped:
            pick_idx = int(rng.integers(0, len(g)))
            picks.append(g.iloc[[pick_idx]])
        return pd.concat(picks, ignore_index=True)
    raise ValueError(f"Unknown mode: {mode}")


def get_generated_survey_data_info_multiday(
    folder_loc,
    name,
    day_selector="all",
    seed=42,
    print_info=False,
    target_taxonomy: str = "legacy6",
):
    target_taxonomy = _validate_target_taxonomy(target_taxonomy)
    activity_df = load_activity_log(folder_loc)
    t = build_agent_day_sequences(
        activity_df,
        target_taxonomy=target_taxonomy,
    )
    t = select_agent_days(t, mode=day_selector, seed=seed)
    values_count = t["loc_type2"].value_counts()
    relative_frequencies = values_count / t.shape[0]
    rank_relative_freq_list = list(enumerate(relative_frequencies.items(), 1))
    ranks, relative_freqs = zip(
        *[
            (rank, relative_freq)
            for rank, (category, relative_freq) in rank_relative_freq_list
        ]
    )
    if print_info:
        print(name)
        print("*" * 50)
        print(f"Average Location: {t.loc_type2.apply(len).mean()}")
        print(f"Median Location: {t.loc_type2.apply(len).median()}")
        print(f"Number of samples: {t.shape[0]}")
        print(f"Travel time: {t.travel_time.mean() / (60 * 60)}")
        print("*" * 50)
    return t, ranks, relative_freqs


def get_generated_survey_data_info(
    folder_loc,
    name,
    # new_loc_type=True,
    # sim_loc_type=False,
    print_info=False,
    target_taxonomy: str = "legacy6",
):
    t, ranks, relative_freqs = get_generated_survey_data_no_preprocess(
        f'{PROJECT_DIR}/{folder_loc}',
        # new_loc_type,
        # sim_loc_type
        target_taxonomy=target_taxonomy,
    )
    if print_info is True:
        print(name)
        print('*'*50)
        print(
            f"Average Location: {t.loc_type2.apply(lambda x: len(x)).mean()}"
        )
        print(
            f"Median Location: {t.loc_type2.apply(lambda x: len(x)).median()}"
        )
        print(f'Number of samples: {t.shape[0]}')
        print(f'Travel time: {t.travel_time.mean()/(60*60)}')
        print('*'*50)
    return t, ranks, relative_freqs


def get_transient_prob(t, loc_types_list, loc_type_key='loc_type'):
    # Respect the provided state codes directly (supports non-contiguous sets
    # like [1, 2, 3, 4, 5, 7]); do not assume 1..N indexing.
    state_codes = [int(code) for code in loc_types_list]
    transition_counts: dict[tuple[int, int], int] = {}

    for each in t[loc_type_key]:
        seq = _coerce_int_sequence(each)
        for i in range(len(seq) - 1):
            key = (seq[i], seq[i + 1])
            transition_counts[key] = transition_counts.get(key, 0) + 1

    main_lis = []
    for src in state_codes:
        sub_lis = []
        for dst in state_codes:
            sub_lis.append(transition_counts.get((src, dst), 0))
        main_lis.append(sub_lis)

    df = pd.DataFrame(main_lis, columns=state_codes, index=state_codes)
    return df


def load_pol_simulation(
    pol_file,
    seed=42,
    sample_one_day_per_agent=False,
    synthetic_start_date="2017-01-01",
):
    """Load POL trajectories as eval-ready agent-day rows.

    Default behavior returns one row per (agent_id, day) with a synthetic
    `date` derived from `tick // 288`. This enables day-based selectors
    (`first`, `second`, `last`, etc.) in evaluation notebooks.

    Legacy behavior can be restored by setting `sample_one_day_per_agent=True`,
    which samples one random day-chain per agent (the old behavior) and still
    emits a synthetic `date` for the sampled day.
    """
    pol_path = _resolve_path(pol_file)
    data_pol = pd.read_csv(pol_path)
    data_pol["tick"] = pd.to_numeric(data_pol["tick"], errors="coerce")
    data_pol["agent_id"] = pd.to_numeric(
        data_pol["agent_id"], errors="coerce"
    ).astype("Int64")
    data_pol = data_pol.dropna(subset=["tick", "agent_id"]).copy()
    data_pol["tick"] = data_pol["tick"].astype(int)
    data_pol["day_index"] = data_pol["tick"] // 288
    base_date = pd.Timestamp(synthetic_start_date)

    grouped = (
        data_pol.groupby(["agent_id", "day_index"], sort=True)["building_type"]
        .apply(list)
        .reset_index()
    )

    # Match legacy splitting behavior: after day 0,
    # prefix day chains with Home.
    mask = grouped["day_index"] > 0
    if mask.any():
        grouped.loc[mask, "building_type"] = grouped.loc[
            mask, "building_type"
        ].apply(lambda vals: ["H"] + list(vals))

    grouped = grouped.rename(columns={"building_type": "loc_type_cat"})
    grouped["date"] = base_date + pd.to_timedelta(
        grouped["day_index"], unit="D"
    )
    grouped["loc_type"] = grouped["loc_type_cat"].apply(
        lambda x: convert_to_numeric_loc_types(x)
    )
    grouped["location"] = grouped["loc_type"].apply(len)

    if sample_one_day_per_agent:
        if seed is not None:
            sampled = grouped.groupby("agent_id", group_keys=False).sample(
                n=1, random_state=seed
            )
        else:
            sampled = grouped.groupby("agent_id", group_keys=False).sample(n=1)
        grouped = sampled.sort_values(["agent_id"]).reset_index(drop=True)

    return grouped.drop(columns=["day_index"])


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return []
        try:
            parsed = ast.literal_eval(txt)
        except Exception:
            return []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
    return []


def _coerce_int_sequence(value: Any) -> list[int]:
    seq = _coerce_list(value)
    out: list[int] = []
    for item in seq:
        if isinstance(item, str):
            txt = item.strip()
            if txt in {"H", "Home"}:
                out.append(1)
                continue
            if txt in {"W", "Work"}:
                out.append(2)
                continue
            if txt in {"Res", "Restaurant"}:
                out.append(3)
                continue
            if txt in {"Sch", "School"}:
                out.append(4)
                continue
            if txt in {"Rec", "Recreation"}:
                out.append(5)
                continue
            if txt in {"O", "Other", "Errands"}:
                out.append(7)
                continue
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def _pick_first_existing_column(
    df: pd.DataFrame, candidates: list[str]
) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_agent_type_series(
    series: pd.Series,
    *,
    context: str = "agent_type",
    allow_na: bool = True,
    allow_unclassified_zero: bool = False,
) -> pd.Series:
    if not allow_unclassified_zero:
        return validate_agent_type_series(
            series,
            context=context,
            allow_na=allow_na,
        )

    numeric = pd.to_numeric(series, errors="coerce")
    if not allow_na and numeric.isna().any():
        bad_idx = numeric[numeric.isna()].index[:10].tolist()
        raise ValueError(
            f"{context}: contains non-numeric values at {bad_idx}"
        )
    valid_values = {0, *VALID_AGENT_TYPES}
    invalid_mask = numeric.notna() & ~numeric.isin(list(valid_values))
    if invalid_mask.any():
        invalid_vals = sorted(
            numeric[invalid_mask].astype(int).unique().tolist()
        )
        raise ValueError(
            f"{context}: invalid values {invalid_vals}; expected only "
            f"{sorted(valid_values)}"
        )
    return numeric.astype("Int64")


def _derive_person_class_from_agent_type(
    agent_type_series: pd.Series
) -> pd.Series:
    cohort = agent_type_series.map(COHORT_LABEL_BY_TYPE).astype(object)
    cohort.loc[agent_type_series == 0] = "unclassified"
    invalid_mask = agent_type_series.notna() & cohort.isna()
    if invalid_mask.any():
        bad_values = sorted(
            agent_type_series[invalid_mask].astype(int).unique().tolist()
        )
        raise ValueError(
            "Invalid agent_type values in cohort derivation: "
            f"{bad_values}. Expected {sorted({0, *VALID_AGENT_TYPES})}."
        )
    return cohort


def load_input_agent_type_map(input_folder: str) -> Optional[pd.DataFrame]:
    input_dir = _resolve_path(input_folder)
    path = input_dir / "input_agents.txt"
    if not path.exists():
        return None
    try:
        agents = pd.read_csv(path, sep="\t")
    except Exception:
        return None
    required = {"agent_id", "agent_type"}
    if not required.issubset(set(agents.columns)):
        return None
    out = agents[["agent_id", "agent_type"]].copy()
    out["agent_id"] = pd.to_numeric(
        out["agent_id"], errors="coerce"
    ).astype("Int64")
    out["agent_type"] = validate_agent_type_series(
        out["agent_type"],
        context=f"{path} agent_type",
        allow_na=False,
    )
    out["person_class"] = _derive_person_class_from_agent_type(
        out["agent_type"]
    )
    out = out.dropna(subset=["agent_id"]).drop_duplicates(subset=["agent_id"])
    return out


def attach_run_agent_types_from_input(
    run_df: pd.DataFrame,
    input_folder: str,
) -> Tuple[pd.DataFrame, bool, str]:
    agent_map = load_input_agent_type_map(input_folder)
    if agent_map is None:
        raise ValueError(
            f"Missing/invalid input_agents mapping under: {input_folder}"
        )

    out = run_df.copy()
    out["agent_id"] = pd.to_numeric(
        out["agent_id"], errors="coerce"
    ).astype("Int64")

    merged = out.merge(
        agent_map,
        on="agent_id",
        how="left",
        suffixes=("", "_from_input"),
    )
    if "agent_type_from_input" in merged.columns:
        existing = (
            validate_agent_type_series(
                merged["agent_type"],
                context="run_df.agent_type",
                allow_na=True,
            )
            if "agent_type" in merged.columns
            else pd.Series(pd.NA, index=merged.index, dtype="Int64")
        )
        mapped = validate_agent_type_series(
            merged["agent_type_from_input"],
            context=f"{input_folder} agent_type",
            allow_na=False,
        )
        merged["agent_type"] = existing.where(existing.notna(), mapped)
        merged = merged.drop(columns=["agent_type_from_input"])
    elif "agent_type" in merged.columns:
        merged["agent_type"] = validate_agent_type_series(
            merged["agent_type"],
            context="run_df.agent_type",
            allow_na=True,
        )

    if "person_class_from_input" in merged.columns:
        existing_pc = (
            merged["person_class"].astype(str)
            if "person_class" in merged.columns
            else pd.Series(np.nan, index=merged.index, dtype=object)
        )
        existing_pc = existing_pc.where(existing_pc.notna(), np.nan)
        merged["person_class"] = existing_pc.where(
            existing_pc.notna(), merged["person_class_from_input"]
        )
        merged = merged.drop(columns=["person_class_from_input"])

    return merged, True, str(input_folder)


def normalize_eval_dataframe(
    df: pd.DataFrame, source_kind: str = "simulation"
) -> pd.DataFrame:
    """Normalizes different trajectory tables into a common eval schema."""
    out = df.copy()
    print(
        f"Normalizing {source_kind} dataframe with columns: "
        f"{sorted(out.columns)}"
    )
    seq_col = _pick_first_existing_column(
        out,
        ["loc_type", "loc_type2", "loc_type_new"]
    )
    if seq_col is None:
        out["loc_type"] = [[] for _ in range(len(out))]
    else:
        out["loc_type"] = out[seq_col].apply(_coerce_int_sequence)

    if "location" not in out.columns:
        out["location"] = out["loc_type"].apply(len)
    else:
        out["location"] = pd.to_numeric(out["location"], errors="coerce")

    time_col = _pick_first_existing_column(out, ["combined_time"])
    if time_col is None:
        out["combined_time"] = [[] for _ in range(len(out))]
    else:
        out["combined_time"] = out[time_col].apply(_coerce_list)

    if "travel_time" not in out.columns:
        out["travel_time"] = np.nan
    else:
        out["travel_time"] = pd.to_numeric(out["travel_time"], errors="coerce")

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    elif "survey_date" in out.columns:
        out["date"] = pd.to_datetime(out["survey_date"], errors="coerce")
    else:
        out["date"] = pd.NaT

    if "agent_id" not in out.columns:
        out["agent_id"] = np.arange(len(out))
    out["agent_id"] = pd.to_numeric(
        out["agent_id"], errors="coerce"
    ).astype("Int64")

    if "agent_type" in out.columns:
        out["agent_type"] = _normalize_agent_type_series(
            out["agent_type"],
            context=f"{source_kind} agent_type",
            allow_na=False,
            allow_unclassified_zero=(source_kind == "actual"),
        )

    if "person_class" in out.columns:
        out["person_class"] = (
            out["person_class"].astype(str).str.strip().str.lower()
        )
        out.loc[
            out["person_class"].isin({"", "nan", "none"}), "person_class"
        ] = np.nan
    elif "agent_type" in out.columns:
        out["person_class"] = _derive_person_class_from_agent_type(
            out["agent_type"]
        )

    out["source_kind"] = source_kind
    return out


def filter_eval_frame_by_cohort(
    df: pd.DataFrame,
    cohort: str,
) -> Tuple[Optional[pd.DataFrame], str]:
    cohort_key = str(cohort).strip().lower()
    if cohort_key not in AGENT_TYPE_BY_COHORT_LABEL:
        return None, f"Unsupported cohort '{cohort}'."
    if "agent_type" not in df.columns:
        return None, "Missing agent_type column."
    target = int(parse_agent_type(
        AGENT_TYPE_BY_COHORT_LABEL[cohort_key],
        context=f"cohort={cohort_key}",
    ))
    agent_type = _normalize_agent_type_series(
        df["agent_type"],
        context="filter_eval_frame_by_cohort agent_type",
        allow_na=False,
        allow_unclassified_zero=True,
    )
    subset = df.loc[agent_type == target].copy()
    if subset.empty:
        return None, f"No rows available for cohort '{cohort_key}'."
    return subset, ""


def build_cohort_model_frames(
    model_frames: Dict[str, pd.DataFrame],
    cohort: str,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    out: Dict[str, pd.DataFrame] = {}
    skipped: Dict[str, str] = {}
    for model_name, model_df in model_frames.items():
        subset, reason = filter_eval_frame_by_cohort(model_df, cohort)
        if subset is None:
            skipped[model_name] = reason
            continue
        out[model_name] = subset
    return out, skipped


def load_actual_reference_frames(
    sf_file: str = "dataset/NHTS_2017_csv/processed_data/sf",
    all_file: str = "dataset/NHTS_2017_csv/processed_data/all",
    sim_loc_type: bool = True,
    target_taxonomy: str = "legacy6",
) -> Dict[str, pd.DataFrame]:
    """Loads and normalizes canonical reference frames used in evaluation."""
    sf_df, _, _ = get_actual_survey_data(
        str(_resolve_path(sf_file)),
        sim_loc_type=sim_loc_type,
        target_taxonomy=target_taxonomy,
    )
    all_df, _, _ = get_actual_survey_data(
        str(_resolve_path(all_file)),
        sim_loc_type=sim_loc_type,
        target_taxonomy=target_taxonomy,
    )
    return {
        "actual_sf": normalize_eval_dataframe(sf_df, source_kind="actual"),
        "actual_all": normalize_eval_dataframe(all_df, source_kind="actual"),
    }


def load_pol_simulation_normalized(
    pol_file: str = "dataset/simulation.csv",
    seed: Optional[int] = 42,
    target_taxonomy: str = "legacy6",
) -> pd.DataFrame:
    """Loads POL output and normalizes to the common eval schema."""
    target_taxonomy = _validate_target_taxonomy(target_taxonomy)
    if target_taxonomy == "new10":
        raise ValueError(
            "POL is not supported for target_taxonomy='new10' in this phase"
        )
    pol_df = load_pol_simulation(pol_file=pol_file, seed=seed)
    pol_df = pol_df.copy()
    pol_df["loc_type"] = pol_df["loc_type"].apply(
        lambda seq: _map_pol_sequence_to_target(
            seq,
            target_taxonomy=target_taxonomy,
        )
    )
    return normalize_eval_dataframe(pol_df, source_kind="pol")


def load_simulation_run_dict(
    sim_runs: Dict[str, str],
    day_selector: str = "all",
    seed: int = 42,
    run_input_folders: Optional[Dict[str, str]] = None,
    target_taxonomy: str = "legacy6",
) -> Dict[str, pd.DataFrame]:
    """Loads many simulation runs into a {name: normalized_df} mapping."""
    target_taxonomy = _validate_target_taxonomy(target_taxonomy)
    loaded: Dict[str, pd.DataFrame] = {}
    for run_name, run_dir in sim_runs.items():
        run_df, _, _ = get_generated_survey_data_info_multiday(
            folder_loc=run_dir,
            name=run_name,
            day_selector=day_selector,
            seed=seed,
            print_info=False,
            target_taxonomy=target_taxonomy,
        )
        if run_input_folders is not None and run_name in run_input_folders:
            run_df, _, reason = attach_run_agent_types_from_input(
                run_df=run_df,
                input_folder=run_input_folders[run_name],
            )
            print(f"[cohort-join] {run_name}: joined from {reason}")
        elif run_input_folders is not None:
            raise ValueError(
                f"[cohort-join] {run_name}: missing input folder mapping. "
                "Provide run_input_folders entry for every run when cohort "
                "analysis is enabled."
            )
        loaded[run_name] = normalize_eval_dataframe(
            run_df, source_kind="simulation"
        )
    return loaded


def add_temporal_columns(df):
    df = df.copy()
    df['day_of_week'] = df['date'].dt.day_name()
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = df['weekday'] >= 5
    return df


def compute_start_time_by_type(row, target_type):
    locs = row['loc_type2']
    times = row['combined_time']
    for loc, combined in zip(locs, times):
        if loc == target_type:
            start = combined.split('-')[0]
            return _time_to_minutes(start)
    return np.nan


def compute_weekday_consistency(df, target_type):
    df = df.sort_values('date')
    results = []
    for agent_id, g in df.groupby('agent_id'):
        g = g[g['start_min'].notna()]
        if g.empty:
            continue
        by_weekday = g.groupby('weekday')['start_min']
        if by_weekday.ngroups < 2:
            continue
        means = by_weekday.mean()
        results.append(means.std())
    return results


def compute_daytype_presence(df, start_col):
    weekday_total = max(1, df[~df["is_weekend"]].shape[0])
    weekend_total = max(1, df[df["is_weekend"]].shape[0])
    weekday_has = df[~df["is_weekend"] & df[start_col].notna()].shape[0]
    weekend_has = df[df["is_weekend"] & df[start_col].notna()].shape[0]
    return {
        "weekday_rate": weekday_has / weekday_total,
        "weekend_rate": weekend_has / weekend_total,
    }


def compute_consecutive_day_similarity(df):
    df = df.sort_values("date")
    sims = []
    for agent_id, g in df.groupby("agent_id"):
        sequences = list(g["loc_type2"])
        for i in range(len(sequences) - 1):
            sims.append(
                SequenceMatcher(None, sequences[i], sequences[i + 1]).ratio()
            )
    return sims


def compute_daytype_consistency(df, start_col, is_weekend):
    df = df[df["is_weekend"] == is_weekend]
    results = []
    for agent_id, g in df.groupby("agent_id"):
        g = g[g[start_col].notna()]
        if g.shape[0] < 2:
            continue
        results.append(g[start_col].std())
    return results


def compute_weekday_pair_similarity(df):
    df = df.sort_values("date")
    weekday_sims = []
    weekend_sims = []
    for agent_id, g in df.groupby("agent_id"):
        if g.shape[0] < 14:
            continue
        g = g.reset_index(drop=True)
        week1 = g.iloc[:7]
        week2 = g.iloc[7:14]
        for i in range(7):
            ratio = SequenceMatcher(
                None, week1.loc[i, "loc_type2"], week2.loc[i, "loc_type2"]
            ).ratio()
            if week1.loc[i, "weekday"] >= 5:
                weekend_sims.append(ratio)
            else:
                weekday_sims.append(ratio)
    return weekday_sims, weekend_sims


def summarize_start_time_stats(df, start_col, label):
    stats = df.groupby("is_weekend")[start_col].agg(["mean", "median", "std"])
    stats["label"] = label
    return stats.reset_index()


def compute_week_similarity(df):
    df = df.sort_values('date')
    sims = []
    for agent_id, g in df.groupby('agent_id'):
        sequences = list(g['loc_type2'])
        if len(sequences) < 14:
            continue
        week1 = sequences[:7]
        week2 = sequences[7:14]
        match_count = 0
        for a, b in zip(week1, week2):
            match_count += SequenceMatcher(None, a, b).ratio()
        sims.append(match_count / 7)
    return sims


def get_normalized_norm(M1, M2):
    return np.linalg.norm(
        (M1/M1.sum().sum()).to_numpy() - (M2/M2.sum().sum()).to_numpy()
    )


def plotting_the_dendogram(data, label_list, method="ward"):

    # Calculate the linkage matrix
    Z = sch.linkage(data, method=method, optimal_ordering=True)

    # Override the default linewidth.
    plt.rcParams['lines.linewidth'] = 2

    fig, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                           figsize=(4, 3))

    # --- Plotting the dendogram
    S = sch.dendrogram(
        Z,
        ax=ax,
        labels=label_list,
        orientation="right"
    )
    return S

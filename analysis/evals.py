import math
import pprint
from itertools import chain
from collections import defaultdict
import ast
import matplotlib.pyplot as plt
try:
    from pycirclize import Circos
except Exception:  # pragma: no cover - optional plotting dependency
    Circos = None
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, levene, wasserstein_distance
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from scipy.spatial.distance import jensenshannon
from constants import (
    NEW_LOC_TYPES,
    COMMON_LOC_TYPES,
    NHTS_LOC_TYPES,
    SIM_LOC_TYPES,
)
try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional plotting dependency
    sns = None
from matplotlib.patches import Patch
# from itertools import chain
# from collections import defaultdict
# from pycirclize import Circos

# NOTE: plotting / notebook helpers in this file are legacy-compatible and
# intentionally kept stable. Contract-based realism scoring entry points are
# exposed at the bottom via *_new wrappers.


def _require_circos() -> None:
    if Circos is None:
        raise ImportError(
            "pycirclize is required for chord diagram plotting. "
            "Install pycirclize to use these functions."
        )


# Pattern Level
def plot_loc_type_distribution(data, loc_type_labels, font_size=10):
    plt.rcParams.update({'font.size': font_size})

    keys = list(data.keys())
    original_key = keys[0]
    generated_keys = keys[1:]

    fig, ax = plt.subplots(figsize=(12, 7))

    original_counts = data[original_key]['loc_type'].explode().value_counts(
        normalize=True
    ).sort_index()
    normalized_data = [
        original_counts.reindex(loc_type_labels.keys(), fill_value=0)
    ]

    for gen_key in generated_keys:
        generated_counts = data[gen_key]['loc_type'].explode().value_counts(
            normalize=True
        ).sort_index()
        normalized_data.append(
            generated_counts.reindex(loc_type_labels.keys(), fill_value=0)
        )

    labels = list(loc_type_labels.values())
    x = np.arange(len(labels))
    bar_width = 0.8 / len(normalized_data)

    colors = plt.cm.get_cmap('tab10', len(normalized_data))

    for idx, counts in enumerate(normalized_data):
        bars = ax.bar(
            x + idx * bar_width,
            counts.values,
            width=bar_width,
            label=keys[idx],
            color=colors(idx)
        )
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=font_size * 0.5
                )

    ax.set_xticks(x + bar_width * (len(normalized_data) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Proportion", fontsize=font_size)
    ax.set_title(
        "Distribution of Location Types",
        fontsize=font_size + 2,
        weight='bold'
    )

    ax.legend(
        loc="upper right",
        fontsize=font_size * 0.9,
        frameon=True,
        fancybox=True,
        shadow=True
    )

    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_location_boxplot(data, font_size=10, showfliers=False):
    plt.rcParams.update({'font.size': font_size})

    keys = list(data.keys())
    combined_data = []
    combined_labels = []
    for key in keys:
        combined_data.extend(data[key]['location'])
        combined_labels.extend([key] * len(data[key]['location']))

    df = pd.DataFrame(
        {
            'Location Count': combined_data,
            'Type': combined_labels
        }
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    df.boxplot(
        column='Location Count',
        by='Type',
        ax=ax,
        vert=False,
        grid=False,
        showfliers=showfliers
    )
    ax.set_title(
        "Location Count Distribution",
        fontsize=font_size + 2,
        weight='bold'
    )
    ax.set_xlabel("Location Count", fontsize=font_size)
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylabel("")
    plt.suptitle("")
    plt.tight_layout()
    plt.show()


def plot_travel_time_boxplot(data, font_size=10, showfliers=False):
    plt.rcParams.update({'font.size': font_size})

    keys = [
        key for key in data.keys()
        if 'travel_time' in data[key].columns
    ]
    if not keys:
        print("No travel_time data available to plot.")
        return
    combined_data = []
    combined_labels = []
    for key in keys:
        combined_data.extend(data[key]['travel_time'] / 3600)
        combined_labels.extend([key] * len(data[key]['travel_time']))

    df = pd.DataFrame(
        {
            'Travel Time (Hours)': combined_data,
            'Type': combined_labels
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    df.boxplot(
        column='Travel Time (Hours)',
        by='Type',
        ax=ax,
        vert=False,
        grid=False,
        showfliers=showfliers
    )
    ax.set_title(
        "Travel Time Distribution(in Hours)",
        fontsize=font_size + 2,
        weight='bold'
    )
    ax.set_xlabel("Travel Time (Hours)", fontsize=font_size)
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylabel("")
    plt.suptitle("")
    plt.tight_layout()
    plt.show()


def plot_location_count_histograms(data, location_key, font_size=10):
    plt.rcParams.update({'font.size': font_size})

    keys = list(data.keys())
    original_key = keys[0]
    generated_keys = keys[1:]

    # Determine max location ID to set x-axis
    max_location = max(
        max(data[key][location_key].dropna().unique()) for key in data.keys()
    )
    x_values = np.arange(1, int(max_location) + 1)

    fig, axs = plt.subplots(
        len(generated_keys),
        1,
        figsize=(12, 3 * len(generated_keys)),
        sharex=True
    )

    # Ensure axs is iterable
    if len(generated_keys) == 1:
        axs = [axs]

    # Precompute original counts once
    original_counts = data[original_key][location_key].value_counts(
                        normalize=True
                    ).sort_index()
    original_counts = original_counts.reindex(x_values, fill_value=0)

    for idx, gen_key in enumerate(generated_keys):
        ax = axs[idx]

        # Generated counts
        generated_counts = data[gen_key][location_key].value_counts(
                                normalize=True
                            ).sort_index()
        generated_counts = generated_counts.reindex(x_values, fill_value=0)

        # Bar plotting with edge color for clarity
        ax.bar(
            x_values - 0.2,
            original_counts,
            width=0.4,
            align='center',
            alpha=0.8,
            label=f'{original_key}',
            edgecolor='black'
        )
        ax.bar(
            x_values + 0.2,
            generated_counts,
            width=0.4,
            align='center',
            alpha=0.6,
            label=gen_key,
            edgecolor='black'
        )

        ax.set_title(
            f"Location Count Distribution: {gen_key} vs {original_key}",
            fontsize=font_size + 2,
            weight='bold'
        )
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper right", fontsize=font_size * 0.9)

    # X-axis labeling on the last subplot
    axs[-1].set_xticks(x_values)
    axs[-1].set_xlabel(
        f"{location_key} (1 to {int(max_location)})",
        fontsize=font_size + 1
    )

    plt.tight_layout()
    plt.show()


def plot_location_count_histograms_demo_sex(data, location_key, font_size=10):
    plt.rcParams.update({'font.size': font_size})

    keys = list(data.keys())
    original_key_male = keys[0]
    original_key_female = keys[1]
    generated_keys = keys[2:]

    # Determine max location ID to set x-axis
    max_location = max(
        max(data[key][location_key].dropna().unique()) for key in data.keys()
    )
    x_values = np.arange(2, int(max_location) + 1)

    fig, axs = plt.subplots(
        len(generated_keys)+1+len(generated_keys) // 2,
        1,
        figsize=(12, 3 * len(generated_keys)),
        sharex=True
    )

    # Ensure axs is iterable
    if len(generated_keys) == 1:
        axs = [axs]

    # Precompute original counts once
    original_counts_male = data[original_key_male][location_key].value_counts(
                        normalize=True
                    ).sort_index()
    original_counts_male = original_counts_male.reindex(x_values, fill_value=0)

    original_counts_female = data[
        original_key_female
    ][location_key].value_counts(normalize=True).sort_index()
    original_counts_female = original_counts_female.reindex(
        x_values,
        fill_value=0
    )

    axes_count = 0

    ax = axs[axes_count]

    # Bar plotting with edge color for clarity
    ax.bar(
        x_values - 0.2,
        original_counts_male,
        width=0.4,
        align='center',
        alpha=0.8,
        label=f'{original_key_male}',
        edgecolor='black'
    )
    ax.bar(
        x_values + 0.2,
        original_counts_female,
        width=0.4,
        align='center',
        alpha=0.6,
        label=f'{original_key_female}',
        edgecolor='black'
    )

    ax.set_title(
        f"Location Count Distribution: {original_key_male} vs"
        f" {original_key_female}",
        fontsize=font_size + 2,
        weight='bold'
    )
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="upper right", fontsize=font_size * 0.9)
    axes_count += 1

    for idx, gen_key in enumerate(generated_keys):
        ax = axs[axes_count]

        # Generated counts
        generated_counts = data[gen_key][location_key].value_counts(
                                normalize=True
                            ).sort_index()
        generated_counts = generated_counts.reindex(x_values, fill_value=0)

        if idx % 2 == 0:
            original_counts = original_counts_male
            original_key = original_key_male
        if idx % 2 == 1:
            original_counts = original_counts_female
            original_key = original_key_female

        # Bar plotting with edge color for clarity
        ax.bar(
            x_values - 0.2,
            original_counts,
            width=0.4,
            align='center',
            alpha=0.8,
            label=f'{original_key}',
            edgecolor='black'
        )
        ax.bar(
            x_values + 0.2,
            generated_counts,
            width=0.4,
            align='center',
            alpha=0.6,
            label=gen_key,
            edgecolor='black'
        )
        axes_count += 1

        ax.set_title(
            f"Location Count Distribution: {gen_key} vs {original_key}",
            fontsize=font_size + 2,
            weight='bold'
        )
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper right", fontsize=font_size * 0.9)

    for i in range(0, len(generated_keys), 2):
        male_key = generated_keys[i]
        female_key = generated_keys[i+1] if i+1 < len(generated_keys) else None
        ax = axs[axes_count]

        # Generated counts
        generated_counts_male = data[male_key][location_key].value_counts(
                                normalize=True
                            ).sort_index()
        generated_counts_male = generated_counts_male.reindex(
            x_values,
            fill_value=0
        )

        generated_counts_female = data[female_key][location_key].value_counts(
                                normalize=True
                            ).sort_index()
        generated_counts_female = generated_counts_female.reindex(
            x_values,
            fill_value=0
        )

        # Bar plotting with edge color for clarity
        ax.bar(
            x_values - 0.2,
            generated_counts_male,
            width=0.4,
            align='center',
            alpha=0.8,
            label=f'{male_key}',
            edgecolor='black'
        )
        ax.bar(
            x_values + 0.2,
            generated_counts_female,
            width=0.4,
            align='center',
            alpha=0.6,
            label=f'{female_key}',
            edgecolor='black'
        )
        axes_count += 1

        ax.set_title(
            f"Location Count Distribution: {male_key} vs {female_key}",
            fontsize=font_size + 2,
            weight='bold'
        )
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper right", fontsize=font_size * 0.9)

    # X-axis labeling on the last subplot
    axs[-1].set_xticks(x_values)
    axs[-1].set_xlabel(
        f"{location_key} (1 to {int(max_location)})",
        fontsize=font_size + 1
    )

    plt.tight_layout()
    plt.show()


def pattern_level(
    data,
    loc_type_labels,
    font_size=12
):
    for name, df in data.items():
        print(name)
        print('*'*50)
        print(f"Average Location: {df['location'].mean()}")
        print(f"Median Location: {df['location'].median()}")
        print(f'Number of samples: {df.shape[0]}')
        if 'travel_time' in df.columns:
            print(f'Travel time (Hrs): {df.travel_time.mean()/(60*60)}')
        else:
            print('Travel time (Hrs): N/A')
        print('*'*50)
    plot_loc_type_distribution(
        data,
        loc_type_labels,
        font_size
    )
    plot_location_boxplot(
        data,
        font_size
    )
    plot_travel_time_boxplot(
        data,
        font_size
    )
    plot_location_boxplot(
        data,
        font_size,
        showfliers=True
    )
    plot_travel_time_boxplot(
        data,
        font_size,
        showfliers=True
    )
    plot_location_count_histograms(
        data,
        'location',
        font_size
    )


def pattern_level_demo_sex(
    data,
    loc_type_labels,
    font_size=12
):
    for name, df in data.items():
        print(name)
        print('*'*50)
        print(f"Average Location: {df['location'].mean()}")
        print(f"Median Location: {df['location'].median()}")
        print(f'Number of samples: {df.shape[0]}')
        if 'travel_time' in df.columns:
            print(f'Travel time (Hrs): {df.travel_time.mean()/(60*60)}')
        else:
            print('Travel time (Hrs): N/A')
        print('*'*50)
    plot_loc_type_distribution(
        data,
        loc_type_labels,
        font_size
    )
    plot_location_boxplot(
        data,
        font_size
    )
    plot_travel_time_boxplot(
        data,
        font_size
    )
    plot_location_boxplot(
        data,
        font_size,
        showfliers=True
    )
    plot_travel_time_boxplot(
        data,
        font_size,
        showfliers=True
    )
    plot_location_count_histograms_demo_sex(
        data,
        'location',
        font_size
    )


# Trip Level

def get_normalized_norm(M1, M2):
    return np.linalg.norm(
        (M1/M1.sum().sum()).to_numpy() - (M2/M2.sum().sum()).to_numpy()
    )


def get_norms(data):
    res = {
        list(data.keys())[0]: None
    }
    orig_val = data[list(data.keys())[0]]

    for name, data in data.items():
        res[name] = float(get_normalized_norm(orig_val, data))
    return res


def get_norms_demo_sex(data):
    res = {
        list(data.keys())[0]: None,
        list(data.keys())[1]: None
    }
    orig_val_male = data[list(data.keys())[0]]
    orig_val_female = data[list(data.keys())[1]]

    for i, (name, data) in enumerate(data.items()):
        if i % 2 == 0:
            res[name] = float(get_normalized_norm(orig_val_male, data))
        if i % 2 == 1:
            res[name] = float(get_normalized_norm(orig_val_female, data))
    return res


def plot_destination_probability_distribution(data, loc_types, font_size=12):
    plt.rcParams.update({'font.size': font_size})

    if isinstance(loc_types, dict):
        state_codes = list(loc_types.keys())
        state_labels = [loc_types[k] for k in state_codes]
    else:
        state_codes = list(loc_types)
        state_labels = [str(x) for x in state_codes]

    aligned = {}
    for name, matrix in data.items():
        if isinstance(matrix, pd.DataFrame):
            col_sums = matrix.sum(axis=0)
            values = []
            for code in state_codes:
                if code in col_sums.index:
                    values.append(float(col_sums[code]))
                elif str(code) in col_sums.index:
                    values.append(float(col_sums[str(code)]))
                else:
                    values.append(0.0)
            aligned[name] = values
        else:
            aligned[name] = [0.0] * len(state_codes)

    df = pd.DataFrame(aligned, index=state_labels)
    # Normalize each model column to probabilities.
    denom = df.sum(axis=0).replace(0, np.nan)
    df = df.divide(denom, axis=1).fillna(0.0)

    fig, ax = plt.subplots(figsize=(14, 8))
    bar_width = 0.8 / len(df.columns)
    x = np.arange(len(df.index))

    colors = plt.cm.get_cmap('tab10', len(df.columns))

    for idx, column in enumerate(df.columns):
        bars = ax.bar(
            x + idx * bar_width, df[column],
            width=bar_width,
            label=column,
            color=colors(idx)
        )
        for bar in bars:
            height = bar.get_height()
            if height > 0.01:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=font_size * 0.6
                )

    ax.set_xticks(x + bar_width * (len(df.columns) - 1) / 2)
    ax.set_xticklabels(df.index, rotation=45, ha="right")
    ax.set_ylabel("Probability", fontsize=font_size)
    ax.set_title(
        "Destination Probability Distribution",
        fontsize=font_size + 2,
        weight='bold'
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        fontsize=font_size * 0.85,
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=min(4, len(df.columns))
    )

    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def trip_level(data, loc_types, font_size=12):
    pprint.pprint(get_norms(data))
    plot_destination_probability_distribution(
        data,
        loc_types, font_size
    )
    # for k, v in data.items():
    #     print(k)
    #     print('*'*50)
    #     get_chord_diagram(v, loc_types)
    #     print('*'*50)


def trip_level_demo_sex(data, loc_types, font_size=12):
    pprint.pprint(get_norms_demo_sex(data))
    plot_destination_probability_distribution(
        data,
        loc_types, font_size
    )


def get_chord_diagram(
        df,
        loc_type_key='loc_type',
        loc_type_labels=None,
        new_mappings=False,
        sim_mappings=True,
        common_mappings=False,
        newest_mappings=False,
        show_percentage=False,
        ax=None,
        only_return_matrix=False
):
    _require_circos()
    if loc_type_labels is not None:
        dnames = loc_type_labels
    elif new_mappings is True:
        dnames = NEW_LOC_TYPES
    elif common_mappings is True:
        dnames = COMMON_LOC_TYPES
    # elif newest_mappings is True:
    #     dnames = NEWEST_LOC_TYPES_SHORTER
    elif sim_mappings is True:
        dnames = SIM_LOC_TYPES
    else:
        dnames = NHTS_LOC_TYPES
    C = []
    for lt in df[loc_type_key].values:
        for i, a in enumerate(lt):
            if i < len(lt)-1:
                C.append([a, lt[i+1]])

    counts = {}

    result_list = []

    for inner_list in C:
        inner_tuple = tuple(inner_list)

        if inner_tuple not in counts:
            counts[inner_tuple] = 1
        else:
            counts[inner_tuple] += 1
    for each in counts:
        result_list.append([each[0], each[1], counts[each]])

    list_of_codes = list(
        set(
            list(
                chain.from_iterable(df[loc_type_key].values)
            )
        )
    )
    list_of_codes = sorted(list_of_codes)

    C = pd.DataFrame(result_list, columns=['code_from', 'code_to', 'samples'])
    dflow = defaultdict(lambda: {})
    for i in range(len(C)):
        dflow[C["code_from"][i]][C["code_to"][i]] = C["samples"][i]
    M = np.zeros((len(list_of_codes), len(list_of_codes)))
    for i in range(len(M)):
        for j in range(len(M)):
            try:
                M[i, j] = dflow[list_of_codes[i]][list_of_codes[j]]
            except KeyError:
                M[i, j] = 0

    # Compute transition percentages
    category_totals = defaultdict(int)
    for i in range(len(C)):
        category_totals[C["code_from"][i]] += C["samples"][i]

    percentages = {
        dnames[key]: (value / C['samples'].sum()) * 100
        for key, value in category_totals.items()
    }

    labels = [
        f"{dnames[i]}\n({percentages[dnames[i]]:.1f}%)"
        if show_percentage else dnames[i]
        for i in list_of_codes
    ]

    M = pd.DataFrame(M, columns=[dnames[i] for i in list_of_codes])
    M.index = labels
    M.columns = labels
    # M.index = [dnames[i] for i in list_of_codes]

    tab20 = plt.cm.get_cmap('tab20')
    cmaps = {}
    for k, (i, val) in enumerate(dnames.items()):
        if show_percentage and val in percentages:
            val = f"{val}\n({percentages[val]:.1f}%)"
        cmaps[val] = tab20(k)

    if only_return_matrix is True:
        return M
    circos = Circos.initialize_from_matrix(
        M,
        space=5,
        cmap=cmaps,
        label_kws=dict(size=12, orientation="vertical"),
        link_kws=dict(ec="black", lw=0.05, direction=1),
    )
    if ax is not None:
        circos.plotfig(ax=ax)
    else:
        circos.plotfig()
    return M


def plot_chord_diagrams(
    data_dict, loc_dict, figsize=(18, 12), loc_type_labels=None
):
    num_items = len(data_dict)
    n_cols = math.ceil(math.sqrt(num_items))
    n_rows = math.ceil(num_items / n_cols)

    fig, axarr = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        subplot_kw={'projection': 'polar'}
    )
    axarr = axarr.flatten()

    if loc_dict is None:
        for idx, (title, data) in enumerate(data_dict.items()):
            ax = axarr[idx]
            ax.set_title(title, fontsize=18, pad=40)
            get_chord_diagram(
                data,
                loc_type_labels=loc_type_labels,
                sim_mappings=True,
                show_percentage=True,
                ax=ax
            )
        for j in range(idx + 1, len(axarr)):
            fig.delaxes(axarr[j])
    else:
        for idx, (title, data) in enumerate(data_dict.items()):
            ax = axarr[loc_dict[idx]]
            ax.set_title(title, fontsize=18, pad=40)
            get_chord_diagram(
                data,
                loc_type_labels=loc_type_labels,
                sim_mappings=True,
                show_percentage=True,
                ax=ax
            )
        for j in range(0, len(axarr)):
            if j not in loc_dict:
                fig.delaxes(axarr[j])

    plt.tight_layout()
    plt.show()
    return fig, axarr


# Activity Chains Level

def sf_activity_chains_analysis_details(
    val_1,
    overlap_1,
    val_2,
    overlap_2,
    val_3,
    overlap_3,
    overlap
):
    print(
        f"% of Generated Chains Present in Actual Chains"
        f": {round(val_1 * 100, 2)}"
    )
    print(
        "Weight of Generated Chains Present in Actual Chains"
        f": {round(overlap_1, 2)}"
    )
    print(
        f"% of Generated Chains Present in All Actual Chains"
        f": {round(val_2 * 100, 2)}"
    )
    print(
        "Weight of Generated Chains Present in All Actual Chains"
        f": {round(overlap_2, 2)}"
    )
    print(
        f"% of Actual Chains Present in Generated Chains"
        f": {round(val_3 * 100, 2)}"
    )
    print(
        "Weight of Actual Chains Present in Generated Chains"
        f": {round(overlap_3, 2)}"
    )
    print(
        "Weight Overlap of Actual Chains and Generated Chains"
        f": {round(overlap, 2)}"
    )


def sf_activity_chains_analysis(
    t_all_orig,
    t_sf_orig,
    t_sf_gen,
    name='LLama',
    print_details=True
):
    t_all_orig = t_all_orig['loc_type'].apply(tuple).value_counts()
    t_sf_orig = t_sf_orig['loc_type'].apply(tuple).value_counts()
    t_sf_gen = t_sf_gen['loc_type'].apply(tuple).value_counts()
    t_all_orig = t_all_orig / t_all_orig.sum() * 100
    t_sf_orig = t_sf_orig / t_sf_orig.sum() * 100
    t_sf_gen = t_sf_gen / t_sf_gen.sum() * 100

    val_1 = len(
        t_sf_gen.index.intersection(t_sf_orig.index)
    ) / len(t_sf_gen.index)
    overlap_1 = 0
    for value in t_sf_gen.index.intersection(t_sf_orig.index):
        overlap_1 += t_sf_gen[value]

    val_2 = len(
        t_sf_gen.index.intersection(t_all_orig.index)
    ) / len(t_sf_gen.index)
    overlap_2 = 0
    for value in t_sf_gen.index.intersection(t_all_orig.index):
        overlap_2 += t_sf_gen[value]

    val_3 = len(
        t_sf_gen.index.intersection(t_sf_orig.index)
    ) / len(t_sf_orig.index)
    overlap_3 = 0
    for value in t_sf_gen.index.intersection(t_sf_orig.index):
        overlap_3 += t_sf_orig[value]

    overlap = 0
    for value in t_sf_gen.index.intersection(t_sf_orig.index):
        overlap += min(t_sf_gen[value], t_sf_orig[value])

    if print_details:
        print(name)
        sf_activity_chains_analysis_details(
            val_1,
            overlap_1,
            val_2,
            overlap_2,
            val_3,
            overlap_3,
            overlap
        )

    return val_1, overlap_1, val_2, overlap_2, val_3, overlap_3, overlap


def activity_chains_analysis_all(data_dict, t_all):
    data_dict_temp = data_dict.copy()
    first_key = next(iter(data_dict_temp))
    t_orig = data_dict_temp.pop(first_key)
    results = []
    for key, value in data_dict_temp.items():
        t_gen = value
        val_1, overlap_1, val_2, overlap_2, val_3, overlap_3, overlap = (
            sf_activity_chains_analysis(
                t_all,
                t_orig,
                t_gen,
                key,
                True
            )
        )
        results.append({
            'Model': key,
            '% of Generated in Actual': round(val_1 * 100, 2),
            'Weight of Generated in Actual': round(overlap_1, 2),
            '% of Generated in All Actual': round(val_2 * 100, 2),
            'Weight of Generated in All Actual': round(overlap_2, 2),
            '% of Actual in Generated': round(val_3 * 100, 2),
            'Weight of Actual in Generated': round(overlap_3, 2),
            'Weight Overlap': round(overlap, 2),
        })
    return pd.DataFrame(results)


def plot_average_location_by_month(
    dfs_input: list[pd.DataFrame],
    df_names: list[str],
    city: str,
    figsize=(12, 7)
):
    if sns is None:
        raise ImportError("seaborn is required for month trend plotting.")
    # Define month names and the custom order for plotting
    month_names_map = {
        1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
        7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
    }
    # Define the desired order of months starting from July
    ordered_months_numerical = [7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]
    ordered_month_labels = [
        month_names_map[m] for m in ordered_months_numerical
    ]
    for i, df_input in enumerate(dfs_input):
        df_name = df_names[i]
        df = df_input.copy()
        df['survey_month'] = df['survey_date'].dt.month
        df = df.groupby('survey_month')['location'].mean().reset_index()
        df['DataFrame_Name'] = df_name
        df['month_label'] = df['survey_month'].map(month_names_map)
        df['month_label'] = pd.Categorical(
            df['month_label'],
            categories=ordered_month_labels,
            ordered=True
        )
        df = df.sort_values('month_label')
        if i == 0:
            combined_avg_locations = df
        else:
            combined_avg_locations = pd.concat([combined_avg_locations, df])
        print(f'{df_name} : {dfs_input[i].shape[0]}')

    plt.figure(figsize=figsize)

    sns.lineplot(
        x='month_label',
        y='location',
        hue='DataFrame_Name',
        data=combined_avg_locations,
        marker='o',
        palette='tab10',
        linewidth=2
    )

    plt.title(
        f'Average Location Count by Month ({city})', fontsize=16, weight='bold'
    )
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Location Count', fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.legend(title='Data Series', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    plt.show()
    # print(combined_avg_locations)
    avg_locs_orig = combined_avg_locations.loc[
        combined_avg_locations['DataFrame_Name'] == df_names[0]
    ]
    for each in df_names[1:]:
        avg_locs_gen = combined_avg_locations.loc[
            combined_avg_locations['DataFrame_Name'] == each
        ]
        # Calculate RMSE
        rmse = np.sqrt(
            np.mean(
                (avg_locs_orig['location'] - avg_locs_gen['location']) ** 2
            )
        )
        print(f'RMSE between {df_names[0]} and {each}: {rmse:.2f}')


def plot_median_location_by_month(
    dfs_input: list[pd.DataFrame],
    df_names: list[str],
    city: str
):
    if sns is None:
        raise ImportError("seaborn is required for month trend plotting.")
    for i, df_input in enumerate(dfs_input):
        df_name = df_names[i]
        df = df_input.copy()
        df['survey_month'] = df['survey_date'].dt.month
        df = df.groupby('survey_month')['location'].median().reset_index()
        df['DataFrame_Name'] = df_name
        if i == 0:
            combined_avg_locations = df
        else:
            combined_avg_locations = pd.concat([combined_avg_locations, df])

    plt.figure(figsize=(12, 7))

    sns.lineplot(
        x='survey_month',
        y='location',
        hue='DataFrame_Name',
        data=combined_avg_locations,
        marker='o',
        palette='tab10',
        linewidth=2
    )

    plt.title(
        f'Median Location Count by Month ({city})', fontsize=16, weight='bold'
    )
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Average Location Count', fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    plt.legend(title='Data Series', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    plt.show()


def location_count_dist_by_age_sex(
    df_t,
    df_gen_t,
    bins=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    by_age_only=False,
    title=None
):
    if sns is None:
        raise ImportError(
            "seaborn is required for age/sex location distribution plotting."
        )
    df = df_t.copy()
    df_gen = df_gen_t.copy()
    df['age_bin'] = pd.cut(
        df['age'],
        bins=bins,
        right=False
    )
    df_gen['age_bin'] = pd.cut(
        df_gen['age'],
        bins=bins,
        right=False
    )

    df['source'] = 'original'
    df_gen['source'] = 'generated'
    df_all = pd.concat([df, df_gen], ignore_index=True)
    df_all['group'] = df_all['source'] + '_' + df_all['sex']

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x='age_bin',
        y='location',
        hue='source',
        data=df_all,
        showfliers=False,
    )
    y_min, y_max = plt.ylim()
    plt.yticks(np.arange(int(y_min), int(y_max) + 1))
    # Show grid lines
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.7)
    plt.xticks(rotation=45)
    plt.title(f'Location Count Distribution by Age {title if title else ""}')
    plt.legend(title='Source')
    plt.tight_layout()

    mean_orig = df.groupby('age_bin')['location'].mean().reset_index()
    mean_gen = df_gen.groupby('age_bin')['location'].mean().reset_index()
    count_orig = df.groupby('age_bin')['location'].count().reset_index()
    count_gen = df_gen.groupby('age_bin')['location'].count().reset_index()

    df_merged_for_rmse = pd.merge(
        mean_orig,
        mean_gen,
        on='age_bin',
        how='inner',
        suffixes=('_Original', '_Generated')
    )

    df_merged_for_rmse = pd.merge(
        df_merged_for_rmse,
        count_orig,
        on='age_bin',
        how='inner'
    )
    df_merged_for_rmse = pd.merge(
        df_merged_for_rmse,
        count_gen,
        on='age_bin',
        how='inner',
        suffixes=('_Count_Original', '_Count_Generated')
    )

    rmse = np.sqrt(
        np.mean(
            (
                df_merged_for_rmse['location_Original']
                - df_merged_for_rmse['location_Generated']
            )**2
        )
    )
    print(
        "\nRoot Mean Square Error (RMSE) between the two DataFrames: "
        f"{rmse:.4f}"
    )

    print(
        df_merged_for_rmse.to_markdown(
            index=False, numalign="left", stralign="left"
        )
    )

    plt.plot(
        mean_orig['age_bin'].astype(str), mean_orig['location'], marker='o',
        color='blue', label='Original Mean', linewidth=2,
    )
    plt.plot(
        mean_gen['age_bin'].astype(str), mean_gen['location'], marker='o',
        color='orange', label='Generated Mean', linewidth=2
    )
    plt.legend()
    plt.show()

    if not by_age_only:
        trend_data = df_all.groupby(
            ['age_bin', 'group']
        )['location'].mean().reset_index()

        comparisons = [
            ('Original: Male vs Female', 'original_male', 'original_female'),
            ('Generated: Male vs Female', 'generated_male', 'generated_female'),  # noqa: E501
            ('Male: Original vs Gen', 'original_male', 'generated_male'),
            ('Female: Original vs Gen', 'original_female', 'generated_female'),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)
        for ax, (title, group1, group2) in zip(axes.flat, comparisons):
            subset = df_all[df_all['group'].isin([group1, group2])]
            sns.boxplot(
                x='age_bin', y='location', hue='group', data=subset, ax=ax
            )
            for group, color in zip([group1, group2], ['blue', 'orange']):
                trend = trend_data[trend_data['group'] == group]
                ax.plot(
                    trend['age_bin'].astype(str),
                    trend['location'],
                    marker='o',
                    linestyle='-', linewidth=2, label=f'{group}', color=color
                )
            ax.set_title(title)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            ax.legend(title='Group')
        fig.suptitle('Location Count Distribution by Age and Sex', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def loc_significance_test_percentage(
    t_actual,
    t_others,
    name='SF',
    font_size=12
):
    plt.rcParams.update({'font.size': font_size})

    # Colors
    actual_color = '#1f77b4'
    synthetic_color = '#ff7f0e'

    # Prepare data
    fig, axes = plt.subplots(
        int(np.ceil(len(t_others) / 2)),
        2,
        figsize=(12, int(np.ceil(len(t_others) / 2)) * 3)
    )
    axes = axes.flatten()

    combined_data = t_actual.copy()
    for e in list(t_others.values()):
        combined_data += e

    max_val = max(combined_data)
    number_counts_actual = {num: 0 for num in range(1, max_val + 1)}
    number_counts_actual.update(Counter(t_actual))
    total_count_actual = len(t_actual)
    number_percentages_actual = {
        num: (count / total_count_actual) * 100 for num, count in number_counts_actual.items()  # noqa: E501
    }

    # Plot percentage histograms
    for i, (key, value) in enumerate(t_others.items()):
        number_counts_temp = {num: 0 for num in range(1, max_val + 1)}
        number_counts_temp.update(Counter(value))
        total_count_temp = len(value)
        number_percentages_temp = {
            num: (count / total_count_temp) * 100 for num, count in number_counts_temp.items()  # noqa: E501
        }

        ax = axes[i]
        bar_width = 0.4
        x_vals = np.array(list(number_percentages_actual.keys()))

        ax.bar(x_vals - bar_width/2,
               list(number_percentages_actual.values()),
               width=bar_width,
               label='Actual',
               color=actual_color,
               alpha=0.7)

        ax.bar(x_vals + bar_width/2,
               list(number_percentages_temp.values()),
               width=bar_width,
               label=key,
               color=synthetic_color,
               alpha=0.7)

        ax.set_xticks(x_vals)
        ax.set_title(f'{name}: Actual vs. {key}')
        ax.set_xlabel('Number of Locations')
        ax.set_ylabel('Percentage')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best')

    plt.tight_layout()
    plt.show()

    # Significance Testing
    alpha = 0.05
    for key, value in t_others.items():
        print('\n' + '-'*50)
        print(f'📊 Comparing Actual vs. {key}')

        # Levene's Test
        stat, p_value = levene(t_actual, value)
        print(f"Levene's test statistic: {stat:.3f}, p-value: {p_value:.3f}")
        print("➡️ Variance: " + ("Equal" if p_value > alpha else "Unequal"))

        # Student's t-test
        t_stat, p_value = ttest_ind(t_actual, value, equal_var=True)
        print(f"Student's t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
        print(
            "➡️ Mean Comparison (Equal Var): "
            + (
                "Same distribution"
                if p_value > alpha else "Different distribution"
            )
        )

        # Welch's t-test
        t_stat, p_value = ttest_ind(t_actual, value, equal_var=False)
        print(f"Welch's t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
        print(
            "➡️ Mean Comparison (Unequal Var): "
            + (
                "Same distribution"
                if p_value > alpha else "Different distribution"
            )
        )


def plot_location_count_percentage_grid(
    actual_counts,
    model_counts: Dict[str, List[float]],
    title_prefix: str = "SF",
    font_size: int = 12,
    ncols: int = 2,
):
    """Visual-only location count percentage comparison grid."""
    plt.rcParams.update({'font.size': font_size})

    model_items = list(model_counts.items())
    if not model_items:
        print("No model location-count series provided.")
        return

    cleaned_actual = [
        int(v) for v in actual_counts
        if pd.notna(v)
    ]
    if not cleaned_actual:
        print("Actual location-count series is empty.")
        return

    max_val = max(cleaned_actual)
    for _, vals in model_items:
        cleaned = [int(v) for v in vals if pd.notna(v)]
        if cleaned:
            max_val = max(max_val, max(cleaned))

    x_vals = np.arange(1, max_val + 1)
    actual_counter = Counter(cleaned_actual)
    actual_pct = np.array([
        actual_counter.get(x, 0) / len(cleaned_actual) * 100 for x in x_vals
    ])

    nrows = math.ceil(len(model_items) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 4 * nrows),
        squeeze=False,
    )
    axes = axes.flatten()

    bar_width = 0.38
    for i, (label, vals) in enumerate(model_items):
        cleaned = [int(v) for v in vals if pd.notna(v)]
        ax = axes[i]
        if not cleaned:
            ax.text(
                0.5,
                0.5,
                f"{label}: N/A (no location counts)",
                transform=ax.transAxes,
                ha='center',
                va='center',
            )
            ax.set_axis_off()
            continue

        model_counter = Counter(cleaned)
        model_pct = np.array([
            model_counter.get(x, 0) / len(cleaned) * 100 for x in x_vals
        ])

        ax.bar(
            x_vals - bar_width / 2,
            actual_pct,
            width=bar_width,
            label='Actual',
            alpha=0.75,
            color='#1f77b4',
        )
        ax.bar(
            x_vals + bar_width / 2,
            model_pct,
            width=bar_width,
            label=label,
            alpha=0.75,
            color='#ff7f0e',
        )
        ax.set_title(f"{title_prefix}: Actual vs. {label}")
        ax.set_xlabel("Number of Locations")
        ax.set_ylabel("Percentage")
        ax.set_xticks(x_vals)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    for j in range(len(model_items), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_location_count_percentage_grid_better(
    actual_counts,
    model_counts: Dict[str, List[float]],
    title_prefix: str = "SF",
    font_size: int = 12,
    ncols: int = 2,
    tail_start: int = 8,
    show_tail_inset: bool = True,
    inset_width: str = "42%",
    inset_height: str = "42%",
    main_xtick_step: Optional[int] = None,
    inset_xtick_step: int = 1,
    height_ratio: float = 4.2,
    width_ratio: float = 7.2,
    actual_color: str = "#5DA5DA",
    sim_color: str = "#ff7f0e",
):
    import math
    from collections import Counter
    from matplotlib.patches import Patch
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    plt.rcParams.update({"font.size": font_size})

    def _clean_counts(vals):
        return [int(v) for v in vals if pd.notna(v)]

    def _build_ticks(start, end, step):
        ticks = np.arange(start, end + 1, step)
        if len(ticks) == 0 or ticks[-1] != end:
            ticks = np.append(ticks, end)
        return ticks

    model_items = list(model_counts.items())
    if not model_items:
        print("No model location-count series provided.")
        return None, None

    cleaned_actual = _clean_counts(actual_counts)
    if not cleaned_actual:
        print("Actual location-count series is empty.")
        return None, None

    tail_start = max(1, int(tail_start))
    inset_xtick_step = max(1, int(inset_xtick_step))

    max_val = max(cleaned_actual)
    for _, vals in model_items:
        cleaned = _clean_counts(vals)
        if cleaned:
            max_val = max(max_val, max(cleaned))

    x_vals = np.arange(1, max_val + 1)

    actual_counter = Counter(cleaned_actual)
    actual_pct = np.array([
        actual_counter.get(x, 0) / len(cleaned_actual) * 100
        for x in x_vals
    ])

    if main_xtick_step is None:
        if max_val <= 15:
            main_xtick_step = 1
        elif max_val <= 30:
            main_xtick_step = 2
        else:
            main_xtick_step = 3

    nrows = math.ceil(len(model_items) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width_ratio * ncols, height_ratio * nrows),
        squeeze=False,
    )
    axes = axes.flatten()

    bar_width = 0.38
    used_axes = []
    inset_font_size = font_size - 4

    for i, (label, vals) in enumerate(model_items):
        ax = axes[i]
        used_axes.append(ax)

        cleaned = _clean_counts(vals)
        if not cleaned:
            ax.text(
                0.5,
                0.5,
                f"{label}: N/A (no location counts)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=font_size,
            )
            ax.set_axis_off()
            continue

        model_counter = Counter(cleaned)
        model_pct = np.array([
            model_counter.get(x, 0) / len(cleaned) * 100
            for x in x_vals
        ])

        ax.bar(
            x_vals - bar_width / 2,
            actual_pct,
            width=bar_width,
            alpha=0.75,
            color=actual_color,
        )
        ax.bar(
            x_vals + bar_width / 2,
            model_pct,
            width=bar_width,
            alpha=0.75,
            color=sim_color,
        )

        ymax_main = max(float(actual_pct.max()), float(model_pct.max()))
        ax.set_ylim(0, ymax_main * 1.12 if ymax_main > 0 else 1.0)
        ax.set_xlim(0.4, max_val + 0.6)

        main_xticks = _build_ticks(1, max_val, main_xtick_step)
        ax.set_xticks(main_xticks)

        ax.set_title(f"{title_prefix}: Actual vs. {label}", fontsize=font_size)
        ax.set_xlabel("Number of Locations", fontsize=font_size)
        ax.set_ylabel("Percentage", fontsize=font_size)
        ax.tick_params(axis="both", labelsize=font_size)
        ax.grid(True, linestyle="--", alpha=0.45)

        if show_tail_inset and tail_start <= max_val:
            ax.axvspan(
                tail_start - 0.5,
                max_val + 0.5,
                color="gray",
                alpha=0.06,
            )

            tail_mask = x_vals >= tail_start
            tail_x = x_vals[tail_mask]
            tail_actual = actual_pct[tail_mask]
            tail_model = model_pct[tail_mask]

            if np.any((tail_actual > 0) | (tail_model > 0)):
                axins = inset_axes(
                    ax,
                    width=inset_width,
                    height=inset_height,
                    loc="upper right",
                    borderpad=1.0,
                )

                axins.bar(
                    tail_x - bar_width / 2,
                    tail_actual,
                    width=bar_width,
                    alpha=0.75,
                    color=actual_color,
                )
                axins.bar(
                    tail_x + bar_width / 2,
                    tail_model,
                    width=bar_width,
                    alpha=0.75,
                    color=sim_color,
                )

                ymax_tail = max(
                    float(tail_actual.max()) if tail_actual.size else 0.0,
                    float(tail_model.max()) if tail_model.size else 0.0,
                )

                axins.set_xlim(tail_x[0] - 0.7, tail_x[-1] + 0.7)
                axins.set_ylim(0, max(1.0, ymax_tail * 1.2))

                inset_xticks = _build_ticks(
                    start=int(tail_x[0]),
                    end=int(tail_x[-1]),
                    step=inset_xtick_step,
                )
                axins.set_xticks(inset_xticks)
                axins.set_title(
                    f"Tail ({tail_start}+)",
                    fontsize=inset_font_size,
                    pad=2,
                )
                axins.tick_params(axis="both", labelsize=inset_font_size)
                axins.grid(True, linestyle="--", alpha=0.35)

    for j in range(len(model_items), len(axes)):
        fig.delaxes(axes[j])

    legend_handles = [
        Patch(
            facecolor=actual_color, edgecolor="none",
            alpha=0.75, label="Actual"
        ),
        Patch(
            facecolor=sim_color, edgecolor="none",
            alpha=0.75, label="Sim"
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        fontsize=font_size,
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

    return fig, np.array(used_axes, dtype=object)


def plot_travel_time_pair_grid(
    actual_travel_seconds: List[float],
    model_travel_seconds: Dict[str, List[float]],
    title_prefix: str = "SF",
    font_size: int = 12,
    ncols: int = 2,
):
    """Visual-only per-model travel-time boxplot grid (Actual vs Model)."""
    plt.rcParams.update({'font.size': font_size})

    cleaned_actual = [
        float(v) / 3600.0 for v in actual_travel_seconds
        if pd.notna(v)
    ]
    if not cleaned_actual:
        print("Actual travel-time series is empty.")
        return

    model_items = list(model_travel_seconds.items())
    if not model_items:
        print("No model travel-time series provided.")
        return

    nrows = math.ceil(len(model_items) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7 * ncols, 4 * nrows),
        squeeze=False,
    )
    axes = axes.flatten()

    for i, (label, vals) in enumerate(model_items):
        ax = axes[i]
        cleaned_model = [
            float(v) / 3600.0 for v in vals
            if pd.notna(v)
        ]
        if not cleaned_model:
            ax.text(
                0.5,
                0.5,
                f"{label}: N/A (no travel_time)",
                transform=ax.transAxes,
                ha='center',
                va='center',
            )
            ax.set_axis_off()
            continue

        bp = ax.boxplot(
            [cleaned_actual, cleaned_model],
            labels=['Actual', label],
            showmeans=True,
            showfliers=False,
            patch_artist=True,
        )
        bp['boxes'][0].set_facecolor('#1f77b4')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('#ff7f0e')
        bp['boxes'][1].set_alpha(0.7)
        ax.set_ylabel("Travel Time (hours)")
        ax.set_title(f"{title_prefix}: Actual vs. {label}")
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    for j in range(len(model_items), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def _metric_row(
    model: str,
    metric: str,
    value: Optional[float],
    unit: str,
    evaluable: bool,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    row = {
        "model": model,
        "metric": metric,
        "value": (float(value) if value is not None else None),
        "unit": unit,
        "evaluable": bool(evaluable),
        "reason": reason if (reason and not evaluable) else "",
    }
    return row


def _coerce_list(value: Any) -> List[Any]:
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


def _coerce_sequence_to_ints(seq_like: Any) -> List[int]:
    seq = _coerce_list(seq_like)
    out: List[int] = []
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


def _resolve_sequences(df: pd.DataFrame) -> List[List[int]]:
    for col in ("loc_type", "loc_type2", "loc_type_new"):
        if col in df.columns:
            return df[col].apply(_coerce_sequence_to_ints).tolist()
    return [[] for _ in range(len(df))]


def _resolve_time_lists(df: pd.DataFrame) -> Optional[List[List[str]]]:
    if "combined_time" not in df.columns:
        return None
    values = df["combined_time"].apply(_coerce_list).tolist()
    has_non_empty = any(len(v) > 0 for v in values)
    return values if has_non_empty else None


def _parse_time_to_minutes(value: str) -> Optional[int]:
    for fmt in ("%I:%M %p", "%H:%M"):
        try:
            dt = pd.to_datetime(value, format=fmt)
            return int(dt.hour * 60 + dt.minute)
        except Exception:
            continue
    return None


def _parse_time_range(range_text: str) -> Optional[Tuple[int, int]]:
    if not isinstance(range_text, str) or "-" not in range_text:
        return None
    parts = range_text.split("-")
    if len(parts) != 2:
        return None
    start_m = _parse_time_to_minutes(parts[0].strip())
    end_m = _parse_time_to_minutes(parts[1].strip())
    if start_m is None or end_m is None:
        return None
    if end_m < start_m:
        end_m += 24 * 60
    return start_m, end_m


def _distribution_from_sequences(
    seqs: Iterable[Iterable[int]],
) -> Dict[Tuple[int, ...], float]:
    counts: Dict[Tuple[int, ...], int] = defaultdict(int)
    total = 0
    for seq in seqs:
        vals = tuple(int(x) for x in seq)
        if not vals:
            continue
        counts[vals] += 1
        total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _transition_matrix(
    sequences: Iterable[Iterable[int]],
    state_codes: Iterable[int],
) -> Optional[np.ndarray]:
    states = [int(s) for s in state_codes]
    idx = {s: i for i, s in enumerate(states)}
    mat = np.zeros((len(states), len(states)), dtype=float)
    edge_count = 0
    for seq in sequences:
        vals = [int(x) for x in seq]
        for a, b in zip(vals[:-1], vals[1:]):
            if a in idx and b in idx:
                mat[idx[a], idx[b]] += 1.0
                edge_count += 1
    if edge_count == 0:
        return None
    mat /= mat.sum()
    return mat


def _transition_distance(
    matrix: Optional[np.ndarray],
    reference: Optional[np.ndarray],
) -> Dict[str, Optional[float]]:
    if matrix is None or reference is None:
        return {"mean_abs": None, "max_abs": None, "l2": None}
    diff = np.abs(matrix - reference)
    return {
        "mean_abs": float(diff.mean()),
        "max_abs": float(diff.max()),
        "l2": float(np.linalg.norm(diff)),
    }


def _kl_divergence(a: np.ndarray, b: np.ndarray) -> float:
    mask = (a > 0) & (b > 0)
    return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() == 0:
        p = np.ones_like(p)
    if q.sum() == 0:
        q = np.ones_like(q)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _distribution_jsd(
    a: Dict[Tuple[int, ...], float],
    b: Dict[Tuple[int, ...], float],
) -> Optional[float]:
    if not a or not b:
        return None
    keys = sorted(set(a.keys()) | set(b.keys()), key=lambda x: (len(x), x))
    p = np.array([a.get(k, 0.0) for k in keys], dtype=float)
    q = np.array([b.get(k, 0.0) for k in keys], dtype=float)
    return _jsd(p, q)


def _histogram_jsd(
    values: np.ndarray,
    reference: np.ndarray,
    clip_max: int = 12,
) -> Optional[float]:
    if len(values) == 0 or len(reference) == 0:
        return None
    v = np.clip(values.astype(int), 0, clip_max)
    r = np.clip(reference.astype(int), 0, clip_max)
    hv = np.bincount(v, minlength=clip_max + 1).astype(float)
    hr = np.bincount(r, minlength=clip_max + 1).astype(float)
    return _jsd(hv, hr)


def _abs_error(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs(float(a) - float(b))


def _location_counts(df: pd.DataFrame) -> np.ndarray:
    if "location" in df.columns:
        vals = pd.to_numeric(df["location"], errors="coerce").dropna()
        if len(vals):
            return vals.to_numpy(dtype=float)
    seqs = _resolve_sequences(df)
    return np.array([len(seq) for seq in seqs], dtype=float)


def _travel_time_hours(df: pd.DataFrame) -> Optional[float]:
    if "travel_time" not in df.columns:
        return None
    vals = pd.to_numeric(df["travel_time"], errors="coerce").dropna()
    if len(vals) == 0:
        return None
    return float(vals.mean() / 3600.0)


def _loc_type_distribution(
    df: pd.DataFrame,
    state_codes: Iterable[int],
) -> Optional[np.ndarray]:
    states = [int(s) for s in state_codes]
    idx = {s: i for i, s in enumerate(states)}
    counts = np.zeros(len(states), dtype=float)
    total = 0.0
    for seq in _resolve_sequences(df):
        for code in seq:
            if code in idx:
                counts[idx[code]] += 1.0
                total += 1.0
    if total == 0:
        return None
    return counts / total


def _infer_entity_ids(df: pd.DataFrame) -> np.ndarray:
    if "agent_id" in df.columns:
        return df["agent_id"].astype(str).to_numpy()
    if {"HOUSEID", "PERSONID"}.issubset(df.columns):
        return (
            df["HOUSEID"].astype(str) + ":" + df["PERSONID"].astype(str)
        ).to_numpy()
    if {"__household_id", "__person_id"}.issubset(df.columns):
        return (
            df["__household_id"].astype(str)
            + ":"
            + df["__person_id"].astype(str)
        ).to_numpy()
    return np.arange(len(df)).astype(str)


def _infer_is_weekend(df: pd.DataFrame) -> np.ndarray:
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
    elif "survey_date" in df.columns:
        d = pd.to_datetime(df["survey_date"], errors="coerce")
    else:
        d = pd.Series(pd.NaT, index=df.index)
    if d.notna().any():
        return (d.dt.weekday >= 5).fillna(False).to_numpy(dtype=bool)
    return np.zeros(len(df), dtype=bool)


def _first_start_for_activity(
    seq: List[int],
    pairs: List[Optional[Tuple[int, int]]],
    target_activity: int,
) -> Optional[float]:
    for i, (act, pair) in enumerate(zip(seq, pairs)):
        if act != target_activity:
            continue
        if i == 0:
            continue
        if seq[i - 1] == target_activity:
            continue
        if pair is None:
            continue
        return float(pair[0])
    return None


def _temporal_start_stats(
    df: pd.DataFrame,
    target_activity: int,
) -> Dict[str, Any]:
    seqs = _resolve_sequences(df)
    time_lists = _resolve_time_lists(df)
    if time_lists is None:
        return {
            "intra_mean_std_minutes": None,
            "inter_std_minutes": None,
            "weekday_start_samples": 0,
            "evaluable": False,
            "reason": "No combined_time data available.",
        }

    ids = _infer_entity_ids(df)
    is_weekend = _infer_is_weekend(df)

    rows = []
    for seq, times, entity, weekend in zip(seqs, time_lists, ids, is_weekend):
        pairs = [_parse_time_range(t) for t in times]
        start = _first_start_for_activity(
            seq=seq, pairs=pairs, target_activity=target_activity
        )
        rows.append(
            {
                "entity_id": entity,
                "is_weekend": bool(weekend),
                "start_min": start,
            }
        )
    temp = pd.DataFrame(rows)
    temp = temp[~temp["is_weekend"] & temp["start_min"].notna()].copy()
    if temp.empty:
        return {
            "intra_mean_std_minutes": None,
            "inter_std_minutes": None,
            "weekday_start_samples": 0,
            "evaluable": False,
            "reason": "No weekday arrival start samples for this activity.",
        }

    intra = temp.groupby("entity_id")["start_min"].std().dropna()
    means = temp.groupby("entity_id")["start_min"].mean().dropna()
    return {
        "intra_mean_std_minutes": float(intra.mean()) if len(intra) else None,
        "inter_std_minutes": float(means.std()) if len(means) else None,
        "weekday_start_samples": int(len(temp)),
        "evaluable": True,
        "reason": "",
    }


def _dwell_minutes_by_activity(
    df: pd.DataFrame,
) -> Optional[Dict[int, np.ndarray]]:
    seqs = _resolve_sequences(df)
    time_lists = _resolve_time_lists(df)
    if time_lists is None:
        return None

    dwell: Dict[int, List[float]] = defaultdict(list)
    for seq, times in zip(seqs, time_lists):
        pairs = [_parse_time_range(t) for t in times]
        for act, pair in zip(seq, pairs):
            if pair is None:
                continue
            start_m, end_m = pair
            duration = float(end_m - start_m)
            if duration > 0:
                dwell[int(act)].append(duration)

    return {
        act: np.array(vals, dtype=float)
        for act, vals in dwell.items()
        if len(vals) > 0
    }


def _trips_per_day(df: pd.DataFrame) -> np.ndarray:
    seqs = _resolve_sequences(df)
    return np.array([max(len(seq) - 1, 0) for seq in seqs], dtype=int)


def _chain_overlap_stats(
    actual_sf_df: pd.DataFrame,
    actual_all_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> Dict[str, Optional[float]]:
    sf_dist = _distribution_from_sequences(_resolve_sequences(actual_sf_df))
    all_dist = _distribution_from_sequences(_resolve_sequences(actual_all_df))
    gen_dist = _distribution_from_sequences(_resolve_sequences(model_df))

    sf_keys = set(sf_dist.keys())
    all_keys = set(all_dist.keys())
    gen_keys = set(gen_dist.keys())

    if not sf_keys or not gen_keys:
        return {
            "pct_generated_in_actual_sf": None,
            "weight_generated_in_actual_sf": None,
            "pct_generated_in_actual_all": None,
            "weight_generated_in_actual_all": None,
            "pct_actual_sf_in_generated": None,
            "weight_actual_sf_in_generated": None,
            "weight_overlap_sf_generated": None,
        }

    inter_sf = gen_keys.intersection(sf_keys)
    inter_all = gen_keys.intersection(all_keys)

    pct_generated_in_actual_sf = len(inter_sf) / max(1, len(gen_keys)) * 100.0
    pct_generated_in_actual_all = (
        len(inter_all) / max(1, len(gen_keys)) * 100.0
    )
    pct_actual_sf_in_generated = len(inter_sf) / max(1, len(sf_keys)) * 100.0

    weight_generated_in_actual_sf = float(
        sum(gen_dist[k] for k in inter_sf) * 100.0
    )
    weight_generated_in_actual_all = float(
        sum(gen_dist[k] for k in inter_all) * 100.0
    )
    weight_actual_sf_in_generated = float(
        sum(sf_dist[k] for k in inter_sf) * 100.0
    )
    weight_overlap_sf_generated = float(
        sum(min(gen_dist[k], sf_dist[k]) for k in inter_sf) * 100.0
    )

    return {
        "pct_generated_in_actual_sf": pct_generated_in_actual_sf,
        "weight_generated_in_actual_sf": weight_generated_in_actual_sf,
        "pct_generated_in_actual_all": pct_generated_in_actual_all,
        "weight_generated_in_actual_all": weight_generated_in_actual_all,
        "pct_actual_sf_in_generated": pct_actual_sf_in_generated,
        "weight_actual_sf_in_generated": weight_actual_sf_in_generated,
        "weight_overlap_sf_generated": weight_overlap_sf_generated,
    }


def build_pattern_metrics_table(
    actual_sf_df: pd.DataFrame,
    model_frames: Dict[str, pd.DataFrame],
    state_codes: Iterable[int] = (1, 2, 3, 4, 5, 7),
    loc_type_labels: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Returns notebook-first pattern metrics with raw values only."""
    rows: List[Dict[str, Any]] = []

    actual_loc = _location_counts(actual_sf_df)
    actual_loc_mean = float(actual_loc.mean()) if len(actual_loc) else None
    # actual_loc_median = (
    #     float(np.median(actual_loc)) if len(actual_loc) else None
    # )
    actual_travel_hours = _travel_time_hours(actual_sf_df)
    actual_loc_dist = _loc_type_distribution(actual_sf_df, state_codes)

    for model_name, model_df in model_frames.items():
        model_loc = _location_counts(model_df)
        model_loc_mean = float(model_loc.mean()) if len(model_loc) else None
        model_loc_median = (
            float(np.median(model_loc)) if len(model_loc) else None
        )
        model_travel_hours = _travel_time_hours(model_df)
        model_loc_dist = _loc_type_distribution(model_df, state_codes)

        rows.append(
            _metric_row(
                model=model_name,
                metric="location_count_mean",
                value=model_loc_mean,
                unit="stops/day",
                evaluable=model_loc_mean is not None,
                reason="No location counts available.",
            )
        )
        rows.append(
            _metric_row(
                model=model_name,
                metric="location_count_median",
                value=model_loc_median,
                unit="stops/day",
                evaluable=model_loc_median is not None,
                reason="No location counts available.",
            )
        )
        rows.append(
            _metric_row(
                model=model_name,
                metric="location_count_mean_abs_error_vs_actual",
                value=_abs_error(model_loc_mean, actual_loc_mean),
                unit="stops/day",
                evaluable=(
                    model_loc_mean is not None
                    and actual_loc_mean is not None
                ),
                reason="Missing actual or model location means.",
            )
        )

        travel_reason = "No travel_time column with valid values."
        rows.append(
            _metric_row(
                model=model_name,
                metric="travel_time_mean_hours",
                value=model_travel_hours,
                unit="hours/day",
                evaluable=model_travel_hours is not None,
                reason=travel_reason,
            )
        )
        rows.append(
            _metric_row(
                model=model_name,
                metric="travel_time_mean_abs_error_hours_vs_actual",
                value=_abs_error(model_travel_hours, actual_travel_hours),
                unit="hours/day",
                evaluable=(
                    model_travel_hours is not None
                    and actual_travel_hours is not None
                ),
                reason="Missing actual or model travel-time means.",
            )
        )

        loc_dist_l1 = None
        if model_loc_dist is not None and actual_loc_dist is not None:
            loc_dist_l1 = float(np.abs(model_loc_dist - actual_loc_dist).sum())
        rows.append(
            _metric_row(
                model=model_name,
                metric="loc_type_distribution_l1_vs_actual",
                value=loc_dist_l1,
                unit="probability_mass",
                evaluable=loc_dist_l1 is not None,
                reason="Missing loc-type distribution for actual or model.",
            )
        )

    return pd.DataFrame(rows)


def build_trip_metrics_table(
    actual_sf_df: pd.DataFrame,
    model_frames: Dict[str, pd.DataFrame],
    state_codes: Iterable[int] = (1, 2, 3, 4, 5, 7),
    loc_type_labels: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Returns trip-level transition diagnostics with transition diffs."""
    rows: List[Dict[str, Any]] = []

    actual_sequences = _resolve_sequences(actual_sf_df)
    actual_matrix = _transition_matrix(actual_sequences, state_codes)

    for model_name, model_df in model_frames.items():
        model_sequences = _resolve_sequences(model_df)
        model_matrix = _transition_matrix(model_sequences, state_codes)
        dist = _transition_distance(model_matrix, actual_matrix)

        rows.append(
            _metric_row(
                model=model_name,
                metric="transition_matrix_mean_abs_diff_vs_actual",
                value=dist["mean_abs"],
                unit="probability",
                evaluable=dist["mean_abs"] is not None,
                reason="Missing transitions in actual or model.",
            )
        )
        rows.append(
            _metric_row(
                model=model_name,
                metric="transition_matrix_max_abs_diff_vs_actual",
                value=dist["max_abs"],
                unit="probability",
                evaluable=dist["max_abs"] is not None,
                reason="Missing transitions in actual or model.",
            )
        )
        rows.append(
            _metric_row(
                model=model_name,
                metric="transition_matrix_l2_diff_vs_actual",
                value=dist["l2"],
                unit="norm",
                evaluable=dist["l2"] is not None,
                reason="Missing transitions in actual or model.",
            )
        )

        destination_l1 = None
        if model_matrix is not None and actual_matrix is not None:
            model_dest = model_matrix.sum(axis=0)
            actual_dest = actual_matrix.sum(axis=0)
            destination_l1 = float(
                np.abs(model_dest - actual_dest).sum()
            )
        rows.append(
            _metric_row(
                model=model_name,
                metric="destination_probability_l1_vs_actual",
                value=destination_l1,
                unit="probability_mass",
                evaluable=destination_l1 is not None,
                reason="Missing destination distributions in actual or model.",
            )
        )

    return pd.DataFrame(rows)


def build_chain_metrics_table(
    actual_sf_df: pd.DataFrame,
    actual_all_df: pd.DataFrame,
    model_frames: Dict[str, pd.DataFrame],
    loc_type_labels: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Returns chain JSD plus overlap-style metrics."""
    rows: List[Dict[str, Any]] = []

    actual_chain_dist = _distribution_from_sequences(
        _resolve_sequences(actual_sf_df)
    )

    for model_name, model_df in model_frames.items():
        model_chain_dist = _distribution_from_sequences(
            _resolve_sequences(model_df)
        )
        chain_jsd = _distribution_jsd(model_chain_dist, actual_chain_dist)
        rows.append(
            _metric_row(
                model=model_name,
                metric="chain_jsd_vs_actual_sf",
                value=chain_jsd,
                unit="jsd",
                evaluable=chain_jsd is not None,
                reason="Missing chain distributions for actual or model.",
            )
        )

        overlap = _chain_overlap_stats(actual_sf_df, actual_all_df, model_df)
        for metric_name, value, unit in [
            (
                "pct_generated_chains_in_actual_sf",
                overlap["pct_generated_in_actual_sf"],
                "percent",
            ),
            (
                "weight_generated_chains_in_actual_sf",
                overlap["weight_generated_in_actual_sf"],
                "percent",
            ),
            (
                "pct_generated_chains_in_actual_all",
                overlap["pct_generated_in_actual_all"],
                "percent",
            ),
            (
                "weight_generated_chains_in_actual_all",
                overlap["weight_generated_in_actual_all"],
                "percent",
            ),
            (
                "pct_actual_sf_chains_in_generated",
                overlap["pct_actual_sf_in_generated"],
                "percent",
            ),
            (
                "weight_actual_sf_chains_in_generated",
                overlap["weight_actual_sf_in_generated"],
                "percent",
            ),
            (
                "chain_weight_overlap_actual_sf_generated",
                overlap["weight_overlap_sf_generated"],
                "percent",
            ),
        ]:
            rows.append(
                _metric_row(
                    model=model_name,
                    metric=metric_name,
                    value=value,
                    unit=unit,
                    evaluable=value is not None,
                    reason="Missing chain samples for actual or model.",
                )
            )

    return pd.DataFrame(rows)


def build_temporal_dwell_trip_table(
    actual_sf_df: pd.DataFrame,
    model_frames: Dict[str, pd.DataFrame],
    temporal_activity_codes: Iterable[int] = (2, 4),
    state_codes: Iterable[int] = (1, 2, 3, 4, 5, 7),
    trips_clip_max: int = 12,
    loc_type_labels: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """Returns dwell, trips/day and temporal start diagnostics."""
    if loc_type_labels is None:
        loc_type_labels = SIM_LOC_TYPES
    rows: List[Dict[str, Any]] = []

    actual_trips = _trips_per_day(actual_sf_df)
    actual_trip_mean = (
        float(actual_trips.mean()) if len(actual_trips) else None
    )
    actual_dwell = _dwell_minutes_by_activity(actual_sf_df) or {}
    actual_temporal = {
        int(act): _temporal_start_stats(actual_sf_df, int(act))
        for act in temporal_activity_codes
    }

    for model_name, model_df in model_frames.items():
        model_trips = _trips_per_day(model_df)
        model_trip_mean = (
            float(model_trips.mean())
            if len(model_trips)
            else None
        )
        trips_w = None
        trips_jsd = None
        if len(model_trips) and len(actual_trips):
            trips_w = float(wasserstein_distance(model_trips, actual_trips))
            trips_jsd = _histogram_jsd(
                model_trips, actual_trips, clip_max=trips_clip_max
            )

        rows.extend(
            [
                _metric_row(
                    model_name,
                    "trips_per_person_day_mean",
                    model_trip_mean,
                    "trips/day",
                    evaluable=model_trip_mean is not None,
                    reason="No trips/day samples in model.",
                ),
                _metric_row(
                    model_name,
                    "trips_per_person_day_mean_abs_error_vs_actual",
                    _abs_error(model_trip_mean, actual_trip_mean),
                    "trips/day",
                    evaluable=(
                        model_trip_mean is not None
                        and actual_trip_mean is not None
                    ),
                    reason="Missing trips/day means for model or actual.",
                ),
                _metric_row(
                    model_name,
                    "trips_per_person_day_wasserstein_vs_actual",
                    trips_w,
                    "trips",
                    evaluable=trips_w is not None,
                    reason="Missing trips/day samples for model or actual.",
                ),
                _metric_row(
                    model_name,
                    "trips_per_person_day_jsd_vs_actual",
                    trips_jsd,
                    "jsd",
                    evaluable=trips_jsd is not None,
                    reason="Missing trips/day samples for model or actual.",
                ),
            ]
        )

        model_dwell = _dwell_minutes_by_activity(model_df)
        dwell_acts = sorted(set(int(s) for s in state_codes))
        dwell_distances: List[float] = []
        for act in dwell_acts:
            act_name = str(loc_type_labels.get(act, str(act))).lower()
            ref_vals = actual_dwell.get(act)
            model_vals = None if model_dwell is None else model_dwell.get(act)
            dwell_w = None
            if (
                model_vals is not None
                and ref_vals is not None
                and len(model_vals)
                and len(ref_vals)
            ):
                dwell_w = float(wasserstein_distance(model_vals, ref_vals))
                dwell_distances.append(dwell_w)
            rows.append(
                _metric_row(
                    model=model_name,
                    metric=f"dwell_wasserstein_{act_name}_minutes_vs_actual",
                    value=dwell_w,
                    unit="minutes",
                    evaluable=dwell_w is not None,
                    reason=(
                        "Missing comparable dwell samples for model or actual."
                        if dwell_w is None else ""
                    ),
                )
            )

        dwell_mean = (
            float(np.mean(dwell_distances))
            if len(dwell_distances)
            else None
        )
        rows.append(
            _metric_row(
                model=model_name,
                metric="dwell_wasserstein_mean_minutes_vs_actual",
                value=dwell_mean,
                unit="minutes",
                evaluable=dwell_mean is not None,
                reason="No evaluable dwell activity distances.",
            )
        )

        for act in temporal_activity_codes:
            act = int(act)
            act_name = str(loc_type_labels.get(act, str(act))).lower()
            model_stats = _temporal_start_stats(model_df, act)
            ref_stats = actual_temporal[act]

            model_intra = model_stats["intra_mean_std_minutes"]
            model_inter = model_stats["inter_std_minutes"]
            ref_intra = ref_stats["intra_mean_std_minutes"]
            ref_inter = ref_stats["inter_std_minutes"]

            rows.extend(
                [
                    _metric_row(
                        model=model_name,
                        metric=(
                            f"{act_name}_weekday_start_intra_std_mean_minutes"
                        ),
                        value=model_intra,
                        unit="minutes",
                        evaluable=model_intra is not None,
                        reason=model_stats.get(
                            "reason", "Insufficient temporal samples."
                        ),
                    ),
                    _metric_row(
                        model=model_name,
                        metric=(
                            f"{act_name}_weekday_start_intra_std_"
                            "abs_error_minutes_vs_actual"
                        ),
                        value=_abs_error(model_intra, ref_intra),
                        unit="minutes",
                        evaluable=(
                            model_intra is not None and ref_intra is not None
                        ),
                        reason=(
                            "Missing model or actual intra-person"
                            " start-time spread."
                        ),
                    ),
                    _metric_row(
                        model=model_name,
                        metric=f"{act_name}_weekday_start_inter_std_minutes",
                        value=model_inter,
                        unit="minutes",
                        evaluable=model_inter is not None,
                        reason=model_stats.get(
                            "reason", "Insufficient temporal samples."
                        ),
                    ),
                    _metric_row(
                        model=model_name,
                        metric=(
                            f"{act_name}_weekday_start_inter_"
                            "std_abs_error_minutes_vs_actual"
                        ),
                        value=_abs_error(model_inter, ref_inter),
                        unit="minutes",
                        evaluable=(
                            model_inter is not None
                            and ref_inter is not None
                        ),
                        reason=(
                            "Missing model or actual inter-person start-time"
                            " spread."
                        ),
                    ),
                    _metric_row(
                        model=model_name,
                        metric=f"{act_name}_weekday_start_samples",
                        value=model_stats.get("weekday_start_samples", 0),
                        unit="samples",
                        evaluable=bool(
                            model_stats.get("weekday_start_samples", 0) > 0
                        ),
                        reason=model_stats.get(
                            "reason", "No weekday start-time samples."
                        ),
                    ),
                ]
            )

    return pd.DataFrame(rows)


def build_notebook_evaluation_tables(
    actual_sf_df: pd.DataFrame,
    actual_all_df: pd.DataFrame,
    model_frames: Dict[str, pd.DataFrame],
    temporal_activity_codes: Iterable[int] = (2, 4),
    state_codes: Iterable[int] = (1, 2, 3, 4, 5, 7),
    trips_clip_max: int = 12,
    loc_type_labels: Optional[Dict[int, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Convenience bundle used by the canonical evaluation notebook."""
    if loc_type_labels is None:
        loc_type_labels = SIM_LOC_TYPES
    return {
        "pattern": build_pattern_metrics_table(
            actual_sf_df=actual_sf_df,
            model_frames=model_frames,
            state_codes=state_codes,
            loc_type_labels=loc_type_labels,
        ),
        "trip": build_trip_metrics_table(
            actual_sf_df=actual_sf_df,
            model_frames=model_frames,
            state_codes=state_codes,
            loc_type_labels=loc_type_labels,
        ),
        "chain": build_chain_metrics_table(
            actual_sf_df=actual_sf_df,
            actual_all_df=actual_all_df,
            model_frames=model_frames,
            loc_type_labels=loc_type_labels,
        ),
        "temporal_dwell_trips": build_temporal_dwell_trip_table(
            actual_sf_df=actual_sf_df,
            model_frames=model_frames,
            temporal_activity_codes=temporal_activity_codes,
            state_codes=state_codes,
            trips_clip_max=trips_clip_max,
            loc_type_labels=loc_type_labels,
        ),
    }


def _resolve_loc_type_names(
    loc_type_labels=None,
    new_mappings=False,
    sim_mappings=True,
    common_mappings=False,
    newest_mappings=False,
):
    """
    Resolve the code -> display-name mapping.
    """
    if loc_type_labels is not None:
        return loc_type_labels
    if new_mappings is True:
        return NEW_LOC_TYPES
    if common_mappings is True:
        return COMMON_LOC_TYPES
    # if newest_mappings is True:
    #     return NEWEST_LOC_TYPES_SHORTER
    if sim_mappings is True:
        return SIM_LOC_TYPES
    return NHTS_LOC_TYPES


def _normalize_loc_dict(num_items, loc_dict):
    """
    Accept:
      - None -> {0: 0, 1: 1, ...}
      - list/tuple -> item idx -> subplot idx
      - dict -> item idx -> subplot idx
    """
    if loc_dict is None:
        return {i: i for i in range(num_items)}

    if isinstance(loc_dict, dict):
        return {int(k): int(v) for k, v in loc_dict.items()}

    return {i: int(v) for i, v in enumerate(loc_dict)}


def _make_fallback_sector_colors(labels, cmap_name="tab20"):
    """
    Fallback stable label -> color mapping.
    """
    cmap = plt.get_cmap(cmap_name)
    return {label: cmap(i % cmap.N) for i, label in enumerate(labels)}


def _make_semantic_sector_colors(labels, cmap_name="tab20"):
    """
    Preferred paired semantic palette so the chord plot visually matches
    the style of your other blue/orange plots without reusing
    Actual/Sim meaning.

    Default mapping:
      Home / Work        -> blue pair
      Restaurant / School -> orange pair
      Recreation / Errands -> green pair
    """
    semantic = {
        "Home": "#1f77b4",        # dark blue
        "Work": "#aec7e8",        # light blue
        "Restaurant": "#ff7f0e",  # dark orange
        "School": "#ffbb78",      # light orange
        "Recreation": "#2ca02c",  # dark green
        "Errands": "#98df8a",     # light green
    }

    if all(label in semantic for label in labels):
        return {label: semantic[label] for label in labels}

    return _make_fallback_sector_colors(labels, cmap_name=cmap_name)


def _extract_present_codes(df, loc_type_key):
    """
    Unique sorted codes present in df[loc_type_key].
    """
    return sorted(set(chain.from_iterable(df[loc_type_key].dropna().values)))


def _build_transition_matrix(df, dnames, loc_type_key="loc_type"):
    """
    Build:
      - transition matrix M (label x label)
      - outgoing percentages per label
      - present labels in order
    """
    transitions = []

    for seq in df[loc_type_key].dropna().values:
        if len(seq) < 2:
            continue
        transitions.extend(zip(seq[:-1], seq[1:]))

    if not transitions:
        raise ValueError(f"No transitions found in column '{loc_type_key}'.")

    pair_counts = Counter(transitions)
    present_codes = _extract_present_codes(df, loc_type_key)
    present_labels = [dnames[c] for c in present_codes]

    M = pd.DataFrame(
        0.0,
        index=present_labels,
        columns=present_labels,
    )

    for (src, dst), n in pair_counts.items():
        M.loc[dnames[src], dnames[dst]] = n

    from_totals = Counter()
    total_transitions = sum(pair_counts.values())

    for (src, _dst), n in pair_counts.items():
        from_totals[dnames[src]] += n

    percentages = {
        label: (100.0 * from_totals.get(label, 0) / total_transitions)
        for label in present_labels
    }

    return M, percentages, present_labels


def _format_sector_label(label, pct, label_mode):
    """
    label_mode:
      - 'none'
      - 'percent'
      - 'name'
      - 'name_percent'
    """
    if label_mode == "none":
        return None
    if label_mode == "percent":
        return f"{pct:.1f}%"
    if label_mode == "name":
        return label
    if label_mode == "name_percent":
        return f"{label}\n({pct:.1f}%)"
    raise ValueError(
        "label_mode must be one of: 'none', 'percent', 'name', 'name_percent'"
    )


def get_chord_diagram_better(
    df,
    loc_type_key="loc_type",
    loc_type_labels=None,
    new_mappings=False,
    sim_mappings=True,
    common_mappings=False,
    newest_mappings=False,
    font_size=12,
    ring_label_font_size_multiplier=0.7,
    label_mode="percent",
    label_position="inside",         # 'inside' or 'outside'
    min_pct_for_ring_label=0.0,      # try 4.0 or 4.5 if you want cleaner plots
    space=5,
    r_lim=(90, 100),
    sector_colors=None,
    cmap_name="tab20",
    link_kws=None,
    ring_label_color="black",
    ax=None,
    only_return_matrix=False,
):
    _require_circos()
    """
    Improved chord-diagram builder.

    Main features:
      - common semantic coloring
      - compact ring labels
      - separate ring label font scaling
      - legend-friendly labels (names stay in legend,
            ring can show only percentages)
    """
    dnames = _resolve_loc_type_names(
        loc_type_labels=loc_type_labels,
        new_mappings=new_mappings,
        sim_mappings=sim_mappings,
        common_mappings=common_mappings,
        newest_mappings=newest_mappings,
    )

    M, percentages, present_labels = _build_transition_matrix(
        df=df,
        dnames=dnames,
        loc_type_key=loc_type_key,
    )

    if only_return_matrix:
        return M

    if sector_colors is None:
        sector_colors = _make_semantic_sector_colors(
            present_labels, cmap_name=cmap_name
        )
    else:
        sector_colors = {
            label: sector_colors[label] for label in present_labels
        }

    if link_kws is None:
        link_kws = dict(ec="black", lw=0.05, direction=1)

    # Hide default pyCirclize labels and add our own compact labels
    circos = Circos.chord_diagram(
        M,
        space=space,
        r_lim=r_lim,
        cmap=sector_colors,
        label_kws=dict(size=0, color="none"),
        link_kws=link_kws,
    )

    ring_label_font_size = font_size * ring_label_font_size_multiplier

    if label_position == "inside":
        label_r = (r_lim[0] + r_lim[1]) / 2.0
    elif label_position == "outside":
        label_r = r_lim[1] + 4
    else:
        raise ValueError("label_position must be 'inside' or 'outside'.")

    for sector in circos.sectors:
        label = sector.name
        pct = percentages.get(label, 0.0)

        if pct < min_pct_for_ring_label:
            continue

        text = _format_sector_label(label, pct, label_mode)
        if text is None:
            continue

        sector.text(
            text,
            r=label_r,
            adjust_rotation=True,
            orientation="horizontal",
            size=ring_label_font_size,
            color=ring_label_color,
            ha="center",
            va="center",
            clip_on=False,
        )

    if ax is not None:
        circos.plotfig(ax=ax)
    else:
        circos.plotfig(figsize=(8, 8))

    return M, percentages, circos


def plot_chord_diagrams_better(
    data_dict,
    loc_dict=None,
    figsize=(18, 12),
    loc_type_labels=None,
    loc_type_key="loc_type",
    font_size=12,
    ring_label_font_size_multiplier=0.7,
    label_mode="percent",
    label_position="inside",
    min_pct_for_ring_label=0.0,
    sector_colors=None,
    cmap_name="tab20",
    space=5,
    r_lim=(90, 100),
    link_kws=None,
    ring_label_color="black",
    # Shared legend options
    legend=True,
    legend_title=None,
    legend_ncol=None,
    legend_loc="upper center",
    legend_bbox_to_anchor=(0.5, 0.995),
    legend_frameon=True,
    legend_fancybox=True,
    legend_framealpha=0.95,
    legend_edgecolor="0.8",
):
    """
    Multi-panel chord plot with one shared legend at the top.

    Design choices for consistency with your histogram plot:
      - legend at top center
      - framed legend box
      - legend font size stays == font_size
      - ring labels use separate scaled font size
      - semantic paired category colors
    """
    num_items = len(data_dict)
    pos_map = _normalize_loc_dict(num_items, loc_dict)

    n_slots = max(pos_map.values()) + 1
    n_cols = math.ceil(math.sqrt(n_slots))
    n_rows = math.ceil(n_slots / n_cols)

    fig, axarr = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        subplot_kw={"projection": "polar"},
    )
    axarr = np.atleast_1d(axarr).ravel()

    dnames = _resolve_loc_type_names(
        loc_type_labels=loc_type_labels,
        sim_mappings=True,
    )

    all_present_codes = sorted({
        code
        for df in data_dict.values()
        for seq in df[loc_type_key].dropna().values
        for code in seq
    })
    all_present_labels = [dnames[c] for c in all_present_codes]

    if sector_colors is None:
        sector_colors = _make_semantic_sector_colors(
            all_present_labels, cmap_name=cmap_name
        )

    if legend_ncol is None:
        legend_ncol = min(len(all_present_labels), 6)

    used_positions = set()
    circos_objects = {}

    for idx, (title, data) in enumerate(data_dict.items()):
        subplot_idx = pos_map[idx]
        used_positions.add(subplot_idx)

        ax = axarr[subplot_idx]
        ax.set_title(title, fontsize=font_size, pad=18)

        _, _, circos = get_chord_diagram_better(
            df=data,
            loc_type_key=loc_type_key,
            loc_type_labels=loc_type_labels,
            font_size=font_size,
            ring_label_font_size_multiplier=ring_label_font_size_multiplier,
            label_mode=label_mode,
            label_position=label_position,
            min_pct_for_ring_label=min_pct_for_ring_label,
            space=space,
            r_lim=r_lim,
            sector_colors=sector_colors,
            cmap_name=cmap_name,
            link_kws=link_kws,
            ring_label_color=ring_label_color,
            ax=ax,
        )
        circos_objects[title] = circos

    for j, ax in enumerate(axarr):
        if j not in used_positions:
            fig.delaxes(ax)

    if legend and all_present_labels:
        handles = [
            Patch(
                facecolor=sector_colors[label],
                edgecolor="black",
                linewidth=0.4,
                label=label,
            )
            for label in all_present_labels
        ]

        leg = fig.legend(
            handles=handles,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            ncol=legend_ncol,
            fontsize=font_size,
            title=legend_title,
            frameon=legend_frameon,
            fancybox=legend_fancybox,
            framealpha=legend_framealpha,
        )

        if legend_edgecolor is not None:
            leg.get_frame().set_edgecolor(legend_edgecolor)

        if leg.get_title() is not None:
            leg.get_title().set_fontsize(font_size)

        # Reserve space at the top for the framed legend box
        plt.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        plt.tight_layout()

    plt.show()
    return fig, axarr, circos_objects


def _scenario_coerce_sequence(seq_like: Any) -> List[Any]:
    if isinstance(seq_like, list):
        return seq_like
    if isinstance(seq_like, tuple):
        return list(seq_like)
    if isinstance(seq_like, np.ndarray):
        return seq_like.tolist()
    if isinstance(seq_like, pd.Series):
        return seq_like.tolist()
    if pd.isna(seq_like):
        return []
    return [seq_like]


def _scenario_compress_adjacent(seq_like: Any) -> List[Any]:
    out: List[Any] = []
    for item in _scenario_coerce_sequence(seq_like):
        if not out or item != out[-1]:
            out.append(item)
    return out


def _scenario_count_locations(value: Any) -> int:
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return int(value)
    return len(_scenario_compress_adjacent(value))


def _scenario_weekday_stop_means(
    df: pd.DataFrame,
    day_order: List[str],
    sequence_col: str = "location",
) -> pd.Series:
    t = df.copy()
    t = t[t["date"].notna()].copy()
    t["day_name"] = t["date"].dt.day_name()
    source_col = sequence_col if sequence_col in t.columns else "loc_type"
    if source_col == "location":
        t["stop_count"] = t[source_col].apply(_scenario_count_locations)
    else:
        t["stop_count"] = t[source_col].apply(
            lambda seq: len(_scenario_compress_adjacent(seq))
        )
    return t.groupby("day_name")["stop_count"].mean().reindex(day_order)


def _scenario_weekday_activity_mix(
    df: pd.DataFrame,
    weekday_name: str,
    activity_codes: List[int],
    home_code: int = 1,
) -> pd.Series:
    t = df.copy()
    t = t[t["date"].notna()].copy()
    t["day_name"] = t["date"].dt.day_name()
    t = t[t["day_name"] == weekday_name].copy()

    rows: List[Dict[int, int]] = []
    for seq in t["loc_type"]:
        compressed = [
            int(code)
            for code in _scenario_compress_adjacent(seq)
            if int(code) != home_code
        ]
        row = {code: 0 for code in activity_codes}
        for code in compressed:
            if code in row:
                row[code] += 1
        rows.append(row)

    if not rows:
        return pd.Series(0.0, index=activity_codes)
    return pd.DataFrame(rows).mean().reindex(activity_codes).fillna(0.0)


def _scenario_format_work_share(share: float) -> str:
    pct = share * 100
    if pct >= 10:
        return f"{pct:.0f}%"
    return f"{pct:.1f}%"


def make_weekday_location_figure(
    frames: Dict[str, pd.DataFrame],
    model_order: List[str],
    day_order: List[str],
    day_short_labels: Dict[str, str],
    storm_day_names: Iterable[str],
    model_colors: Dict[str, str],
    model_display: Dict[str, str],
    font_size: int = 24,
    font_size_s: int = 24,
    legend_ncol: int = 4,
    legend_loc: str = "upper center",
    legend_bbox_to_anchor: Tuple[float, float] = (0.5, 1.18),
    figsize: Tuple[int, int] = (15, 8),
    sequence_col: str = "location",
    x_label: str = "Simulation day",
    y_label: str = "Average visited locations per agent-day",
    storm_label: str = "storm",
    storm_fill_color: str = "#dbeafe",
    storm_label_color: str = "#1d4ed8",
):
    summary = {
        name: _scenario_weekday_stop_means(
            frames[name],
            day_order=day_order,
            sequence_col=sequence_col,
        )
        for name in model_order
    }
    x = np.arange(len(day_order))
    fig, ax = plt.subplots(figsize=figsize, dpi=300, constrained_layout=True)

    for idx, day_name in enumerate(day_order):
        if day_name in storm_day_names:
            ax.axvspan(
                idx - 0.45,
                idx + 0.45,
                color=storm_fill_color,
                alpha=0.45,
                zorder=0,
            )

    for model_name in model_order:
        ax.plot(
            x,
            summary[model_name].values,
            marker="o",
            linewidth=2.4,
            markersize=6,
            color=model_colors[model_name],
            label=model_display[model_name],
        )

    ax.set_xticks(x)
    ax.set_xticklabels([day_short_labels[d] for d in day_order], fontsize=font_size)
    ax.set_xlabel(x_label, fontsize=font_size, labelpad=32)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.12)
    for idx, day_name in enumerate(day_order):
        if day_name in storm_day_names:
            ax.text(
                idx,
                ymax * 1.05,
                storm_label,
                ha="center",
                va="bottom",
                fontsize=font_size_s,
                color=storm_label_color,
                fontweight="bold",
            )

    ax.legend(
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncol,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#d0d7de",
        fontsize=font_size,
    )
    return fig, ax, summary


def make_weekly_activity_mix_figure(
    frames: Dict[str, pd.DataFrame],
    model_order: List[str],
    day_order: List[str],
    day_short_labels: Dict[str, str],
    storm_day_names: Iterable[str],
    activity_codes: List[int],
    activity_labels: Dict[int, str],
    activity_colors: Dict[int, str],
    model_colors: Dict[str, str],
    model_short: Dict[str, str],
    font_size: int = 24,
    font_size_s: int = 24,
    font_size_ws: int = 16,
    legend_ncol: int = 4,
    legend_loc: str = "center left",
    legend_bbox_to_anchor: Tuple[float, float] = (1.02, 0.5),
    figsize: Tuple[int, int] = (15, 8),
    work_code: int = 2,
    x_label: str = "Simulation day",
    y_label: str = "Mean out-of-home activity per agent",
    storm_label: str = "storm",
    storm_fill_color: str = "#dbeafe",
    storm_label_color: str = "#1d4ed8",
    legend_title: str = "Top % =\nwork-share\nof trips",
):
    weekly_mix = {
        weekday_name: {
            model_name: _scenario_weekday_activity_mix(
                frames[model_name],
                weekday_name,
                activity_codes,
            )
            for model_name in model_order
        }
        for weekday_name in day_order
    }

    fig, ax = plt.subplots(figsize=figsize, dpi=300, constrained_layout=True)
    day_centers = np.arange(len(day_order), dtype=float)
    n_models = max(len(model_order), 1)
    if n_models == 1:
        group_width = 0.42
    elif n_models == 2:
        group_width = 0.92
    elif n_models == 3:
        group_width = 0.90
    else:
        group_width = 0.88

    bar_width = group_width / n_models
    if n_models == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(
            -(group_width - bar_width) / 2,
            (group_width - bar_width) / 2,
            n_models,
        )

    shade_half_width = group_width / 2 + 0.06
    max_total = max(
        weekly_mix[day][model].sum()
        for day in day_order
        for model in model_order
    )

    for idx, day_name in enumerate(day_order):
        if day_name in storm_day_names:
            ax.axvspan(
                day_centers[idx] - shade_half_width,
                day_centers[idx] + shade_half_width,
                color=storm_fill_color,
                alpha=0.45,
                zorder=0,
            )

    for model_idx, model_name in enumerate(model_order):
        x = day_centers + offsets[model_idx]
        bottoms = np.zeros(len(day_order), dtype=float)
        total_vals = np.array(
            [weekly_mix[day][model_name].sum() for day in day_order],
            dtype=float,
        )
        work_vals = np.array(
            [weekly_mix[day][model_name].loc[work_code] for day in day_order],
            dtype=float,
        )

        for code in activity_codes:
            vals = np.array(
                [weekly_mix[day][model_name].loc[code] for day in day_order],
                dtype=float,
            )
            ax.bar(
                x,
                vals,
                width=bar_width,
                bottom=bottoms,
                color=activity_colors[code],
                edgecolor="white",
                linewidth=0.5,
                label=activity_labels[code] if model_idx == 0 else None,
            )
            bottoms += vals

        for day_idx, xi in enumerate(x):
            ax.text(
                xi,
                -max_total * 0.145,
                model_short[model_name],
                ha="center",
                va="bottom",
                fontsize=font_size_s,
                color=model_colors[model_name],
                rotation=90,
                clip_on=True,
            )
            work_share = (
                work_vals[day_idx] / total_vals[day_idx]
                if total_vals[day_idx] > 0
                else 0.0
            )
            ax.text(
                xi,
                total_vals[day_idx] + max_total * 0.03,
                _scenario_format_work_share(work_share),
                ha="center",
                va="bottom",
                fontsize=font_size_ws,
                color=activity_colors[work_code],
                fontweight="bold",
            )

    ax.set_xticks(day_centers)
    ax.set_xticklabels([day_short_labels[d] for d in day_order], fontsize=font_size)
    ax.set_xlabel(x_label, fontsize=font_size, labelpad=14)
    ax.set_ylabel(y_label, fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_xlim(day_centers[0] - 0.6, day_centers[-1] + 0.6)
    ax.set_ylim(-max_total * 0.16, max_total * 1.16)
    yticks = np.arange(0, max_total * 1.16 + 0.001, 0.5)
    ax.set_yticks(yticks)

    for idx, day_name in enumerate(day_order):
        if day_name in storm_day_names:
            ax.text(
                day_centers[idx],
                max_total * 1.06,
                storm_label,
                ha="center",
                va="bottom",
                fontsize=font_size_s,
                color=storm_label_color,
                fontweight="bold",
            )

    legend = ax.legend(
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=legend_ncol,
        title=legend_title,
        title_fontsize=font_size,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#d0d7de",
        fontsize=font_size,
    )
    legend.get_title().set_multialignment("left")
    legend.get_title().set_fontweight("normal")
    legend._legend_box.align = "left"
    return fig, ax, weekly_mix

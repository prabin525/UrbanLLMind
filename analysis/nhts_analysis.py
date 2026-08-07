"""
Script: NHTS Analysis
Description:
    Processes raw NHTS 2017 CSV data to generate ground truth metrics for the
    simulation region (CBSA 41860). It calculates Activity participation rates,
    top activity patterns, transition matrices, and typical dwell times.
    Serves as the "Golden Standard" for calibrating agent behavior.

Run:
    python analysis/nhts_analysis.py

Output:
    Prints metrics to stdout.
"""
import sys
import pandas as pd
import numpy as np
import yaml
from collections import Counter, defaultdict
sys.path.insert(0, "analysis")
from constants import NHTS_TO_SIM_ACTIVITY  # noqa: E402

NEED_KEYS = ["work", "food", "social", "errands", "rest"]
ACTIVITY_TO_NEED = {
    1: "rest", 2: "work", 3: "food", 4: "work", 5: "social", 7: "errands"
}
NEED_INDEX = {k: i for i, k in enumerate(NEED_KEYS)}


def to_num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")


def parse_time(v):
    try:
        i = int(v)
    except Exception:
        return None
    if i < 0:
        return None
    h, m = i // 100, i % 100
    if h == 24 and m == 0:
        return 1440
    if h > 24 or m > 59:
        return None
    return h * 60 + m


# Load data
per_cols = ["HOUSEID", "PERSONID", "HH_CBSA", "WORKER", "SCHTYP"]
trip_cols = [
    "HOUSEID", "PERSONID", "HH_CBSA", "TDTRPNUM",
    "WHYFROM", "WHYTO", "STRTTIME", "ENDTIME"
]
perpub = pd.read_csv(
    "dataset/NHTS_2017_csv/perpub.csv", usecols=per_cols, low_memory=False
)
trippub = pd.read_csv(
    "dataset/NHTS_2017_csv/trippub.csv", usecols=trip_cols, low_memory=False
)
to_num(perpub, per_cols)
to_num(trippub, trip_cols)

perpub = perpub[perpub["HH_CBSA"] == 41860].drop_duplicates(
    ["HOUSEID", "PERSONID"]
)
trippub = trippub[trippub["HH_CBSA"] == 41860]
merged = trippub.merge(
    perpub[["HOUSEID", "PERSONID", "WORKER", "SCHTYP"]],
    on=["HOUSEID", "PERSONID"]
)
workers = merged[merged["WORKER"] == 1].sort_values(
    ["HOUSEID", "PERSONID", "STRTTIME"]
)

# Build records
records = []
for (hid, pid), grp in workers.groupby(["HOUSEID", "PERSONID"]):
    grp = grp.sort_values("TDTRPNUM")
    acts, durs = [], []
    first_start = parse_time(grp.iloc[0]["STRTTIME"])
    if first_start is None:
        continue
    if first_start > 0:
        acts.append(1)
        durs.append(float(first_start))
    for _, row in grp.iterrows():
        whyto = int(row["WHYTO"]) if pd.notna(row["WHYTO"]) else -9
        dest = NHTS_TO_SIM_ACTIVITY.get(whyto, 7)
        st, et = parse_time(row["STRTTIME"]), parse_time(row["ENDTIME"])
        if st is None or et is None:
            continue
        dur = et - st
        if dur <= 0:
            continue
        acts.append(int(dest))
        durs.append(float(dur))
    total = sum(durs)
    if total < 1440:
        if acts and acts[-1] == 1:
            durs[-1] += 1440 - total
        else:
            acts.append(1)
            durs.append(1440 - total)
    # Compress
    c_acts, c_durs = [acts[0]], [durs[0]]
    for a, d in zip(acts[1:], durs[1:]):
        if a == c_acts[-1]:
            c_durs[-1] += d
        else:
            c_acts.append(a)
            c_durs.append(d)
    ticks = [max(1, int(round(d / 5))) for d in c_durs]
    records.append({
        "activities": c_acts,
        "durations_minutes": c_durs,
        "durations_ticks": ticks
    })

print(f"Worker records: {len(records)}")
print("=" * 60)

# 1. PARTICIPATION
print("\n--- Participation ---")
for code, name in [
    (1, "Home"), (2, "Work"), (3, "Food"), (5, "Recreation"), (7, "Errands")
]:
    n = sum(1 for r in records if code in r["activities"])
    print(f"  {name:>12s}: {n/len(records)*100:.1f}%")

# 2. TOP PATTERNS
short = {1: "H", 2: "W", 3: "F", 4: "S", 5: "R", 7: "E"}
chains = [
    "->".join(short.get(a, "?") for a in r["activities"]) for r in records
]
wcounts = Counter(chains)
print(f"\n--- Top 15 Patterns ({len(wcounts)} unique) ---")
cum = 0
for rank, (pat, cnt) in enumerate(wcounts.most_common(15), 1):
    cum += cnt
    print(
        f"  {rank:>2}. {cnt:>4} ({cnt/len(records)*100:.1f}%) "
        f"{pat}  [cum {cum/len(records)*100:.0f}%]"
    )

# 3. TRANSITIONS
print("\n--- Transitions ---")
trans = defaultdict(lambda: defaultdict(int))
for r in records:
    for i in range(len(r["activities"]) - 1):
        trans[r["activities"][i]][r["activities"][i + 1]] += 1
cols = [1, 2, 3, 5, 7]
h = "From  " + "  ".join(f"{short[c]:>5s}" for c in cols)
print(f"  {h}")
for fc in cols:
    rt = sum(trans[fc].values())
    cells = [
        f"{trans[fc][tc]/rt*100:>4.0f}%" if rt else "   0%" for tc in cols
    ]
    print(f"  {short[fc]:>5s} {'  '.join(cells)}  (n={rt})")

# 4. SEQUENCE ALIGNMENT
print("\n--- Sequence Alignment ---")
with open("input_GABM_task9_stageb_calibrated.yaml") as f:
    cfg = yaml.safe_load(f)
cal = cfg["need_calibration"]["worker"]
init_arr = np.array([cal["initial"][k] for k in NEED_KEYS])
growth_arr = np.array([cal["growth_per_tick"][k] for k in NEED_KEYS])
dc = cfg["dwell_calibration"]["worker"]["typical_dwell_minutes"]
DM = {
    1: dc["home"],
    2: dc["work"],
    3: dc["restaurant"],
    4: dc.get("school", 300),
    5: dc["recreation"],
    7: dc["errands"]
}

cb = defaultdict(lambda: [0, 0])
for rec in records:
    needs = init_arr.copy()
    for act, ticks in zip(rec["activities"], rec["durations_ticks"]):
        tgt = ACTIVITY_TO_NEED.get(act)
        ti = NEED_INDEX.get(tgt) if tgt else None
        if ti is not None:
            if np.argmax(needs) == ti:
                cb[tgt][0] += 1
            cb[tgt][1] += 1
        if ti is None:
            needs += growth_arr * ticks
        else:
            for idx in range(5):
                if idx != ti:
                    needs[idx] += growth_arr[idx] * ticks
            tt = max(1, DM.get(act, 60) / 5)
            needs[ti] = max(0, needs[ti] * max(0, 1 - ticks / tt))

tc, tt2 = 0, 0
for need in NEED_KEYS:
    c, t = cb.get(need, [0, 0])
    tc += c
    tt2 += t
    if t:
        print(f"  {need:>8s}: {c:>5d}/{t:>5d} = {c/t*100:.1f}%")
print(f"  {'TOTAL':>8s}: {tc:>5d}/{tt2:>5d} = {tc/tt2*100:.1f}%")

# 5. VARIABILITY
print("\n--- Need Variability ---")
nf = defaultdict(list)
for rec in records:
    tot = sum(rec["durations_minutes"])
    nm = {k: 0.0 for k in NEED_KEYS}
    for a, d in zip(rec["activities"], rec["durations_minutes"]):
        tgt = ACTIVITY_TO_NEED.get(a)
        if tgt:
            nm[tgt] += d
    for k in NEED_KEYS:
        nf[k].append(nm[k] / tot)

ji = cal.get("jitter_std_fraction", {}).get("initial", {})
jg = cal.get("jitter_std_fraction", {}).get("growth_per_tick", {})
print(
    f"  {'Need':>8s}  {'Mean%':>6s}  {'Std%':>6s}  {'CV':>5s}  "
    f"{'Zero%':>6s}  JitI  JitG"
)
for need in NEED_KEYS:
    v = np.array(nf[need])
    mn, sd = np.mean(v) * 100, np.std(v) * 100
    cv = np.std(v) / max(np.mean(v), 1e-6)
    z = np.sum(v < 0.001) / len(v) * 100
    print(
        f"  {need:>8s}  {mn:>5.1f}%  {sd:>5.1f}%  {cv:>4.2f}  "
        f"{z:>5.1f}%  {ji.get(need, 0):.3f} {jg.get(need, 0):.3f}"
    )

# 6. CURRENT VALUES
print("\n--- Current Calibration ---")
af = {
    "rest": "home", "work": "work", "food": "restaurant",
    "social": "recreation", "errands": "errands"
}
for need in NEED_KEYS:
    dw = dc.get(af[need], 0)
    print(
        f"  {need:>8s}: init={cal['initial'][need]:.4f}  "
        f"growth={cal['growth_per_tick'][need]:.6f}  dwell={dw}min"
    )

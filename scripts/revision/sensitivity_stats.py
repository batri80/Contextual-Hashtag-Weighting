"""
sensitivity_stats.py
====================
Significance tests for the sensitivity analysis (Section 5.5, Tables 6-7).

Each setting is compared with the default (alpha = 0.33 each, tau_dormant = 30,
tau_decay = 7) by Welch's t-test on the five per-seed Silhouette scores, with
Holm correction within each experiment, followed by a one-way ANOVA across all
settings of that experiment.

Usage:
    python sensitivity_stats.py --runs results/election2020/sensitivity_runs.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="results/election2020/sensitivity_runs.csv")
args = ap.parse_args()

runs = pd.read_csv(args.runs)
keys = ["a1", "a2", "a3", "tau_dormant", "tau_decay"]
DEFAULT = (0.33, 0.33, 0.33, 30.0, 7.0)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


for exp, df in runs.groupby("experiment"):
    groups = {k: g["silhouette"].to_numpy() for k, g in df.groupby(keys)}
    default = next(v for k, v in groups.items() if np.allclose(k, DEFAULT))
    others = [(k, v) for k, v in groups.items() if not np.allclose(k, DEFAULT)]
    p = [stats.ttest_ind(v, default, equal_var=False).pvalue for _, v in others]
    p_adj = holm(p)
    means = [v.mean() for v in groups.values()]
    print(f"\n{exp}: {len(groups)} settings, five seeds each")
    print(f"  mean Silhouette range       {min(means):.4f} - {max(means):.4f}")
    print(f"  default                     {default.mean():.4f} +/- {default.std(ddof=1):.4f} (sample s.d.)")
    print(f"  smallest unadjusted Welch p {min(p):.3f}")
    print(f"  Holm-adjusted p range       {p_adj.min():.2f} - {p_adj.max():.2f}")
    print(f"  one-way ANOVA p             {stats.f_oneway(*groups.values()).pvalue:.2f}")

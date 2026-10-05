"""
table_checks.py
Prints the K-grid metrics behind Table 2 and the statistical tests behind
Table 4 (Shapiro-Wilk, paired t, Welch t, Mann-Whitney U).

Run once in each dataset folder:
    Election 2020:  python table_checks.py
    COVID-19:       python table_checks.py --results covid19_results_archive/results

It reads (if present):
    <results>/enhanced_clustering_results.csv   -> Silhouette, DB and CH for Table 2
    <results>/bert_comparison_k20.pkl           -> tests for Table 4
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=".")
a = ap.parse_args()

csv = os.path.join(a.results, "enhanced_clustering_results.csv")
if os.path.exists(csv):
    d = pd.read_csv(csv)
    d = d[(d["Features"] == "Hashtag-Only") & (d["Algorithm"] == "KMeans")].sort_values("K")
    print("Table 2 (Hashtag-Only, K-means):")
    print(d[["K", "Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]].to_string(index=False))
else:
    print(f"not found: {csv}")

pkl = os.path.join(a.results, "bert_comparison_k20.pkl")
if os.path.exists(pkl):
    b = pickle.load(open(pkl, "rb"))
    h = np.array(b["hashtag"]["scores"], float)
    t = np.array(b["bert"]["scores"], float)
    print("\nTable 4 (K=20):")
    print(f"  Hashtag-Only {h.mean():.4f} +/- {h.std(ddof=1):.4f}   BERT {t.mean():.4f} +/- {t.std(ddof=1):.4f}")
    print(f"  Shapiro-Wilk p: hashtag {stats.shapiro(h).pvalue:.2f}, BERT {stats.shapiro(t).pvalue:.2f}")
    print(f"  Paired t p = {stats.ttest_rel(h, t).pvalue:.2e}")
    print(f"  Welch t  p = {stats.ttest_ind(h, t, equal_var=False).pvalue:.2e}")
    print(f"  Mann-Whitney p = {stats.mannwhitneyu(h, t, alternative='two-sided').pvalue:.4f}")
else:
    print(f"not found: {pkl}")

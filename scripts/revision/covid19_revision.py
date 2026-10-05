"""
covid19_revision.py
===================
COVID-19 numbers for the SNAM response, using the ORIGINAL pipeline outputs.

Run inside the COVID-19 folder that holds:
    train_with_weights.pkl, val_with_weights.pkl, test_with_weights.pkl
    test_hashtag.npz, test_tfidf.npz            (needed for part 2)

Usage:
    python covid19_revision.py              # part 1 + part 2
    python covid19_revision.py --no-ablation  # part 1 only (a few seconds)

Part 1  Comments 1 and 2: split sizes, periods, vocabulary size, test tweets with no
        vocabulary hashtag.
Part 2  Editor (COVID-19 ablation): same protocol as ablation_study.py on Election 2020 -
        L2-normalised hashtag and TF-IDF features, hashtag weight alpha in
        {1.0, 0.75, 0.5, 0.25, 0.0}, K = 20, 3 runs per setting (seeds 42-44).

Outputs (folder covid19_revision/): split_stats.csv, ablation_covid19.csv
"""

import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

ap = argparse.ArgumentParser()
ap.add_argument("--no-ablation", action="store_true")
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--runs", type=int, default=3)
args = ap.parse_args()
OUT = "covid19_revision"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Part 1: split sizes and vocabulary
# ---------------------------------------------------------------------------
print("=" * 70)
print("PART 1  Split sizes and vocabulary (Comments 1 and 2)")
print("=" * 70)
splits = {s: pd.read_pickle(f"{s}_with_weights.pkl") for s in ["train", "val", "test"]}
total = sum(len(d) for d in splits.values())
rows = []
for s, d in splits.items():
    rows.append({"split": s, "tweets": len(d), "percent": round(100 * len(d) / total, 1),
                 "start": d["created_at"].min(), "end": d["created_at"].max()})
stats_df = pd.DataFrame(rows)
print(stats_df.to_string(index=False))
print(f"Total after filtering: {total:,}")

freq = Counter(h for hs in splits["train"]["hashtags"] for h in hs)
vocab = {h for h, f in freq.items() if f >= 10}
no_vocab = int(sum(1 for hs in splits["test"]["hashtags"] if not any(h in vocab for h in hs)))
print(f"Hashtag vocabulary (>=10 training occurrences): {len(vocab):,}")
print(f"Test tweets with no vocabulary hashtag: {no_vocab:,} of {len(splits['test']):,}")
stats_df.loc[len(stats_df)] = {"split": "vocabulary", "tweets": len(vocab)}
stats_df.loc[len(stats_df)] = {"split": "test_no_vocab_hashtag", "tweets": no_vocab}
stats_df.to_csv(f"{OUT}/split_stats.csv", index=False)

if args.no_ablation:
    print(f"\nSaved {OUT}/split_stats.csv")
    raise SystemExit

# ---------------------------------------------------------------------------
# Part 2: ablation (same protocol as ablation_study.py)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"PART 2  Hashtag / TF-IDF ablation, K={args.k}, {args.runs} runs per setting")
print("=" * 70)
H = normalize(sparse.load_npz("test_hashtag.npz"), norm="l2")
T = normalize(sparse.load_npz("test_tfidf.npz"), norm="l2")
assert H.shape[0] == T.shape[0] == len(splits["test"]), "feature rows do not match test set"

res = []
for alpha in [1.0, 0.75, 0.5, 0.25, 0.0]:
    X = sparse.hstack([alpha * H, (1 - alpha) * T]).tocsr()
    scores = []
    for r in range(args.runs):
        lab = KMeans(n_clusters=args.k, random_state=42 + r, n_init=10).fit_predict(X)
        scores.append(silhouette_score(X, lab, sample_size=min(10000, X.shape[0]),
                                       random_state=42 + r))
    res.append({"hashtag_weight": alpha, "tfidf_weight": round(1 - alpha, 2),
                "silhouette_mean": np.mean(scores), "silhouette_sd": np.std(scores, ddof=1),
                "runs": " ".join(f"{s:.4f}" for s in scores)})
    print(f"  {int(alpha*100):3d}% hashtag + {int(round((1-alpha)*100)):3d}% TF-IDF   "
          f"Silhouette = {np.mean(scores):.4f} +/- {np.std(scores, ddof=1):.4f}")

pd.DataFrame(res).to_csv(f"{OUT}/ablation_covid19.csv", index=False)
print(f"\nSaved {OUT}/split_stats.csv and {OUT}/ablation_covid19.csv")

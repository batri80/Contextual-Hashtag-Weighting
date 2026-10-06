"""
idf_baseline.py
===============
Frequency-weighted (IDF) hashtag baseline for Section 5.5.

The introduction contrasts frequency-based hashtag weighting with the contextual
(energy) weights. This script tests that contrast directly, on the same hashtag
vocabulary and with the same protocol as revision_diagnostics.py
(K-means, K = 20, n_init = 10, 300 iterations, seeds 42-46, Silhouette on a
10,000-point sample seeded with the K-means seed).

Each hashtag h in the vocabulary gets idf(h) = ln(N / (1 + df(h))) + 1, where N is
the number of training tweets and df(h) the number of training tweets containing h
(computed from train_hashtag.npz, so the vocabulary and columns are identical to
the energy-weighted features). For each test tweet, the hashtags it contains are
then weighted by

  "idf, per-tweet normalised"  idf(h) / sum of idf over the tweet's hashtags
                               (sums to 1, like the energy and uniform weights)
  "idf, L2-normalised"         idf(h), with the row scaled to unit L2 norm
                               (the usual TF-IDF convention)

The energy and uniform variants are re-run in the same call for a like-for-like
comparison, and Welch's t-test compares every variant with the energy weights.

Run inside the dataset folder that holds train_hashtag.npz and test_hashtag.npz:
    python idf_baseline.py --name election2020

Output: idf_baseline_<name>.csv (per-seed scores) and a summary on the console.
"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="dataset")
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--seeds", type=int, default=5)
args = ap.parse_args()

train = sparse.load_npz("train_hashtag.npz").tocsc()
X = sparse.load_npz("test_hashtag.npz").tocsr().astype(np.float64)
if train.shape[1] != X.shape[1]:
    raise SystemExit(f"Column mismatch: train {train.shape[1]} vs test {X.shape[1]}")

n_train = train.shape[0]
df = np.diff((train != 0).astype(np.int8).tocsc().indptr)  # training tweets per hashtag
idf = np.log(n_train / (1.0 + df)) + 1.0
print(f"Training tweets {n_train:,}; vocabulary {X.shape[1]:,}; "
      f"idf range {idf.min():.2f}-{idf.max():.2f}")

nnz = np.diff(X.indptr)
Xu = X.copy()
Xu.data = np.repeat(1.0 / np.maximum(nnz, 1), nnz).astype(np.float64)

Xi = X.copy()
Xi.data = idf[Xi.indices].astype(np.float64)
Xi_sum = normalize(Xi, norm="l1")
Xi_l2 = normalize(Xi, norm="l2")

variants = {
    "energy (as submitted)": X,
    "uniform 1/|H|": Xu,
    "idf, per-tweet normalised": Xi_sum,
    "idf, L2-normalised": Xi_l2,
}

rows = []
for name, M in variants.items():
    for s in range(args.seeds):
        seed = 42 + s
        t0 = time.time()
        labels = KMeans(n_clusters=args.k, n_init=10, max_iter=300,
                        random_state=seed).fit_predict(M)
        sil = silhouette_score(M, labels, sample_size=min(10000, M.shape[0]), random_state=seed)
        rows.append({"variant": name, "seed": seed, "silhouette": sil,
                     "kmeans_seconds": time.time() - t0})
        print(f"{name:28s} seed={seed} sil={sil:.4f}")

res = pd.DataFrame(rows)
res.to_csv(f"idf_baseline_{args.name}.csv", index=False)

energy = res.loc[res.variant == "energy (as submitted)", "silhouette"].to_numpy()
print(f"\nSummary ({args.name}, K={args.k}, seeds 42-{41 + args.seeds}; sample s.d.)")
for name in variants:
    v = res.loc[res.variant == name, "silhouette"].to_numpy()
    p = "" if name.startswith("energy") else f"   Welch p vs energy = {stats.ttest_ind(v, energy, equal_var=False).pvalue:.3g}"
    print(f"  {name:28s} {v.mean():.4f} +/- {v.std(ddof=1):.4f}{p}")
print(f"\nSaved idf_baseline_{args.name}.csv")

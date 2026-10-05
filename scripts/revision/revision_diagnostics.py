"""
revision_diagnostics.py
=======================
Revision checks for "Contextual Hashtag Weighting for Social Media Clustering"
(SNAM revision). Run once per dataset, inside the folder that holds that
dataset's Phase 1-4 outputs:

    train_with_weights.pkl, val_with_weights.pkl, test_with_weights.pkl
    hashtag_vocab.pkl, temporal_data.pkl, test_hashtag.npz

Usage:
    python revision_diagnostics.py --name election2020
    python revision_diagnostics.py --name covid19

Outputs:
    diagnostics_<name>.json   - split sizes, duplicates, matrix structure, temporal stats
    diagnostics_<name>.csv    - silhouette for weighting variants (energy / uniform / binary,
                                with and without empty and duplicate rows)

Nothing here modifies existing files.
"""

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="dataset")
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--seeds", type=int, default=5)
args = ap.parse_args()

out = {"dataset": args.name, "K": args.k}


def col(df, c):
    """Numeric column or zeros if the column is absent."""
    if c in df.columns:
        return pd.to_numeric(df[c], errors="coerce").fillna(0).values
    return np.zeros(len(df))


# ---------------------------------------------------------------------------
# 1. Split sizes and date ranges (Reviewer 1, point 1)
# ---------------------------------------------------------------------------
splits = {s: pd.read_pickle(f"{s}_with_weights.pkl") for s in ["train", "val", "test"]}
total = sum(len(d) for d in splits.values())
for s, df in splits.items():
    out[f"{s}_n"] = len(df)
    out[f"{s}_pct"] = round(100 * len(df) / total, 2)
    out[f"{s}_start"] = str(df["created_at"].min())
    out[f"{s}_end"] = str(df["created_at"].max())
out["total_after_filtering"] = total
test = splits["test"].reset_index(drop=True)

# ---------------------------------------------------------------------------
# 2. Duplicates (paper claims de-duplication; Phase 1 does not do it)
# ---------------------------------------------------------------------------
full = pd.concat(splits.values(), ignore_index=True)
id_col = next((c for c in ["tweet_id", "id", "id_str"] if c in full.columns), None)
if id_col:
    out["duplicate_by_id"] = int(full.duplicated(subset=[id_col]).sum())
if "tweet" in full.columns:
    out["duplicate_by_raw_text"] = int(full.duplicated(subset=["tweet"]).sum())
out["duplicate_by_clean_text"] = int(full.duplicated(subset=["clean_text"]).sum())
if "candidate" in full.columns and "tweet" in full.columns:
    both = full.groupby("tweet")["candidate"].nunique()
    out["tweets_in_both_biden_and_trump_files"] = int((both > 1).sum())

# ---------------------------------------------------------------------------
# 3. Engagement factor Gamma = log(1 + retweets + likes) used in E_temporal
# ---------------------------------------------------------------------------
out["engagement_columns_present"] = [
    c for c in ["retweet_count", "likes", "favorite_count"] if c in test.columns
]
gamma = np.log1p(col(test, "retweet_count") + col(test, "likes"))
out["test_gamma_zero_fraction"] = float((gamma == 0).mean())
out["test_gamma_mean"] = float(gamma.mean())
out["test_gamma_max"] = float(gamma.max())
out["note_gamma"] = ("Phase 2 reads 'likes', not 'favorite_count'. If 'likes' is absent, "
                     "Gamma uses retweets only; if Gamma == 0, E_temporal == 0 for that tweet.")

# ---------------------------------------------------------------------------
# 4. Temporal component on the test set
#    Phase 2 builds history from TRAIN only, so for test tweets
#    delta_t = time since the hashtag's last TRAIN occurrence.
# ---------------------------------------------------------------------------
vocab_data = pickle.load(open("hashtag_vocab.pkl", "rb"))
vocab = vocab_data["vocab"]
if isinstance(vocab, list):
    out["WARNING_vocab"] = ("hashtag_vocab.pkl holds a list - it was overwritten by "
                            "preprocess_clustering.py (anomaly-detection paper).")
    vocab = {h: i for i, h in enumerate(vocab)}
out["vocab_size"] = len(vocab)

temporal = pickle.load(open("temporal_data.pkl", "rb"))
tau = temporal["tau_dormant"]
last_train = {h: max(t) for h, t in temporal["history"].items() if len(t)}
active = dormant = nohist = 0
for ts, tags in zip(test["created_at"], test["hashtags"]):
    t = ts.timestamp()
    for h in tags:
        if h not in vocab:
            continue
        if h not in last_train:
            nohist += 1
        elif t - last_train[h] > tau:
            dormant += 1
        else:
            active += 1
occ = max(active + dormant + nohist, 1)
out["test_hashtag_occurrences_in_vocab"] = occ
out["temporal_active_fraction"] = active / occ
out["temporal_dormant_reset_fraction"] = dormant / occ
out["temporal_no_history_fraction"] = nohist / occ

# ---------------------------------------------------------------------------
# 5. Structure of the hashtag feature matrix
# ---------------------------------------------------------------------------
X = sparse.load_npz("test_hashtag.npz").tocsr().astype(np.float64)
X.sort_indices()
nnz = np.diff(X.indptr)
n = X.shape[0]
out["test_rows"] = n
out["rows_with_zero_vocab_hashtags"] = int((nnz == 0).sum())
out["rows_with_one_vocab_hashtag"] = int((nnz == 1).sum())
out["rows_with_2plus_vocab_hashtags"] = int((nnz >= 2).sum())

keys = [
    (tuple(X.indices[X.indptr[i]:X.indptr[i + 1]]),
     tuple(np.round(X.data[X.indptr[i]:X.indptr[i + 1]], 6)))
    for i in range(n)
]
first_idx = {}
for i, k in enumerate(keys):
    first_idx.setdefault(k, i)
uniq = np.array(sorted(first_idx.values()))
out["unique_rows"] = len(uniq)
out["exact_duplicate_row_fraction"] = 1 - len(uniq) / n

# How far are energy weights from uniform 1/|H_d|?
devs = []
for i in np.where(nnz >= 2)[0]:
    w = X.data[X.indptr[i]:X.indptr[i + 1]]
    devs.append(np.max(np.abs(w - 1.0 / len(w))))
out["mean_max_deviation_from_uniform"] = float(np.mean(devs)) if devs else None
out["p95_max_deviation_from_uniform"] = float(np.percentile(devs, 95)) if devs else None

# ---------------------------------------------------------------------------
# 6. Silhouette for weighting variants at K
# ---------------------------------------------------------------------------
Xu = X.copy()
Xu.data = np.repeat(1.0 / np.maximum(nnz, 1), nnz).astype(np.float64)
Xb = X.copy()
Xb.data[:] = 1.0
nonzero = np.where(nnz > 0)[0]
uniq_nonzero = np.intersect1d(uniq, nonzero)

variants = {
    "energy (as submitted)": X,
    "uniform 1/|H|": Xu,
    "binary": Xb,
    "energy, empty rows removed": X[nonzero],
    "energy, empty+duplicate rows removed": X[uniq_nonzero],
    "uniform, empty+duplicate rows removed": Xu[uniq_nonzero],
}

rows = []
for name, M in variants.items():
    for s in range(args.seeds):
        seed = 42 + s
        t0 = time.time()
        labels = KMeans(n_clusters=args.k, n_init=10, max_iter=300,
                        random_state=seed).fit_predict(M)
        fit_t = time.time() - t0
        sil = silhouette_score(M, labels, sample_size=min(10000, M.shape[0]),
                               random_state=seed)
        rows.append({"variant": name, "n": M.shape[0], "seed": seed,
                     "silhouette": sil, "kmeans_seconds": fit_t})
        print(f"{name:42s} seed={seed} n={M.shape[0]:>7,} sil={sil:.4f} t={fit_t:.1f}s")

res = pd.DataFrame(rows)
summary = res.groupby("variant").agg(n=("n", "first"),
                                     sil_mean=("silhouette", "mean"),
                                     sil_sd=("silhouette", lambda x: x.std(ddof=1)),
                                     kmeans_s=("kmeans_seconds", "mean"))
print("\n", summary.round(4).to_string())

res.to_csv(f"diagnostics_{args.name}.csv", index=False)
out["silhouette_summary"] = summary.round(4).reset_index().to_dict(orient="records")
with open(f"diagnostics_{args.name}.json", "w") as f:
    json.dump(out, f, indent=2, default=str)

print("\n" + json.dumps({k: v for k, v in out.items() if k != "silhouette_summary"},
                        indent=2, default=str))
print(f"\nSaved diagnostics_{args.name}.json and diagnostics_{args.name}.csv")

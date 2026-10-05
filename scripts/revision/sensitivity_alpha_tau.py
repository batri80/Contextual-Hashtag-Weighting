"""
sensitivity_alpha_tau.py
========================
Reviewer 1, comments 3, 4 and 6 - using the ORIGINAL pipeline (phase2_hashtag_weights.py)
unchanged: same weight function, same data, same K-means settings as the submitted paper.

  Comment 3: vary alpha1, alpha2, alpha3 (simplex grid) and recluster
  Comment 4: vary tau_dormant and tau_decay and recluster
  Comment 6: Shapiro-Wilk normality on the existing per-run scores (no new clustering)

Run inside the folder that holds the Phase 2 outputs of ONE dataset:
    train_with_weights.pkl, test_with_weights.pkl, pmi_matrix.pkl, temporal_data.pkl
    (optional) test_hashtag.npz          -> checks the default setting reproduces it
    (optional) bert_comparison_k20.pkl, clustering_results.pkl -> comment 6

Usage:
    python sensitivity_alpha_tau.py --name election2020
Outputs (folder sensitivity_<name>/):
    sensitivity_alpha.csv, sensitivity_tau.csv   mean +/- sd over seeds
    sensitivity_runs.csv                         every run
    normality_tests.csv                          comment 6
"""

import argparse
import bisect
import os
import pickle
import time
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="dataset")
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--alpha-step", type=float, default=0.25)
ap.add_argument("--tau-dormant-grid", default="1,3,7,14,30", help="days")
ap.add_argument("--tau-decay-grid", default="1,3,7,14", help="days")
ap.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
ap.add_argument("--sil-sample", type=int, default=10000)
args = ap.parse_args()
OUT = f"sensitivity_{args.name}"
os.makedirs(OUT, exist_ok=True)
DAY = 24 * 3600
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# ---------------------------------------------------------------------------
# Load the original Phase 2 inputs
# ---------------------------------------------------------------------------
log("Loading Phase 2 outputs")
train_df = pd.read_pickle("train_with_weights.pkl")
test_df = pd.read_pickle("test_with_weights.pkl").reset_index(drop=True)
log(f"  train {len(train_df):,}  test {len(test_df):,}")

# vocabulary exactly as phase2 build_hashtag_vocab(train_df, min_freq=10)
freq = Counter(h for hs in train_df["hashtags"] for h in hs)
vocab = [h for h, f in freq.items() if f >= 10]
vset = set(vocab)
col = {h: i for i, h in enumerate(vocab)}
log(f"  vocabulary {len(vocab):,}")

pmi_matrix = pickle.load(open("pmi_matrix.pkl", "rb"))["pmi"]

temporal = pickle.load(open("temporal_data.pkl", "rb"))
hashtag_history = temporal["history"]
hashtag_max_freq = temporal["max_freq"]
delta_T_sec = temporal.get("delta_T", 7 * DAY)

# hashtag embeddings: reuse phase2 file if intact, otherwise recompute identically
hashtag_embeddings = None
if os.path.exists("hashtag_embeddings.pkl"):
    obj = pickle.load(open("hashtag_embeddings.pkl", "rb"))
    if isinstance(obj, dict) and obj and isinstance(next(iter(obj)), str):
        hashtag_embeddings = obj
        log("  hashtag embeddings loaded from hashtag_embeddings.pkl")
if hashtag_embeddings is None:
    log("  hashtag_embeddings.pkl was overwritten - recomputing with the same model")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embedding_model)
    embs = model.encode([h.replace("_", " ") for h in vocab], batch_size=128,
                        show_progress_bar=False)
    hashtag_embeddings = dict(zip(vocab, embs))


# ---------------------------------------------------------------------------
# Weight function copied from phase2_hashtag_weights.py (unchanged)
# ---------------------------------------------------------------------------
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)


def safe_numeric_convert(value, default=0):
    if pd.isna(value) or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_contextual_weights_fast(row, alpha1, alpha2, alpha3, tau_dormant_sec, tau_decay_sec,
                                    beta1=0.5, beta2=0.5):
    hashtags_in_post = [h for h in row["hashtags"] if h in vset]
    if len(hashtags_in_post) == 0:
        return {}
    text_embedding = row["text_embedding"]
    timestamp = row["created_at"].timestamp()
    retweet_count = safe_numeric_convert(row.get("retweet_count", 0))
    likes_count = safe_numeric_convert(row.get("likes", 0))
    Gamma = np.log(1 + retweet_count + likes_count)

    energies = {}
    for h_i in hashtags_in_post:
        E_hashtag = 0.0
        if len(hashtags_in_post) > 1:
            V_sum, count = 0.0, 0
            for h_j in hashtags_in_post:
                if h_i == h_j:
                    continue
                V_pmi = pmi_matrix.get((h_i, h_j), 0.0)
                if h_i in hashtag_embeddings and h_j in hashtag_embeddings:
                    V_sem = (cosine_similarity(hashtag_embeddings[h_i], hashtag_embeddings[h_j]) + 1) / 2
                else:
                    V_sem = 0.0
                V_sum += beta1 * V_pmi + beta2 * V_sem
                count += 1
            if count > 0:
                E_hashtag = -alpha1 * V_sum / count

        E_text = 0.0
        if h_i in hashtag_embeddings:
            E_text = -alpha2 * (cosine_similarity(hashtag_embeddings[h_i], text_embedding) + 1) / 2

        E_temporal = 0.0
        if h_i in hashtag_history and len(hashtag_history[h_i]) > 0:
            times = hashtag_history[h_i]
            idx = bisect.bisect_left(times, timestamp)
            if idx > 0:
                delta_t = timestamp - times[idx - 1]
                if delta_t > tau_dormant_sec:
                    A_i = 0.0
                else:
                    start_idx = bisect.bisect_left(times, timestamp - delta_T_sec)
                    rho_i = (idx - start_idx) / hashtag_max_freq.get(h_i, 1)
                    A_i = rho_i * np.exp(-delta_t / tau_decay_sec)
                E_temporal = -alpha3 * A_i * Gamma

        energies[h_i] = E_hashtag + E_text + E_temporal

    exp_neg_E = {h: np.exp(-E) for h, E in energies.items()}
    Z = sum(exp_neg_E.values())
    return {h: v / Z for h, v in exp_neg_E.items()}


def feature_matrix(alpha, tau_dorm_days, tau_dec_days):
    rows, cols, data = [], [], []
    for i, row in enumerate(test_df.itertuples(index=False)):
        w = compute_contextual_weights_fast(row._asdict(), *alpha,
                                            tau_dorm_days * DAY, tau_dec_days * DAY)
        for h, v in w.items():
            rows.append(i)
            cols.append(col[h])
            data.append(v)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(test_df), len(vocab)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def db_ch(X, labels, C):
    n, k = X.shape[0], C.shape[0]
    x2 = np.asarray(X.multiply(X).sum(1)).ravel()
    mu = np.asarray(X.mean(0)).ravel()
    XC = np.asarray(X @ C.T)
    c2 = (C * C).sum(1)
    d = np.sqrt(np.maximum(x2 - 2 * XC[np.arange(n), labels] + c2[labels], 0))
    cnt = np.bincount(labels, minlength=k)
    ne = cnt > 0
    s = (np.bincount(labels, weights=d, minlength=k) / np.maximum(cnt, 1))[ne]
    Cn = C[ne]
    M = np.sqrt(np.maximum((Cn ** 2).sum(1)[:, None] + (Cn ** 2).sum(1)[None, :] - 2 * Cn @ Cn.T, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        R = (s[:, None] + s[None, :]) / M
    np.fill_diagonal(R, -np.inf)
    db = float(np.mean(np.max(R, axis=1)))
    kk = int(ne.sum())
    W = float((d ** 2).sum())
    B = float((cnt[ne] * ((Cn - mu) ** 2).sum(1)).sum())
    return db, (B / (kk - 1)) / (W / (n - kk))


def evaluate(X, label, params):
    out = []
    for s in range(args.seeds):
        seed = 42 + s
        km = KMeans(n_clusters=args.k, n_init=10, random_state=seed)
        lab = km.fit_predict(X)
        sil = silhouette_score(X, lab, sample_size=min(args.sil_sample, X.shape[0]), random_state=seed)
        db, ch = db_ch(X, lab, km.cluster_centers_)
        out.append({"experiment": label, **params, "seed": seed,
                    "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch})
    g = pd.DataFrame(out)
    log(f"  {label:6s} {params}  sil={g.silhouette.mean():.4f}±{g.silhouette.std():.4f}  "
        f"DB={g.davies_bouldin.mean():.3f}  CH={g.calinski_harabasz.mean():.1f}")
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
runs = []
DEFAULT_ALPHA = (0.33, 0.33, 0.33)
DEFAULT_TAU = (30, 7)

log("Default setting (as in the paper)")
X0 = feature_matrix(DEFAULT_ALPHA, *DEFAULT_TAU)
if os.path.exists("test_hashtag.npz"):
    ref = sparse.load_npz("test_hashtag.npz")
    same = (ref.shape[0] == X0.shape[0] and ref.nnz == X0.nnz and
            np.allclose(np.sort(ref.data), np.sort(X0.data), atol=1e-6))
    log(f"  reproduces submitted test_hashtag.npz: {same}")
runs += evaluate(X0, "alpha", {"a1": 0.33, "a2": 0.33, "a3": 0.33,
                               "tau_dormant": DEFAULT_TAU[0], "tau_decay": DEFAULT_TAU[1]})

log("Comment 3: alpha grid (tau fixed at 30 / 7 days)")
step = args.alpha_step
grid = [round(i * step, 4) for i in range(int(round(1 / step)) + 1)]
for a1 in grid:
    for a2 in grid:
        a3 = round(1 - a1 - a2, 4)
        if a3 < 0:
            continue
        X = feature_matrix((a1, a2, a3), *DEFAULT_TAU)
        runs += evaluate(X, "alpha", {"a1": a1, "a2": a2, "a3": a3,
                                      "tau_dormant": DEFAULT_TAU[0], "tau_decay": DEFAULT_TAU[1]})

log("Comment 4: tau grid (alpha fixed at 0.33 each)")
for td in [float(x) for x in args.tau_dormant_grid.split(",")]:
    for tc in [float(x) for x in args.tau_decay_grid.split(",")]:
        X = feature_matrix(DEFAULT_ALPHA, td, tc)
        runs += evaluate(X, "tau", {"a1": 0.33, "a2": 0.33, "a3": 0.33,
                                    "tau_dormant": td, "tau_decay": tc})

res = pd.DataFrame(runs)
res.to_csv(f"{OUT}/sensitivity_runs.csv", index=False)
keys = ["a1", "a2", "a3", "tau_dormant", "tau_decay"]
for exp in ["alpha", "tau"]:
    agg = (res[res.experiment == exp]
           .groupby(keys)[["silhouette", "davies_bouldin", "calinski_harabasz"]]
           .agg(["mean", "std"]))
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.reset_index().to_csv(f"{OUT}/sensitivity_{exp}.csv", index=False)

# ---------------------------------------------------------------------------
# Comment 6: normality of existing per-run scores (no new clustering)
# ---------------------------------------------------------------------------
log("Comment 6: normality tests on existing per-run scores")
nrows = []


def add(name, x):
    x = np.asarray(x, float)
    if len(x) >= 3 and np.ptp(x) > 0:
        nrows.append({"scores": name, "n": len(x), "mean": x.mean(), "sd": x.std(ddof=1),
                      "shapiro_W": stats.shapiro(x).statistic, "shapiro_p": stats.shapiro(x).pvalue})


if os.path.exists("bert_comparison_k20.pkl"):
    bc = pickle.load(open("bert_comparison_k20.pkl", "rb"))
    h, b = bc["hashtag"]["scores"], bc["bert"]["scores"]
    add("Table 3: Hashtag-Only K=20", h)
    add("Table 3: BERT K=20", b)
    nrows.append({"scores": "Table 3: Hashtag-Only vs BERT", "n": len(h),
                  "paired_t_p": stats.ttest_rel(h, b).pvalue,
                  "welch_t_p": stats.ttest_ind(h, b, equal_var=False).pvalue,
                  "wilcoxon_p": stats.wilcoxon(h, b).pvalue,
                  "mannwhitney_p": stats.mannwhitneyu(h, b, alternative="two-sided").pvalue})
if os.path.exists("clustering_results.pkl"):
    cr = pickle.load(open("clustering_results.pkl", "rb"))
    for key in ["hashtag", "tfidf", "combined", "bert"]:
        if key in cr and "runs" in cr[key]:
            add(f"Phase 6: {key}", [r["silhouette"] for r in cr[key]["runs"]])
for name, g in res.groupby(keys):
    if name == (0.33, 0.33, 0.33, 30, 7):
        add("Sensitivity default (0.33,0.33,0.33; 30,7)", g.silhouette.values)
pd.DataFrame(nrows).to_csv(f"{OUT}/normality_tests.csv", index=False)

log(f"Done. Results in {OUT}/")

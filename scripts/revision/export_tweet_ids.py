"""
export_tweet_ids.py
===================
Writes the tweet identifiers of the chronological train/validation/test splits,
so that the splits can be rebuilt without redistributing tweet text (X/Twitter
terms of use allow sharing IDs, not text).

Run inside a dataset folder that holds the Phase 1/2 outputs:
    train_with_weights.pkl, val_with_weights.pkl, test_with_weights.pkl
(or train_data.pkl, val_data.pkl, test_data.pkl).

Usage:
    python export_tweet_ids.py --name election2020
    python export_tweet_ids.py --name covid19

Output: results/<name>/tweet_ids_<split>.csv with columns
    split, position (chronological order within the split), tweet_id, created_at
"""
import argparse
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--out", default="results")
args = ap.parse_args()

ID_COLUMNS = ["tweet_id", "id", "id_str", "status_id"]
out_dir = os.path.join(args.out, args.name)
os.makedirs(out_dir, exist_ok=True)

for split in ["train", "val", "test"]:
    path = next((p for p in [f"{split}_with_weights.pkl", f"{split}_data.pkl"]
                 if os.path.exists(p)), None)
    if path is None:
        raise SystemExit(f"No pickle found for the {split} split in {os.getcwd()}")
    df = pd.read_pickle(path)
    col = next((c for c in ID_COLUMNS if c in df.columns), None)
    if col is None:
        raise SystemExit(f"{path}: none of {ID_COLUMNS} found; columns are {list(df.columns)}")
    ids = pd.DataFrame({
        "split": split,
        "position": range(len(df)),
        "tweet_id": df[col].astype("int64").astype(str),
        "created_at": pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M:%S"),
    })
    target = os.path.join(out_dir, f"tweet_ids_{split}.csv")
    ids.to_csv(target, index=False)
    print(f"{split:5s}: {len(ids):>9,} ids  ->  {target}")

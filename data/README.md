# Data

Tweet text is not included in this repository, in line with X/Twitter's terms
of use. Download the two public collections and place the files in a separate
working folder for each dataset.

| Dataset | Source | Files used by `phase1_data_prep.py` |
|---|---|---|
| US Election 2020 | Kaggle collection of tweets with #DonaldTrump and #JoeBiden (cited in the paper as Hui, Kaggle) | `hashtag_joebiden.csv` and `hashtag_donaldtrump.csv`; set `biden_file` and `trump_file` at the top of the script |
| COVID-19 | Kaggle COVID-19 tweets collection (cited in the paper as Chakraborty, Kaggle) | the collection's CSV file |

## Why tweet IDs are not provided

The Election 2020 collection stores tweet IDs as floating-point numbers (for
example `1.316529221557252e+18`), which cannot hold the full 19 digits of a
tweet ID, and the COVID-19 files we processed hold them rounded in the same
way. The IDs in our split files therefore cannot be used to retrieve tweets.

Instead, the splits are rebuilt deterministically from the collections:
`pipeline/phase1_data_prep.py` applies the same filters (at least two hashtags,
more than 10 characters of cleaned text), sorts by timestamp and cuts the data
70/15/15 without shuffling. The resulting split sizes and periods are given in
Table 1 of the paper and in `results/election2020/diagnostics_summary.json` and
`results/covid19/split_stats.csv`, so a rebuilt split can be checked against them.

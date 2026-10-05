# Data

Tweet text is not included in this repository. X/Twitter's terms of use allow
sharing tweet identifiers but not the text. Download the two public collections
and place the files in a separate working folder for each dataset.

| Dataset | Source | Files used by `phase1_data_prep.py` |
|---|---|---|
| US Election 2020 | Kaggle collection of tweets with #DonaldTrump and #JoeBiden (cited in the paper as Hui, Kaggle) | the Biden and Trump CSV files; set `biden_file` and `trump_file` at the top of the script |
| COVID-19 | Kaggle COVID-19 tweets collection (cited in the paper as Chakraborty, Kaggle) | the collection's CSV file |

After Phase 1, `scripts/revision/export_tweet_ids.py` writes the tweet
identifiers of each split, in chronological order, to
`results/<dataset>/tweet_ids_<split>.csv`. Together with the fixed seeds,
these identifiers are enough to rebuild the exact splits from the original
collections.

# Contextual Hashtag Weighting: Zero-Energy Framework

Code and results accompanying "Contextual Hashtag Weighting for Social Media
Clustering: A Zero-Energy Framework Evaluated Across Two Domains"
(submitted to *Social Network Analysis and Mining*).

## Current state (read this first)

**The paper describes the code as it actually runs.** While revising the
paper we checked every description in it against these scripts and corrected
the descriptions, not the code. The main corrections are:

- The hashtag and text embeddings come from the sentence-transformer
  `all-MiniLM-L6-v2` (384 dimensions), not from GloVe.
- The split is chronological 70/15/15 (train/validation/test), not 80/20.
- The temporal energy is scaled by an engagement factor
  `Gamma_d = ln(1 + retweets + likes)`.
- The data are not de-duplicated.

**The contextual weights perform on par with uniform weights.** On
Election 2020 the Boltzmann weights stay within about 0.014 of `1/|H_d|` on
average. Replacing them with uniform weights gives Silhouette 0.149 ± 0.004,
against 0.143 ± 0.007 for the energy weights (Welch p = 0.14). The gains
reported in the paper are therefore attributed mainly to the hashtag
representation. See `results/election2020/diagnostics_summary.json`.

## Repository structure

```
pipeline/                 The original experimental pipeline, run once per dataset
  phase1_data_prep.py       Cleaning, filtering (>=2 hashtags, >10 characters), 70/15/15 split
  phase2_hashtag_weights.py Vocabulary, PMI, embeddings, temporal activity, energy weights
  phase3_tfidf.py           TF-IDF features (5,000 terms, uni/bigrams, min_df 5, max_df 0.8)
  phase4_combine_features.py Hashtag matrices and combined hashtag+TF-IDF features
  fix_vocab_indices.py      Rebuilds contiguous vocabulary indices (run before Phase 4)
  phase5_optimal_k.py ... phase10_report.py   Clustering, analysis, BERT baseline, report
  enhanced_phase6.py        K grid {2,5,10,15,20} with Silhouette, DB and CH (Tables 2-3)
  bridge_npz_to_pkl.py      Converts sparse .npz features to dense .pkl where needed
scripts/
  experiments/              Ablation (Table 5), BERT comparison at K=20 (Table 4),
                            top hashtags per cluster
  revision/                 Scripts added during the revision: diagnostics,
                            sensitivity analysis (Tables 6-7), COVID-19 splits and
                            ablation, statistical checks for Tables 4, 6 and 7
results/
  election2020/             Result files behind Tables 4, 6, 7 and Section 5.5
  covid19/                  Result files for COVID-19 (see the README there)
manuscript/                 Revised manuscript (LaTeX), bibliography, response letter
docs/
  TABLE_FIGURE_MAPPING.md   Maps every table and reported number to its script and file
data/
  README.md                 Where to obtain the two datasets and where to put them
```

## Reproducing the paper

1. Obtain the datasets (`data/README.md`). Tweet text and tweet IDs are not
   redistributed here (see that file for why).
2. In a folder for each dataset, run the pipeline in order:
   `phase1` → `phase2` → `phase3` → `fix_vocab_indices` → `phase4` →
   `enhanced_phase6` → `phase9_bert`.
3. Run the scripts in `scripts/experiments/` and `scripts/revision/` from the
   same folder. Each script's docstring states its inputs, outputs and protocol.
4. `docs/TABLE_FIGURE_MAPPING.md` gives the script and output file for every
   table and for each number quoted in the text.

All clustering uses K-means (k-means++, `n_init=10`, 300 iterations). Multi-seed
experiments use seeds 42–46, and Silhouette is computed on a 10,000-point sample.

**One caution.** These scripts were developed alongside another project, and an
unrelated script (`preprocess_clustering.py`, not included) overwrites
`hashtag_vocab.pkl` with a different format if run in the same folder. Run this
pipeline in its own folder. `revision_diagnostics.py` warns if it finds the
overwritten file.

## Requirements

Python 3.10 or later and the packages in `requirements.txt`. Embedding
computation (Phase 2 and the sensitivity analysis) and the BERT baseline
(Phase 9) are the slow steps. Everything else runs in minutes on a laptop.

## Citation

If you use this code, please cite the paper. Details are in the manuscript's
Data availability section.
